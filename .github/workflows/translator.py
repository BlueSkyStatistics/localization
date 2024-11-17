# pip install aiohttp
import argparse
import sys
import json
import asyncio
from datetime import datetime
from collections import defaultdict
from itertools import chain
from pathlib import Path
from traceback import format_exc
from typing import Dict, List, Tuple
from csv import DictWriter
import aiohttp
import logging

parser = argparse.ArgumentParser(
    prog='Translator',
    description='Translator for new/changed .json files',
)
parser.add_argument('--new_files', help='List of new files', default='')
parser.add_argument('--changed_files', help='List of changed files', default='')
parser.add_argument('--concurrent_tasks', help='Number of concurrent translation tasks', type=int, default=5)
parser.add_argument('--max_langs_per_request', help='Maximum languages per one prediction request', type=int, default=2)

parser.add_argument('--token', help='Project id in alita', required=True, type=str)
parser.add_argument('--llm_url', help='Llm prediction provider url', required=True, type=str)

parser.add_argument('--alita_project_id', help='Project id in alita', required=True, type=int)
parser.add_argument('--prompt_version_id', help='Prediction prompt version id in alita', required=True, type=int)

parser.add_argument('--lang_map', help='A folder-name mapping of languages', required=True)

# args = parser.parse_args([
#     '--alita_project_id', '5',
#     '--llm_url', 'https://eye.projectalita.ai/main',
#     '--prompt_version_id', '123',
#     '--lang_map', '''{
#   "de": "german",
#   "es": "spanish",
#   "fr": "french",
#   "ro": "romanian",
#   "ru": "russian",
#   "zh_cn": "simplified chinese",
#   "zh_tw": "traditional chinese"
# }'''
# ])
args = parser.parse_args()
args.llm_url = args.llm_url.rstrip("/")
args.lang_map = json.loads(args.lang_map)

# Constants
TMP_DIR = Path('.tmp')
TMP_DIR.mkdir(parents=True, exist_ok=True)
PREDICT_URL = f'{args.llm_url}/api/v1/prompt_lib/predict/prompt_lib/{args.alita_project_id}/{args.prompt_version_id}'

# Logger setup
logging.basicConfig(filename=TMP_DIR / f'_log.log', encoding='utf-8', level=logging.DEBUG,
                    format='%(levelname)s\t-\t%(message)s')
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(console_handler)

exec_log = defaultdict(dict)


def deep_merge(base_dict: dict, updater_dict: dict, in_place: bool = False) -> dict:
    if in_place:
        result = base_dict
    else:
        result = base_dict.copy()
    for key, value in updater_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value, in_place=in_place)
        else:
            result[key] = value
    return result


def filter_existing(file_name: str, lang_list: List[dict], dest_dir: Path = Path()) -> List[dict]:
    """Filter existing files to avoid redundant processing."""
    exec_log[file_name].update({i['key']: False for i in lang_list})
    exec_log[file_name]['file_name'] = file_name
    result = []
    for i in lang_list:
        if dest_dir.joinpath(i['key'], file_name).exists():
            logger.info(f"\t{i['key']}/{file_name} exists, skipping")
            exec_log[file_name][i['key']] = 'existed'
        else:
            result.append(i)
    return result


async def fetch_and_save(session: aiohttp.ClientSession, file_path: Path, lang_list: List[dict], tmp_dir: Path):
    """Fetch translations and save them."""
    if file_path.suffix != '.json' or tmp_dir.joinpath(file_path.name).exists():
        logger.warning(f'\tSkipping fetch: {file_path.name} (invalid or already exists)')
        return

    filtered_langs = filter_existing(file_path.name, lang_list)
    if not filtered_langs:
        logger.info(f'\tAll translations exist for: {file_path.name}, langs: {[i["key"] for i in lang_list]}')
        return

    logger.info(
        f'\tFetching: {file_path.name}, langs: {[i["key"] for i in lang_list]}, filtered: {[i["key"] for i in filtered_langs]}')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = json.load(f)
    except json.JSONDecodeError:
        logger.error(f'\tCould not parse original json: {file_path}')
        return

    payload = {'user_input': json.dumps({'languages': filtered_langs, 'original_json': contents})}
    print(payload)
    for i in lang_list:
        exec_log[file_path.name][i['key']] = 'MOCKED'
    return

    try:
        async with session.post(PREDICT_URL, json=payload) as response:
            if response.ok:
                resp = await response.json()
                dest_file_path = tmp_dir.joinpath(file_path.name)
                with open(dest_file_path, 'w', encoding='utf-8') as out:
                    json.dump(resp, out, ensure_ascii=False)
                for i in lang_list:
                    if exec_log[file_path.name][i['key']] is False:
                        exec_log[file_path.name][i['key']] = True
                logger.info(f'Fetching done for: {file_path.name}')
            else:
                logger.error(f'Failed to fetch translation for {file_path.name}: {response.status}')
    except Exception as e:
        logger.error(f'Error fetching {file_path.name}: {format_exc()}')


async def translate(languages: Dict[str, str],
                    token: str,
                    file_list: list,
                    tmp_dir: Path = TMP_DIR,
                    concurrent_tasks: int = 5,
                    max_langs_per_request: int = 2,
                    ) -> None:
    """Main translation function."""
    logger.info('Translating...')
    tmp_dir.mkdir(exist_ok=True)
    lang_list = [{'language': v, 'key': k} for k, v in languages.items()]

    async with aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}) as session:
        sem = asyncio.Semaphore(concurrent_tasks)

        async def bound_fetch(file_path: Path, lang_chunk: list):
            async with sem:
                await fetch_and_save(session, file_path, lang_chunk, tmp_dir)

        tasks = []
        for f in file_list:
            # f = Path(f)
            # if f.exists() and f.suffix == '.json':
            for i in range(0, len(lang_list), max_langs_per_request):
                chunk = lang_list[i:i + max_langs_per_request]
                tasks.append(bound_fetch(Path(f), chunk))

        await asyncio.gather(*tasks)

    with open(tmp_dir / f'_exec_log.csv', 'w', encoding='utf-8') as out:
        writer = DictWriter(out, ['file_name', *languages.keys()])
        writer.writeheader()
        writer.writerows(exec_log.values())
    logger.info('Translation done')


def extract_json(json_string: str) -> dict:
    """Extract JSON object from a string."""
    json_start = json_string.find('{')
    json_end = json_string.rfind('}') + 1
    return json.loads(json_string[json_start:json_end])


class FileStructureError(Exception):
    """Custom exception for file structure errors."""
    pass


def fix_corrupted_structure(data: dict, languages: Dict[str, str]) -> dict:
    """Fix corrupted JSON structures."""
    fixed_dict = defaultdict(dict)
    for k, langs in data.items():
        if not set(langs.keys()).issubset(set(languages.keys())):
            if isinstance(k, dict):
                langs = fix_corrupted_structure(langs, languages)
            logger.debug(f'Key: {k=}, {langs.keys()=}, missing={set(languages.keys()).difference(langs.keys())}')
            raise FileStructureError(data)
        for lng, v in langs.items():
            fixed_dict[lng][k] = v
    return fixed_dict


def extract_data(folder_path: Path, languages: Dict[str, str], force_overwrite: bool = False, dest_dir: Path = Path()):
    """Extract data from JSON files."""
    logger.info('Extracting...')
    for lang_code in languages.keys():
        (dest_dir / lang_code).mkdir(exist_ok=True)

    for file_path in folder_path.iterdir():
        if file_path.suffix != '.json':
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f'Extracting: {file_path.name}')
            content = extract_json(data['messages'][0]['content'])

            if not set(content.keys()).issubset(set(languages.keys())):
                logger.debug(f'\tFixing {file_path.name}')
                content = fix_corrupted_structure(content, languages)
                data['messages'][0]['content'] = json.dumps(content, ensure_ascii=False)
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            for lang, translation in content.items():
                dest_file_path = dest_dir.joinpath(lang, file_path.name)
                if dest_file_path.exists():
                    if force_overwrite:
                        logger.warning(f'\tFile exists, overwriting: {dest_file_path}')
                    else:
                        with open(dest_file_path, 'r', encoding='utf-8') as f:
                            src = json.load(f)
                        translation = deep_merge(src, translation)

                with open(dest_file_path, 'w', encoding='utf-8') as f:
                    json.dump(translation, f, ensure_ascii=False, indent=2)

        except json.JSONDecodeError as e:
            logger.error(f'\tCannot read file: {file_path}, {e}')
        except FileStructureError as e:
            logger.error(f'\tFile structure might be corrupted: {file_path}\n{e}')
        except Exception as e:
            logger.error(f'\tUnhandled error: {file_path}\n{e}')

    logger.info('Extraction done')


if __name__ == '__main__':
    new_files = [Path(i.strip()) for i in args.new_files.split('\n') if str(i).startswith('en/')]
    changed_files = [Path(i.strip()) for i in args.changed_files.split('\n') if str(i).startswith('en/')]

    print(new_files)
    print(changed_files)

    asyncio.run(translate(
        languages=args.lang_map,
        tmp_dir=TMP_DIR,
        token=args.token,
        file_list=list(set(chain(new_files, changed_files))),
    ))
    # extract_data(TMP_DIR, languages=lang_map)
