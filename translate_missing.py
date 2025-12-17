#!/usr/bin/env python3
"""
Translation automation script for locale files.
Compares all language files against English baseline and auto-translates missing entries.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote, unquote

import aiohttp
from dataclasses import dataclass

# Configuration
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", 5))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
BASE_LANGUAGE = "en"
TRANSLATE_MISSING = os.environ.get("TRANSLATE_MISSING", 'false').lower() == 'true'  # Whether to translate existing empty keys

# Language code to full name mapping
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ro": "Romanian",
    "tr": "Turkish",
    "zh_cn": "Simplified Chinese",
    "zh_tw": "Traditional Chinese",
}


@dataclass
class TranslationEntry:
    """Represents a single translation entry."""
    key_path: Tuple[str]
    english_text: str
    translated_text: str = ""

    @property
    def idx(self) -> str:
        return self.get_idx(self.key_path)

    @staticmethod
    def get_idx(key_path: Tuple[str]) -> str:
        return '.'.join(quote(i) for i in key_path)

    @staticmethod
    def decode_idx(idx: str) -> tuple:
        return tuple(unquote(i) for i in idx.split('.'))


@dataclass
class FileTranslationBatch:
    """Represents all translations for a single file."""
    language: str
    file_name: str
    translations: List[TranslationEntry]
    successful_translations: int = 0


class TranslationManager:
    """Manages translation operations with concurrent API requests."""

    def __init__(self, base_dir: Path, api_key: str):
        self.base_dir = base_dir
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.translation_count = 0
        self.error_count = 0

    def get_all_language_dirs(self) -> List[str]:
        """Get all language directories except the base language."""
        return [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(self.base_dir / d) and d != BASE_LANGUAGE and d in LANGUAGE_NAMES.keys()
        ]

    def get_all_json_files(self, lang_dir: str) -> List[str]:
        """Get all JSON files in a language directory."""
        lang_path = self.base_dir / lang_dir
        return [f for f in os.listdir(lang_path) if f.endswith(".json") and '.missing' not in f]

    def load_json(self, file_path: Path) -> Dict:
        """Load JSON file with error handling."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, file_path: Path, data: Dict):
        """Save JSON file with proper formatting."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_nested_value(self, data: Dict, key_path: Tuple[str]):
        """Get value from nested dictionary using key path."""
        current = data
        for key in key_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def set_nested_value(self, data: Dict, key_path: Tuple[str], value: str):
        """Set value in nested dictionary using key path."""
        current = data
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[key_path[-1]] = value

    def find_missing_keys(
            self, base_data: Dict, target_data: Dict, key_path: Tuple[str] = None
    ) -> List[Tuple[str]]:
        """Recursively find missing keys in target compared to base."""
        if key_path is None:
            key_path = tuple()

        missing_keys = []

        for key, value in base_data.items():
            current_path: tuple = (*key_path, key)

            if key not in target_data:
                if isinstance(value, dict):
                    missing_keys.extend(self.find_missing_keys(value, {}, current_path))
                else:
                    missing_keys.append(current_path)
            elif isinstance(value, list):
                if isinstance(target_data.get(key), list):
                    if len(value) > len(target_data[key]):
                        missing_keys.append(current_path)
                else:
                    print('Type mismatch for key:', current_path)
                    missing_keys.append(current_path)
            elif isinstance(value, dict):
                if isinstance(target_data.get(key), dict):
                    # Recursively check nested dictionaries
                    missing_keys.extend(
                        self.find_missing_keys(value, target_data[key], current_path)
                    )
                else:
                    # Target has a value but it should be a dict
                    missing_keys.extend(self.find_missing_keys(value, {}, current_path))
            else:
                if TRANSLATE_MISSING:
                    # If translation of existing keys is enabled, check if value is empty
                    if not bool(target_data.get(key)):
                        missing_keys.append(current_path)

        return missing_keys

    async def translate_file_batch_mock(
            self, session: aiohttp.ClientSession, batch: FileTranslationBatch
    ) -> None:
        async with self.semaphore:
            for entry in batch.translations:
                entry.translated_text = f'Translation for {entry.key_path} in {batch.language}, [{entry.english_text=}]'
                batch.successful_translations += 1

    async def translate_file_batch(
            self, session: aiohttp.ClientSession, batch: FileTranslationBatch
    ) -> None:
        """Translate all missing keys for a file in a single API request."""
        async with self.semaphore:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Build the translation request with all keys
            entries_text = ""
            for i, entry in enumerate(batch.translations, 1):
                entries_text += f"\n{i}. Key: {entry.idx}\n   Text: {entry.english_text}\n"

            prompt = f"""Translate the following JSON locale entries to {LANGUAGE_NAMES.get(batch.language, batch.language)}.
Preserve all HTML tags, code blocks, and special formatting exactly as they appear.
Only translate the actual text content, not the HTML tags, code, or technical terms.

For each entry, provide the translation in the following format:
<translation key="title.name">translated text here</translation>

Entries to translate:{entries_text}

Provide all translations:"""

            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate all entries while preserving all HTML tags, code blocks, and formatting. Return translations in the exact XML format requested.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            }

            try:
                async with session.post(
                        OPENAI_URL,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result["choices"][0]["message"]["content"].strip()

                        # Parse the translations from the response
                        pattern = r'<translation key="(.+)">([\s\S]*?)</translation>'
                        matches = dict(re.findall(pattern, response_text))

                        for entry in batch.translations:
                            entry.translated_text = matches.get(entry.idx, '')
                            if entry.translated_text:
                                batch.successful_translations += 1

                    else:
                        error_text = await response.text()
                        print(f"API Error {response.status}: {error_text}")
                        self.error_count += 1
            except Exception as e:
                print(f"Translation error: {e}")
                self.error_count += 1

    async def process_file_batch(
            self, session: aiohttp.ClientSession, batch: FileTranslationBatch
    ) -> int:
        """Process all translations for a single file."""
        print(
            f"Translating {batch.language}/{batch.file_name} ({len(batch.translations)} entries)"
        )

        # await self.translate_file_batch_mock(session, batch)
        await self.translate_file_batch(session, batch)

        if batch.successful_translations > 0:
            # Ensure target directory exists
            target_dir = self.base_dir / batch.language
            target_dir.mkdir(parents=True, exist_ok=True)

            # Load current target file or create empty dict
            target_file = target_dir / batch.file_name
            try:
                target_data = self.load_json(target_file)
            except json.JSONDecodeError:
                print(f"\tError: Invalid JSON in {target_file}, skipping.")
                return 0

            # Set all translated values
            success_count = 0
            for entry in batch.translations:
                if entry.translated_text:
                    self.set_nested_value(target_data, entry.key_path, entry.translated_text)
                    success_count += 1
                else:
                    print(f"\tWarning: Missing translation for {entry.idx}")

            # Save updated file
            self.save_json(target_file, target_data)

            self.translation_count += success_count
            return success_count
        return 0

    async def translate_all_missing(self):
        """Main translation workflow."""
        if not self.api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)

        print("Scanning for missing translations...")

        # Collect all translation batches grouped by file
        # batches_by_file: Dict[Tuple[str, str], List[Tuple[Tuple[str], str]]] = defaultdict(list)
        batches: List[FileTranslationBatch] = []
        languages = self.get_all_language_dirs()
        total_missing = 0

        for lang in languages:
            print(f"Checking language: {lang}")
            json_files = self.get_all_json_files(BASE_LANGUAGE)

            for json_file in json_files:
                base_file = self.base_dir / BASE_LANGUAGE / json_file
                target_file = self.base_dir / lang / json_file

                # Load base English file
                base_data = self.load_json(base_file)
                if not base_data:
                    continue

                # Load target language file or create empty dict
                try:
                    target_data = self.load_json(target_file)
                except FileNotFoundError:
                    target_data = {}
                except json.JSONDecodeError:
                    print(f"\tError: Invalid JSON in {target_file}, skipping.")
                    continue

                # Find missing keys
                missing_keys: list[tuple[str]] = self.find_missing_keys(base_data, target_data)

                batch = FileTranslationBatch(
                    language=lang,
                    file_name=json_file,
                    translations=[],
                )
                # Group translations by file
                for key_path in missing_keys:
                    english_text = self.get_nested_value(base_data, key_path)
                    if english_text and isinstance(english_text, str):
                        batch.translations.append(TranslationEntry(key_path=key_path, english_text=english_text))

                if batch.translations:
                    batches.append(batch)
                    total_missing += len(batch.translations)

        if not batches:
            print("✓ No missing translations found!")
            return


        print(f"\nFound {total_missing} missing translations in {len(batches)} files across {len(languages)} languages")
        print(f"Processing with max {MAX_CONCURRENT_REQUESTS} concurrent file requests...\n")

        # Process all file batches concurrently
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[self.process_file_batch(session, batch) for batch in batches],
                return_exceptions=True,
            )

        # Summary
        print("\n" + "=" * 60)
        print(f"✓ Completed: {self.translation_count} translations")
        if self.error_count > 0:
            print(f"✗ Errors: {self.error_count}")
        print("=" * 60)


async def main():
    """Entry point."""
    script_dir = Path(__file__).parent
    manager = TranslationManager(script_dir, OPENAI_API_KEY)
    await manager.translate_all_missing()


if __name__ == "__main__":
    asyncio.run(main())
