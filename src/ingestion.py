"""Document ingestion and chunking for the knowledge base."""

import os
import csv
import json
import email
from email import policy
from pathlib import Path
from dataclasses import dataclass
from typing import Generator
from datetime import datetime
import hashlib

from pypdf import PdfReader
from docx import Document as DocxDocument


@dataclass
class Chunk:
    """A chunk of text from a document."""
    content: str
    source: str
    chunk_id: str
    metadata: dict


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {
        # Documents
        '.pdf', '.md', '.txt', '.docx',
        # Emails
        '.eml',
        # Data exports
        '.csv', '.json'
    }

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> str | list[dict]:
        """Load content from a file. Returns string for docs, list for structured data."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == '.pdf':
            return self._load_pdf(path)
        elif ext == '.docx':
            return self._load_docx(path)
        elif ext in {'.md', '.txt'}:
            return self._load_text(path)
        elif ext == '.eml':
            return self._load_email(path)
        elif ext == '.csv':
            return self._load_csv(path)
        elif ext == '.json':
            return self._load_json(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF."""
        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)

    def _load_docx(self, path: Path) -> str:
        """Extract text from DOCX."""
        doc = DocxDocument(str(path))
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())

    def _load_text(self, path: Path) -> str:
        """Load plain text or markdown."""
        return path.read_text(encoding='utf-8')

    def _load_email(self, path: Path) -> str:
        """Extract content from .eml email file."""
        with open(path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        # Extract headers
        from_addr = msg.get('From', 'Unknown')
        to_addr = msg.get('To', 'Unknown')
        subject = msg.get('Subject', 'No Subject')
        date_str = msg.get('Date', '')

        # Try to parse date
        try:
            date_obj = email.utils.parsedate_to_datetime(date_str)
            date_formatted = date_obj.strftime('%Y-%m-%d %H:%M')
        except:
            date_formatted = date_str

        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    try:
                        body = part.get_content()
                        break
                    except:
                        pass
                elif content_type == 'text/html' and not body:
                    try:
                        # Fallback to HTML if no plain text
                        html_content = part.get_content()
                        # Basic HTML stripping
                        import re
                        body = re.sub(r'<[^>]+>', ' ', html_content)
                        body = ' '.join(body.split())
                    except:
                        pass
        else:
            try:
                body = msg.get_content()
            except:
                body = str(msg.get_payload(decode=True) or '')

        # Format as structured text for better context
        email_text = f"""EMAIL
From: {from_addr}
To: {to_addr}
Date: {date_formatted}
Subject: {subject}

{body}"""

        return email_text

    def _load_csv(self, path: Path) -> list[dict]:
        """Load CSV file as list of records."""
        records = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            # Try to detect delimiter
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except:
                dialect = 'excel'

            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                # Filter out empty values
                record = {k: v for k, v in row.items() if v and v.strip()}
                if record:
                    records.append(record)

        return records

    def _load_json(self, path: Path) -> list[dict]:
        """Load JSON file as list of records."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            # Check if it's a wrapper with a data array
            for key in ['data', 'records', 'items', 'results', 'entries']:
                if key in data and isinstance(data[key], list):
                    return [r for r in data[key] if isinstance(r, dict)]
            # Single record
            return [data]
        else:
            return []

    def _format_record(self, record: dict, source: str, index: int) -> str:
        """Format a data record as readable text."""
        lines = [f"RECORD #{index + 1} from {Path(source).name}"]

        for key, value in record.items():
            if value is None:
                continue
            # Clean up the key
            clean_key = key.replace('_', ' ').replace('-', ' ').title()
            # Format value
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            lines.append(f"{clean_key}: {value}")

        return "\n".join(lines)

    def chunk_text(self, text: str, source: str, extra_metadata: dict = None) -> Generator[Chunk, None, None]:
        """Split text into overlapping chunks."""
        # Clean up whitespace
        text = ' '.join(text.split())

        base_metadata = extra_metadata or {}

        # Create unique ID prefix including record index if present
        id_prefix = source
        if 'record_index' in base_metadata:
            id_prefix = f"{source}:r{base_metadata['record_index']}"

        if len(text) <= self.chunk_size:
            chunk_id = hashlib.md5(f"{id_prefix}:0:{text[:50]}".encode()).hexdigest()[:12]
            yield Chunk(
                content=text,
                source=source,
                chunk_id=chunk_id,
                metadata={"chunk_index": 0, "total_length": len(text), **base_metadata}
            )
            return

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '! ', '? ', '\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.chunk_size // 2:
                        end = last_sep + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = hashlib.md5(f"{id_prefix}:{chunk_index}:{chunk_text[:50]}".encode()).hexdigest()[:12]
                yield Chunk(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    metadata={"chunk_index": chunk_index, "start": start, "end": end, **base_metadata}
                )
                chunk_index += 1

            start = end - self.chunk_overlap

    def process_file(self, file_path: str) -> Generator[Chunk, None, None]:
        """Load and chunk a single file."""
        path = Path(file_path)
        ext = path.suffix.lower()

        content = self.load_file(file_path)

        # Handle structured data (CSV/JSON) differently
        if isinstance(content, list):
            for i, record in enumerate(content):
                record_text = self._format_record(record, file_path, i)
                # Each record becomes its own chunk(s)
                yield from self.chunk_text(
                    record_text,
                    file_path,
                    extra_metadata={"record_index": i, "type": ext[1:]}
                )
        else:
            # Regular text content
            extra_meta = {}
            if ext == '.eml':
                extra_meta["type"] = "email"
            yield from self.chunk_text(content, file_path, extra_metadata=extra_meta)

    def process_directory(self, dir_path: str) -> Generator[Chunk, None, None]:
        """Process all supported files in a directory."""
        path = Path(dir_path)
        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    yield from self.process_file(str(file_path))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


def get_supported_files(dir_path: str) -> list[str]:
    """Get list of supported files in a directory."""
    path = Path(dir_path)
    files = []
    for file_path in path.rglob('*'):
        if file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
            files.append(str(file_path))
    return sorted(files)
