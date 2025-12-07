"""Google Drive integration for Vault.

This module provides real Google Drive API integration to sync documents
into the Vault knowledge base.

Setup:
1. Create a Google Cloud project
2. Enable Google Drive API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download credentials.json to ~/.vault/credentials.json
5. Run: vault sync drive
"""

import os
import io
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Generator
from dataclasses import dataclass

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False


# OAuth scopes for Drive read-only access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Default paths
CONFIG_DIR = Path.home() / '.vault'
CREDENTIALS_FILE = CONFIG_DIR / 'credentials.json'
TOKEN_FILE = CONFIG_DIR / 'drive_token.json'

# Supported MIME types and their extensions
SUPPORTED_MIME_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'text/plain': '.txt',
    'text/markdown': '.md',
    'text/csv': '.csv',
    'application/json': '.json',
    # Google Docs - will be exported
    'application/vnd.google-apps.document': '.docx',
    'application/vnd.google-apps.spreadsheet': '.csv',
}

# Export MIME types for Google Docs
EXPORT_MIME_TYPES = {
    'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.google-apps.spreadsheet': 'text/csv',
}


@dataclass
class DriveFile:
    """Represents a Google Drive file."""
    id: str
    name: str
    mime_type: str
    modified_time: datetime
    size: int
    parents: list[str]
    web_link: str

    @property
    def extension(self) -> str:
        """Get file extension based on MIME type."""
        return SUPPORTED_MIME_TYPES.get(self.mime_type, '')

    @property
    def is_google_doc(self) -> bool:
        """Check if this is a Google Docs/Sheets/etc file."""
        return self.mime_type.startswith('application/vnd.google-apps.')


class DriveClient:
    """Client for Google Drive API integration."""

    def __init__(self):
        self.service = None
        self.credentials = None

    def is_available(self) -> bool:
        """Check if Google API libraries are installed."""
        return GOOGLE_API_AVAILABLE

    def is_configured(self) -> bool:
        """Check if credentials file exists."""
        return CREDENTIALS_FILE.exists()

    def is_authenticated(self) -> bool:
        """Check if we have valid tokens."""
        if not TOKEN_FILE.exists():
            return False
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
            return creds and creds.valid
        except:
            return False

    def authenticate(self) -> bool:
        """Perform OAuth authentication flow."""
        if not GOOGLE_API_AVAILABLE:
            raise RuntimeError(
                "Google API libraries not installed. Run:\n"
                "pip install google-auth-oauthlib google-api-python-client"
            )

        if not self.is_configured():
            raise RuntimeError(
                f"Drive credentials not found. Please:\n"
                f"1. Create OAuth credentials in Google Cloud Console\n"
                f"2. Save credentials.json to {CREDENTIALS_FILE}"
            )

        creds = None

        # Load existing token if available
        if TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_FILE), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the token for future use
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())

        self.credentials = creds
        self.service = build('drive', 'v3', credentials=creds)
        return True

    def get_storage_quota(self) -> dict:
        """Get storage quota information."""
        if not self.service:
            self.authenticate()

        about = self.service.about().get(fields='storageQuota,user').execute()
        quota = about.get('storageQuota', {})
        user = about.get('user', {})

        return {
            'email': user.get('emailAddress'),
            'display_name': user.get('displayName'),
            'limit': int(quota.get('limit', 0)),
            'usage': int(quota.get('usage', 0)),
            'usage_in_drive': int(quota.get('usageInDrive', 0))
        }

    def list_files(
        self,
        query: str = '',
        folder_id: str = None,
        mime_types: list[str] = None,
        modified_after: datetime = None,
        max_results: int = 100
    ) -> list[DriveFile]:
        """List files in Drive.

        Args:
            query: Additional query string
            folder_id: Only list files in this folder
            mime_types: Filter by MIME types
            modified_after: Only files modified after this date
            max_results: Maximum files to return

        Returns:
            List of DriveFile objects
        """
        if not self.service:
            self.authenticate()

        # Build query
        query_parts = []

        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")

        if mime_types:
            mime_queries = [f"mimeType='{mt}'" for mt in mime_types]
            query_parts.append(f"({' or '.join(mime_queries)})")

        if modified_after:
            date_str = modified_after.isoformat() + 'Z'
            query_parts.append(f"modifiedTime > '{date_str}'")

        if query:
            query_parts.append(query)

        # Don't include trashed files
        query_parts.append("trashed = false")

        full_query = ' and '.join(query_parts)

        # List files
        files = []
        page_token = None

        while len(files) < max_results:
            remaining = max_results - len(files)

            results = self.service.files().list(
                q=full_query,
                pageSize=min(remaining, 100),
                pageToken=page_token,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, parents, webViewLink)"
            ).execute()

            for item in results.get('files', []):
                # Parse modified time
                mod_time_str = item.get('modifiedTime', '')
                try:
                    modified_time = datetime.fromisoformat(mod_time_str.replace('Z', '+00:00'))
                except:
                    modified_time = datetime.now()

                files.append(DriveFile(
                    id=item['id'],
                    name=item['name'],
                    mime_type=item.get('mimeType', ''),
                    modified_time=modified_time,
                    size=int(item.get('size', 0)),
                    parents=item.get('parents', []),
                    web_link=item.get('webViewLink', '')
                ))

            page_token = results.get('nextPageToken')
            if not page_token:
                break

        return files

    def list_folders(self, parent_id: str = None) -> list[DriveFile]:
        """List folders in Drive."""
        return self.list_files(
            folder_id=parent_id,
            mime_types=['application/vnd.google-apps.folder'],
            max_results=1000
        )

    def download_file(self, file: DriveFile) -> Optional[bytes]:
        """Download a file's content.

        Args:
            file: DriveFile to download

        Returns:
            File content as bytes, or None if failed
        """
        if not self.service:
            self.authenticate()

        try:
            if file.is_google_doc:
                # Export Google Docs to appropriate format
                export_mime = EXPORT_MIME_TYPES.get(file.mime_type)
                if not export_mime:
                    return None

                request = self.service.files().export_media(
                    fileId=file.id,
                    mimeType=export_mime
                )
            else:
                request = self.service.files().get_media(fileId=file.id)

            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

            return buffer.getvalue()

        except HttpError as e:
            print(f"Error downloading {file.name}: {e}")
            return None

    def sync_to_vault(
        self,
        vectorstore,
        loader,
        folder_id: str = None,
        days_back: int = 30,
        max_files: int = 200
    ) -> int:
        """Sync Drive files to Vault knowledge base.

        Args:
            vectorstore: VectorStore instance
            loader: DocumentLoader instance
            folder_id: Optional folder to sync from
            days_back: Sync files modified in last N days
            max_files: Maximum files to sync

        Returns:
            Number of files synced
        """
        modified_after = datetime.now() - timedelta(days=days_back)

        # Get supported MIME types
        supported_mimes = list(SUPPORTED_MIME_TYPES.keys())

        print(f"Scanning Drive for files modified in last {days_back} days...")

        files = self.list_files(
            folder_id=folder_id,
            mime_types=supported_mimes,
            modified_after=modified_after,
            max_results=max_files
        )

        print(f"Found {len(files)} files. Downloading and indexing...")

        files_synced = 0

        for file in files:
            print(f"  Processing: {file.name}")

            # Download content
            content = self.download_file(file)
            if not content:
                continue

            # Save to temp file and process
            ext = file.extension
            if file.is_google_doc:
                ext = SUPPORTED_MIME_TYPES.get(file.mime_type, '.txt')

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Process with loader
                chunks = list(loader.process_file(tmp_path))

                # Update source to use Drive file name
                for chunk in chunks:
                    chunk.source = f"drive:{file.name}"
                    chunk.metadata['drive_id'] = file.id
                    chunk.metadata['source'] = 'google_drive'
                    chunk.metadata['modified'] = file.modified_time.isoformat()

                vectorstore.add_chunks(chunks)
                files_synced += 1

            except Exception as e:
                print(f"    Error processing {file.name}: {e}")

            finally:
                os.unlink(tmp_path)

        return files_synced


def get_drive_client() -> DriveClient:
    """Get a Drive client instance."""
    return DriveClient()
