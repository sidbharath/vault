"""Gmail integration for Vault.

This module provides real Gmail API integration to sync emails
into the Vault knowledge base.

Setup:
1. Create a Google Cloud project
2. Enable Gmail API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download credentials.json to ~/.vault/credentials.json
5. Run: vault sync gmail
"""

import os
import base64
import email
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

# Google API imports - these are real, production-ready
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False


# OAuth scopes for Gmail read-only access
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Default paths
CONFIG_DIR = Path.home() / '.vault'
CREDENTIALS_FILE = CONFIG_DIR / 'credentials.json'
TOKEN_FILE = CONFIG_DIR / 'gmail_token.json'


@dataclass
class GmailMessage:
    """Represents an email message."""
    id: str
    thread_id: str
    from_addr: str
    to_addr: str
    subject: str
    date: datetime
    body: str
    labels: list[str]
    is_unread: bool

    def to_text(self) -> str:
        """Convert to text format for indexing."""
        date_str = self.date.strftime('%Y-%m-%d %H:%M')
        labels_str = ', '.join(self.labels) if self.labels else 'None'

        return f"""EMAIL
From: {self.from_addr}
To: {self.to_addr}
Date: {date_str}
Subject: {self.subject}
Labels: {labels_str}
Status: {'UNREAD' if self.is_unread else 'Read'}

{self.body}"""


class GmailClient:
    """Client for Gmail API integration."""

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
                f"Gmail credentials not found. Please:\n"
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
        self.service = build('gmail', 'v1', credentials=creds)
        return True

    def get_profile(self) -> dict:
        """Get the authenticated user's email profile."""
        if not self.service:
            self.authenticate()

        profile = self.service.users().getProfile(userId='me').execute()
        return {
            'email': profile.get('emailAddress'),
            'messages_total': profile.get('messagesTotal'),
            'threads_total': profile.get('threadsTotal')
        }

    def list_labels(self) -> list[dict]:
        """List all Gmail labels."""
        if not self.service:
            self.authenticate()

        results = self.service.users().labels().list(userId='me').execute()
        return results.get('labels', [])

    def fetch_messages(
        self,
        query: str = '',
        max_results: int = 100,
        label_ids: list[str] = None,
        after_date: datetime = None
    ) -> list[GmailMessage]:
        """Fetch messages matching the query.

        Args:
            query: Gmail search query (e.g., 'from:client@example.com')
            max_results: Maximum number of messages to fetch
            label_ids: Filter by label IDs (e.g., ['INBOX', 'UNREAD'])
            after_date: Only fetch messages after this date

        Returns:
            List of GmailMessage objects
        """
        if not self.service:
            self.authenticate()

        # Build query
        if after_date:
            date_str = after_date.strftime('%Y/%m/%d')
            query = f"{query} after:{date_str}".strip()

        # List message IDs
        request_params = {
            'userId': 'me',
            'maxResults': max_results,
            'q': query
        }
        if label_ids:
            request_params['labelIds'] = label_ids

        results = self.service.users().messages().list(**request_params).execute()
        message_ids = results.get('messages', [])

        # Fetch full message content
        messages = []
        for msg_info in message_ids:
            try:
                msg = self._fetch_message_detail(msg_info['id'])
                if msg:
                    messages.append(msg)
            except HttpError as e:
                print(f"Error fetching message {msg_info['id']}: {e}")

        return messages

    def _fetch_message_detail(self, message_id: str) -> Optional[GmailMessage]:
        """Fetch full details of a single message."""
        msg = self.service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()

        # Parse headers
        headers = {h['name'].lower(): h['value'] for h in msg['payload'].get('headers', [])}

        from_addr = headers.get('from', 'Unknown')
        to_addr = headers.get('to', 'Unknown')
        subject = headers.get('subject', 'No Subject')
        date_str = headers.get('date', '')

        # Parse date
        try:
            # Gmail dates can be complex, try common formats
            from email.utils import parsedate_to_datetime
            date = parsedate_to_datetime(date_str)
        except:
            date = datetime.now()

        # Extract body
        body = self._extract_body(msg['payload'])

        # Get labels
        labels = msg.get('labelIds', [])
        is_unread = 'UNREAD' in labels

        return GmailMessage(
            id=message_id,
            thread_id=msg.get('threadId', ''),
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            date=date,
            body=body,
            labels=labels,
            is_unread=is_unread
        )

    def _extract_body(self, payload: dict) -> str:
        """Extract plain text body from message payload."""
        body = ""

        if 'body' in payload and payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='replace')

        elif 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType', '')

                if mime_type == 'text/plain':
                    if part['body'].get('data'):
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                        break

                elif mime_type.startswith('multipart/'):
                    body = self._extract_body(part)
                    if body:
                        break

        return body.strip()

    def sync_to_vault(
        self,
        vectorstore,
        loader,
        days_back: int = 30,
        query: str = '',
        max_messages: int = 500
    ) -> int:
        """Sync Gmail messages to Vault knowledge base.

        Args:
            vectorstore: VectorStore instance
            loader: DocumentLoader instance
            days_back: How many days of email to sync
            query: Optional Gmail search query
            max_messages: Maximum messages to sync

        Returns:
            Number of messages synced
        """
        after_date = datetime.now() - timedelta(days=days_back)

        print(f"Fetching emails from the last {days_back} days...")
        messages = self.fetch_messages(
            query=query,
            max_results=max_messages,
            after_date=after_date
        )

        print(f"Found {len(messages)} messages. Indexing...")

        chunks_added = 0
        for msg in messages:
            text = msg.to_text()
            source = f"gmail:{msg.id}"

            chunks = list(loader.chunk_text(
                text,
                source,
                extra_metadata={
                    'type': 'email',
                    'source': 'gmail',
                    'from': msg.from_addr,
                    'subject': msg.subject,
                    'date': msg.date.isoformat(),
                    'is_unread': msg.is_unread
                }
            ))

            chunks_added += vectorstore.add_chunks(chunks)

        return chunks_added


def get_gmail_client() -> GmailClient:
    """Get a Gmail client instance."""
    return GmailClient()
