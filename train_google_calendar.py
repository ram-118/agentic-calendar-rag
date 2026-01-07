"""
Train RAG System with Google Calendar Data
Fetches events from Google Calendar and builds FAISS index
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Google Calendar API
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'


def authenticate_google_calendar():
    """Authenticate with Google Calendar API"""
    creds = None
    
    # Load saved token if exists
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"{CREDENTIALS_FILE} not found. "
                    "Please download OAuth credentials from Google Cloud Console."
                )
            
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save token for future use
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds


def fetch_calendar_events(
    service,
    calendar_id: str = 'primary',
    days_ahead: int = 30
) -> List[Dict[str, Any]]:
    """
    Fetch events from Google Calendar
    
    Args:
        service: Google Calendar service object
        calendar_id: Calendar ID (default: primary/main calendar)
        days_ahead: Number of days to fetch (default: 30)
    
    Returns:
        List of calendar events
    """
    events = []
    
    try:
        now = datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'
        
        logger.info(f"Fetching events from {time_min} to {time_max}")
        
        page_token = None
        while True:
            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                pageToken=page_token,
                singleEvents=True,
                orderBy='startTime',
                maxResults=100
            ).execute()
            
            page_token = events_result.get('nextPageToken')
            batch_events = events_result.get('items', [])
            
            logger.info(f"Retrieved {len(batch_events)} events")
            events.extend(batch_events)
            
            if not page_token:
                break
        
        logger.info(f"Total events fetched: {len(events)}")
        return events
    
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        raise


def format_events_for_rag(events: List[Dict[str, Any]]) -> str:
    """
    Format Google Calendar events into knowledge base format
    
    Args:
        events: List of Google Calendar events
    
    Returns:
        Formatted text for RAG training
    """
    if not events:
        return "No events found in calendar."
    
    formatted = []
    current_date = None
    
    # Sort events by start time
    sorted_events = sorted(
        events,
        key=lambda e: e.get('start', {}).get('dateTime', e.get('start', {}).get('date', ''))
    )
    
    for event in sorted_events:
        try:
            # Get event start time
            start = event.get('start', {})
            if 'dateTime' in start:
                start_dt = datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
            else:
                start_dt = datetime.fromisoformat(start['date'])
            
            # Format date header if date changed
            date_str = start_dt.strftime('%A, %B %d, %Y')
            if date_str != current_date:
                current_date = date_str
                formatted.append(f"\n{current_date}")
            
            # Get event title and details
            title = event.get('summary', 'Untitled Event')
            description = event.get('description', '')
            location = event.get('location', '')
            
            # Format time
            if 'dateTime' in start:
                end = event.get('end', {})
                end_dt = datetime.fromisoformat(end['dateTime'].replace('Z', '+00:00'))
                time_range = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
            else:
                time_range = "All day"
            
            # Format event line
            event_line = f"{time_range} | {title}"
            if location:
                event_line += f" ({location})"
            if description:
                event_line += f" - {description}"
            
            formatted.append(event_line)
        
        except Exception as e:
            logger.warning(f"Error formatting event {event.get('summary', 'Unknown')}: {e}")
            continue
    
    return '\n'.join(formatted)


def train_rag_with_calendar(days_ahead: int = 30):
    """
    Main function: Authenticate, fetch calendar data, and train RAG
    
    Args:
        days_ahead: Number of days of calendar data to fetch
    """
    logger.info("="*70)
    logger.info("Google Calendar RAG Training")
    logger.info("="*70)
    
    try:
        # Step 1: Authenticate
        logger.info("\n[1/5] Authenticating with Google Calendar...")
        creds = authenticate_google_calendar()
        logger.info("[✓] Authentication successful")
        
        # Step 2: Build service and fetch events
        logger.info("\n[2/5] Building Google Calendar service...")
        service = build('calendar', 'v3', credentials=creds)
        logger.info("[✓] Service built")
        
        logger.info("\n[3/5] Fetching calendar events...")
        events = fetch_calendar_events(service, days_ahead=days_ahead)
        
        if not events:
            logger.warning("[!] No events found in calendar")
            return False
        
        logger.info(f"[✓] Fetched {len(events)} events")
        
        # Step 3: Format events for RAG
        logger.info("\n[4/5] Formatting calendar data...")
        formatted_data = format_events_for_rag(events)
        
        # Save to knowledge_base directory
        import os
        kb_dir = 'knowledge_base'
        os.makedirs(kb_dir, exist_ok=True)
        
        calendar_file = os.path.join(kb_dir, 'google_calendar.txt')
        with open(calendar_file, 'w', encoding='utf-8') as f:
            f.write(formatted_data)
        logger.info(f"[✓] Calendar data saved to {calendar_file}")
        
        # Step 4: Train RAG system
        logger.info("\n[5/5] Training RAG system...")
        rag = RAGSystem(knowledge_base_path='knowledge_base')
        rag.build_index(force_rebuild=True)
        logger.info("[✓] RAG system trained successfully")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total events indexed: {len(events)}")
        logger.info(f"Data file: {calendar_file}")
        logger.info(f"Date range: {days_ahead} days from now")
        logger.info(f"RAG stats: {rag.get_stats()}")
        logger.info("="*70)
        
        return True
    
    except FileNotFoundError as e:
        logger.error(f"[✗] {e}")
        logger.error("\nTo set up Google Calendar integration:")
        logger.error("1. Go to https://console.cloud.google.com/")
        logger.error("2. Create a new project or select existing one")
        logger.error("3. Enable Google Calendar API")
        logger.error("4. Create OAuth 2.0 credentials (Desktop app)")
        logger.error("5. Download credentials as JSON")
        logger.error("6. Save as 'credentials.json' in this directory")
        return False
    
    except Exception as e:
        logger.error(f"[✗] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid days argument: {sys.argv[1]}")
            sys.exit(1)
    
    success = train_rag_with_calendar(days_ahead=days)
    sys.exit(0 if success else 1)
