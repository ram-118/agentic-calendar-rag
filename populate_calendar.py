"""
Populate Google Calendar with sample events
Run this first to create test events in your Google Calendar
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Google Calendar API
SCOPES = ['https://www.googleapis.com/auth/calendar']
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


def create_event(service, title: str, date: datetime, start_time: str, end_time: str, description: str = "", location: str = ""):
    """
    Create an event in Google Calendar
    
    Args:
        service: Google Calendar service object
        title: Event title
        date: Event date (datetime object)
        start_time: Start time (HH:MM format)
        end_time: End time (HH:MM format)
        description: Event description
        location: Event location
    """
    try:
        # Parse start and end times
        start_hour, start_min = map(int, start_time.split(':'))
        end_hour, end_min = map(int, end_time.split(':'))
        
        start_dt = date.replace(hour=start_hour, minute=start_min, second=0)
        end_dt = date.replace(hour=end_hour, minute=end_min, second=0)
        
        event = {
            'summary': title,
            'description': description,
            'location': location,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': 'UTC',
            },
        }
        
        event = service.events().insert(calendarId='primary', body=event).execute()
        logger.info(f"[✓] Created: {title} on {date.strftime('%Y-%m-%d')}")
        return event
    
    except HttpError as error:
        logger.error(f"[✗] Error creating event: {error}")
        raise


def populate_sample_events(service):
    """Populate Google Calendar with sample events"""
    
    # Sample events for the next 10 days
    base_date = datetime.now()
    
    sample_events = [
        # Day 1 - January 7 (Wednesday)
        {"title": "Lecture: Data Structures", "offset": 0, "start": "09:00", "end": "10:30", "location": "Room 301"},
        {"title": "Lab: Programming Practice", "offset": 0, "start": "10:45", "end": "12:00", "location": "Lab A"},
        {"title": "Team Meeting - Calendar RAG Project", "offset": 0, "start": "16:15", "end": "18:00", "location": "Conference Room"},
        
        # Day 2 - January 8 (Thursday)
        {"title": "Long Study Session - AI Research", "offset": 1, "start": "10:00", "end": "12:00", "location": "Library"},
        {"title": "Personal Projects - Python Development", "offset": 1, "start": "13:00", "end": "15:00", "location": "Home Office"},
        {"title": "Leisure / Social Time with Friends", "offset": 1, "start": "17:00", "end": "19:00", "location": "Coffee Shop"},
        
        # Day 3 - January 9 (Friday)
        {"title": "Weekly Planning / Goal Setting", "offset": 2, "start": "10:00", "end": "12:00", "location": "Office"},
        {"title": "Lunch Meeting - Team Sync", "offset": 2, "start": "12:30", "end": "13:30", "location": "Cafe"},
        {"title": "Project Development Sprint", "offset": 2, "start": "14:00", "end": "16:00", "location": "Office"},
        
        # Day 4 - January 10 (Saturday)
        {"title": "Lecture: Operating Systems", "offset": 3, "start": "09:00", "end": "11:00", "location": "Room 205"},
        {"title": "Catch-up Study / Assignments", "offset": 3, "start": "13:00", "end": "15:00", "location": "Library"},
        {"title": "Weekend Project Work", "offset": 3, "start": "15:30", "end": "17:30", "location": "Home"},
        
        # Day 5 - January 11 (Sunday)
        {"title": "Lecture: Algorithms", "offset": 4, "start": "09:00", "end": "10:30", "location": "Room 302"},
        {"title": "Lab: Data Science Practice", "offset": 4, "start": "10:45", "end": "12:00", "location": "Lab C"},
        {"title": "Group Study - Algorithm Problems", "offset": 4, "start": "16:15", "end": "18:00", "location": "Study Room"},
        
        # Day 6 - January 12 (Monday)
        {"title": "Lecture: Project Management", "offset": 5, "start": "09:00", "end": "10:30", "location": "Room 301"},
        {"title": "Lab / Presentation Prep", "offset": 5, "start": "10:45", "end": "12:00", "location": "Lab A"},
        {"title": "Internship Work - Final Deliverables", "offset": 5, "start": "15:00", "end": "17:00", "location": "Office"},
        
        # Day 7 - January 13 (Tuesday)
        {"title": "Code Review Session", "offset": 6, "start": "10:00", "end": "11:15", "location": "Conference Room"},
        {"title": "Lab / Exercises", "offset": 6, "start": "11:15", "end": "12:30", "location": "Lab B"},
        {"title": "Office Hours with Mentor", "offset": 6, "start": "17:00", "end": "18:00", "location": "Office"},
        
        # Day 8 - January 14 (Wednesday)
        {"title": "Lecture: Software Engineering", "offset": 7, "start": "09:00", "end": "10:30", "location": "Room 301"},
        {"title": "Lab / Code Review", "offset": 7, "start": "10:45", "end": "12:00", "location": "Lab A"},
        {"title": "Internship Work - Development Tasks", "offset": 7, "start": "15:00", "end": "17:00", "location": "Office"},
        
        # Day 9 - January 15 (Thursday)
        {"title": "Client Meeting - Requirements Review", "offset": 8, "start": "10:00", "end": "11:00", "location": "Video Call"},
        {"title": "Development Sprint Planning", "offset": 8, "start": "14:00", "end": "15:30", "location": "Office"},
        {"title": "Standup - Team Sync", "offset": 8, "start": "09:30", "end": "10:00", "location": "Conference Room"},
        
        # Day 10 - January 16 (Friday)
        {"title": "Lecture: Advanced Topics", "offset": 9, "start": "09:00", "end": "10:30", "location": "Room 302"},
        {"title": "Final Project Submission", "offset": 9, "start": "11:00", "end": "12:00", "location": "Office"},
        {"title": "Week Review & Planning", "offset": 9, "start": "15:00", "end": "16:00", "location": "Office"},
    ]
    
    logger.info("="*70)
    logger.info("Populating Google Calendar with Sample Events")
    logger.info("="*70)
    
    created_count = 0
    for event_data in sample_events:
        try:
            event_date = base_date + timedelta(days=event_data['offset'])
            create_event(
                service,
                title=event_data['title'],
                date=event_date,
                start_time=event_data['start'],
                end_time=event_data['end'],
                location=event_data.get('location', '')
            )
            created_count += 1
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
    
    logger.info("="*70)
    logger.info(f"Successfully created {created_count} sample events")
    logger.info("="*70)
    
    return created_count > 0


if __name__ == "__main__":
    try:
        logger.info("\n[1/2] Authenticating with Google Calendar...")
        creds = authenticate_google_calendar()
        logger.info("[✓] Authentication successful")
        
        logger.info("\n[2/2] Building Google Calendar service...")
        service = build('calendar', 'v3', credentials=creds)
        logger.info("[✓] Service built")
        
        logger.info("\n[3/2] Creating sample events...")
        populate_sample_events(service)
        
        logger.info("\n✓ Calendar population complete!")
        logger.info("Run 'python train_google_calendar.py' to train RAG system")
    
    except FileNotFoundError as e:
        logger.error(f"[✗] {e}")
        logger.error("\nTo set up Google Calendar integration:")
        logger.error("1. Go to https://console.cloud.google.com/")
        logger.error("2. Create a new project or select existing one")
        logger.error("3. Enable Google Calendar API")
        logger.error("4. Create OAuth 2.0 credentials (Desktop app)")
        logger.error("5. Download credentials as JSON")
        logger.error("6. Save as 'credentials.json' in this directory")
    
    except Exception as e:
        logger.error(f"[✗] Error: {e}")
        import traceback
        traceback.print_exc()
