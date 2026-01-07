"""
MCP Server for Google Calendar
Exposes 4 tools: list_events, get_event_details, search_events, create_event
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# FastAPI app
app = FastAPI(title="Calendar MCP Server", version="1.0.0")

# Global service object
service = None

# ==================== Models ====================

class ListEventsRequest(BaseModel):
    start_date: str  # ISO format: "2026-01-01"
    end_date: str    # ISO format: "2026-01-31"


class GetEventDetailsRequest(BaseModel):
    event_id: str


class SearchEventsRequest(BaseModel):
    keyword: str


class CreateEventRequest(BaseModel):
    title: str
    date: str  # ISO format: "2026-01-15"
    start_time: str  # Format: "10:00" (24-hour)
    end_time: str    # Format: "11:00" (24-hour)
    description: str = ""
    location: str = ""


class EventResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== Authentication ====================

def authenticate_google_calendar():
    """Authenticate with Google Calendar API"""
    global service
    
    try:
        creds = None
        
        # Load saved token if exists
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as refresh_err:
                    logger.warning(f"Token refresh failed: {refresh_err}. Using mock calendar.")
                    raise
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
            
            # Save credentials for future use
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build the service
        service = build('calendar', 'v3', credentials=creds)
        logger.info("✅ Google Calendar authenticated successfully")
        return service
    except Exception as e:
        logger.warning(f"Google Calendar authentication failed: {str(e)}")
        logger.info("⚠️ Using MOCK CALENDAR for testing. To use real Google Calendar:")
        logger.info("   1. Go to: https://console.cloud.google.com/")
        logger.info("   2. Select project: summer-reef-483509-f3")
        logger.info("   3. Go to: APIs & Services → OAuth consent screen")
        logger.info("   4. Click 'Add users' under 'Test users'")
        logger.info("   5. Add: ramamaresam@gmail.com")
        logger.info("   6. Save and try again")
        logger.info("---")
        service = None  # Use mock calendar
        return service


# ==================== Tool Implementations ====================

def get_mock_events(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Get mock calendar events for testing when Google Calendar is not available"""
    mock_events = [
        {
            'id': '1',
            'title': 'Team Meeting - Calendar RAG Project',
            'start': '2026-01-07T16:15:00',
            'end': '2026-01-07T18:00:00',
            'description': 'Planning calendar RAG integration',
            'location': 'Room 301'
        },
        {
            'id': '2',
            'title': 'Lecture: Data Structures',
            'start': '2026-01-07T09:00:00',
            'end': '2026-01-07T10:30:00',
            'description': '',
            'location': 'Room 301'
        },
        {
            'id': '3',
            'title': 'Lab: Programming Practice',
            'start': '2026-01-07T10:45:00',
            'end': '2026-01-07T12:00:00',
            'description': '',
            'location': 'Lab A'
        }
    ]
    return mock_events

def list_events_impl(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    List all events between two dates
    
    Args:
        start_date: ISO format "2026-01-01"
        end_date: ISO format "2026-01-31"
    
    Returns:
        Dict with list of events
    """
    try:
        # If Google Calendar service is not available, use mock
        if service is None:
            logger.info("Using mock calendar events (Google Calendar not connected)")
            events = get_mock_events(start_date, end_date)
            return {
                'success': True,
                'count': len(events),
                'events': events,
                'message': f"Mock calendar: Found {len(events)} events between {start_date} and {end_date}",
                'source': 'mock'
            }
        
        # Convert to RFC3339 format for API
        start_datetime = datetime.fromisoformat(start_date)
        end_datetime = datetime.fromisoformat(end_date)
        
        start_rfc = start_datetime.isoformat() + 'Z'
        end_rfc = end_datetime.isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_rfc,
            timeMax=end_rfc,
            maxResults=100,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Format events for response
        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event.get('id'),
                'title': event.get('summary', 'Untitled'),
                'start': event.get('start', {}).get('dateTime', event.get('start', {}).get('date')),
                'end': event.get('end', {}).get('dateTime', event.get('end', {}).get('date')),
                'description': event.get('description', ''),
                'location': event.get('location', ''),
                'attendees': [a.get('email', '') for a in event.get('attendees', [])],
                'url': event.get('htmlLink', '')
            })
        
        return {
            'success': True,
            'count': len(formatted_events),
            'events': formatted_events,
            'message': f"Found {len(formatted_events)} events between {start_date} and {end_date}"
        }
    
    except Exception as e:
        logger.error(f"Error listing events: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to list events: {str(e)}"
        }


def get_event_details_impl(event_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific event
    
    Args:
        event_id: The ID of the event
    
    Returns:
        Detailed event information
    """
    try:
        event = service.events().get(
            calendarId='primary',
            eventId=event_id
        ).execute()
        
        return {
            'success': True,
            'event': {
                'id': event.get('id'),
                'title': event.get('summary', 'Untitled'),
                'start': event.get('start', {}).get('dateTime', event.get('start', {}).get('date')),
                'end': event.get('end', {}).get('dateTime', event.get('end', {}).get('date')),
                'description': event.get('description', ''),
                'location': event.get('location', ''),
                'attendees': [
                    {'email': a.get('email'), 'status': a.get('responseStatus')}
                    for a in event.get('attendees', [])
                ],
                'organizer': event.get('organizer', {}).get('email', ''),
                'created': event.get('created'),
                'updated': event.get('updated'),
                'status': event.get('status'),
                'url': event.get('htmlLink', '')
            },
            'message': f"Retrieved details for event: {event.get('summary', 'Untitled')}"
        }
    
    except HttpError as e:
        logger.error(f"Error getting event {event_id}: {str(e)}")
        return {
            'success': False,
            'error': f"Event not found or access denied: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error getting event details: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to get event details: {str(e)}"
        }


def search_events_impl(keyword: str) -> Dict[str, Any]:
    """
    Search for events by keyword
    
    Args:
        keyword: Search term to find in event titles and descriptions
    
    Returns:
        List of matching events
    """
    try:
        # Get events from the last year to now and next 2 years
        now = datetime.utcnow()
        start_date = (now - timedelta(days=365)).isoformat() + 'Z'
        end_date = (now + timedelta(days=730)).isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_date,
            timeMax=end_date,
            maxResults=250,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Filter events by keyword
        keyword_lower = keyword.lower()
        matching_events = []
        
        for event in events:
            title = event.get('summary', '').lower()
            description = event.get('description', '').lower()
            location = event.get('location', '').lower()
            
            if (keyword_lower in title or 
                keyword_lower in description or 
                keyword_lower in location):
                matching_events.append({
                    'id': event.get('id'),
                    'title': event.get('summary', 'Untitled'),
                    'start': event.get('start', {}).get('dateTime', event.get('start', {}).get('date')),
                    'end': event.get('end', {}).get('dateTime', event.get('end', {}).get('date')),
                    'description': event.get('description', ''),
                    'location': event.get('location', ''),
                    'match_reason': 'Keyword found in event'
                })
        
        return {
            'success': True,
            'keyword': keyword,
            'count': len(matching_events),
            'events': matching_events,
            'message': f"Found {len(matching_events)} events matching '{keyword}'"
        }
    
    except Exception as e:
        logger.error(f"Error searching events: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to search events: {str(e)}"
        }


def check_availability_impl(start_datetime: str, end_datetime: str) -> bool:
    """
    Check if time slot is available (no conflicting events)
    
    Args:
        start_datetime: ISO format datetime
        end_datetime: ISO format datetime
    
    Returns:
        True if available, False if conflict exists
    """
    try:
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_datetime,
            timeMax=end_datetime,
            maxResults=50,
            singleEvents=True
        ).execute()
        
        events = events_result.get('items', [])
        return len(events) == 0
    
    except Exception as e:
        logger.error(f"Error checking availability: {str(e)}")
        return False


def create_event_impl(
    title: str,
    date: str,  # ISO format: "2026-01-15"
    start_time: str,  # "10:00"
    end_time: str,    # "11:00"
    description: str = "",
    location: str = ""
) -> Dict[str, Any]:
    """
    Check availability and create a new event
    
    Args:
        title: Event title
        date: Date in ISO format
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
        description: Event description
        location: Event location
    
    Returns:
        Result of event creation
    """
    try:
        # Parse datetime
        start_str = f"{date}T{start_time}:00"
        end_str = f"{date}T{end_time}:00"
        
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        
        # Check availability
        start_rfc = start_dt.isoformat() + 'Z'
        end_rfc = end_dt.isoformat() + 'Z'
        
        is_available = check_availability_impl(start_rfc, end_rfc)
        
        if not is_available:
            return {
                'success': False,
                'error': f"Time slot not available. Conflict with existing event.",
                'available': False
            }
        
        # Create event
        event = {
            'summary': title,
            'description': description,
            'location': location,
            'start': {
                'dateTime': start_dt.isoformat(),
                'timeZone': 'UTC'
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': 'UTC'
            }
        }
        
        created_event = service.events().insert(
            calendarId='primary',
            body=event
        ).execute()
        
        return {
            'success': True,
            'available': True,
            'event_id': created_event.get('id'),
            'event': {
                'title': created_event.get('summary'),
                'start': created_event.get('start'),
                'end': created_event.get('end'),
                'location': created_event.get('location'),
                'url': created_event.get('htmlLink')
            },
            'message': f"Event '{title}' created successfully"
        }
    
    except ValueError as e:
        logger.error(f"Invalid datetime format: {str(e)}")
        return {
            'success': False,
            'error': f"Invalid date/time format: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error creating event: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to create event: {str(e)}"
        }


# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize Google Calendar service on startup"""
    global service
    try:
        service = authenticate_google_calendar()
        logger.info("MCP Server started successfully")
    except Exception as e:
        logger.error(f"Failed to start MCP Server: {str(e)}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Calendar MCP Server",
        "version": "1.0.0"
    }


@app.post("/tools/list_events")
async def list_events(request: ListEventsRequest):
    """List events between two dates"""
    result = list_events_impl(request.start_date, request.end_date)
    return EventResponse(
        success=result.get('success'),
        data=result,
        error=result.get('error')
    )


@app.post("/tools/get_event_details")
async def get_event_details(request: GetEventDetailsRequest):
    """Get details of a specific event"""
    result = get_event_details_impl(request.event_id)
    return EventResponse(
        success=result.get('success'),
        data=result,
        error=result.get('error')
    )


@app.post("/tools/search_events")
async def search_events(request: SearchEventsRequest):
    """Search events by keyword"""
    result = search_events_impl(request.keyword)
    return EventResponse(
        success=result.get('success'),
        data=result,
        error=result.get('error')
    )


@app.post("/tools/create_event")
async def create_event(request: CreateEventRequest):
    """Create a new event with availability check"""
    result = create_event_impl(
        request.title,
        request.date,
        request.start_time,
        request.end_time,
        request.description,
        request.location
    )
    return EventResponse(
        success=result.get('success'),
        data=result,
        error=result.get('error')
    )


@app.get("/tools")
async def list_tools():
    """List all available tools"""
    return {
        "tools": [
            {
                "name": "list_events",
                "description": "Get events between two dates",
                "inputs": {
                    "start_date": "ISO format string (e.g., 2026-01-01)",
                    "end_date": "ISO format string (e.g., 2026-01-31)"
                }
            },
            {
                "name": "get_event_details",
                "description": "Get details of a specific event",
                "inputs": {
                    "event_id": "Google Calendar event ID"
                }
            },
            {
                "name": "search_events",
                "description": "Search events by keyword",
                "inputs": {
                    "keyword": "Search term to find in events"
                }
            },
            {
                "name": "create_event",
                "description": "Create a new event (checks availability first)",
                "inputs": {
                    "title": "Event title",
                    "date": "ISO format date (e.g., 2026-01-15)",
                    "start_time": "HH:MM format (24-hour)",
                    "end_time": "HH:MM format (24-hour)",
                    "description": "Event description (optional)",
                    "location": "Event location (optional)"
                }
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
