"""
Main Agent: Combines RAG and MCP Tools
Answers user queries using both knowledge base and calendar data
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import httpx
from rag_system import initialize_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys and endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")


class AgentResponse:
    """Response structure from agent"""
    
    def __init__(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        tools_used: List[str],
        calendar_data: Optional[Dict[str, Any]] = None,
        kb_context: Optional[str] = None
    ):
        self.query = query
        self.answer = answer
        self.sources = sources
        self.tools_used = tools_used
        self.calendar_data = calendar_data
        self.kb_context = kb_context
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp,
            "sources": self.sources,
            "tools_used": self.tools_used,
            "calendar_data": self.calendar_data,
            "kb_context_length": len(self.kb_context) if self.kb_context else 0
        }


class MainAgent:
    """Main agent combining RAG and MCP tools"""
    
    def __init__(self, rag_system=None, mcp_url: str = MCP_SERVER_URL):
        """
        Initialize main agent
        
        Args:
            rag_system: Initialized RAG system
            mcp_url: MCP server URL
        """
        self.rag = rag_system or initialize_rag()
        self.mcp_url = mcp_url
        self.mcp_client = httpx.Client(timeout=30.0)
        self.conversation_history = []
    
    def _check_mcp_server(self) -> bool:
        """Check if MCP server is available"""
        try:
            response = self.mcp_client.get(f"{self.mcp_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def _retrieve_knowledge_base(self, query: str, top_k: int = 5) -> tuple:
        """
        Retrieve relevant documents from knowledge base
        
        Returns:
            Tuple of (context_string, search_results)
        """
        results = self.rag.search(query, top_k=top_k)
        context = self.rag.get_context(query, top_k=top_k)
        return context, results
    
    def _list_calendar_events(self, days_ahead: int = 7) -> Optional[Dict[str, Any]]:
        """Get upcoming calendar events"""
        try:
            today = datetime.now()
            start_date = today.date().isoformat()
            end_date = (today + timedelta(days=days_ahead)).date().isoformat()
            
            response = self.mcp_client.post(
                f"{self.mcp_url}/tools/list_events",
                json={"start_date": start_date, "end_date": end_date}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error listing calendar events: {str(e)}")
            return None
    
    def _search_calendar_events(self, keyword: str) -> Optional[Dict[str, Any]]:
        """Search calendar events by keyword"""
        try:
            response = self.mcp_client.post(
                f"{self.mcp_url}/tools/search_events",
                json={"keyword": keyword}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching calendar events: {str(e)}")
            return None
    
    def _get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific event"""
        try:
            response = self.mcp_client.post(
                f"{self.mcp_url}/tools/get_event_details",
                json={"event_id": event_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting event details: {str(e)}")
            return None
    
    def _generate_answer(
        self,
        query: str,
        kb_context: str,
        calendar_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate answer using Claude/GPT via prompt
        Since we don't have API key, we'll create a structured response
        """
        
        # Build comprehensive context
        answer_prompt = f"""
You are a helpful AI assistant for managing calendar and company information.

User Query: {query}

KNOWLEDGE BASE CONTEXT:
{kb_context}

"""
        
        if calendar_data:
            answer_prompt += f"""
CALENDAR DATA:
{json.dumps(calendar_data, indent=2, default=str)}

"""
        
        answer_prompt += """
Based on the knowledge base and calendar data provided above, provide a clear, 
concise answer to the user's query. If the answer is not found in the provided 
context, explicitly state that the information is not available.

Answer:
"""
        
        # For demo purposes, we'll generate a response based on patterns
        # In production, this would call OpenAI API
        return self._generate_response_pattern(query, kb_context, calendar_data)
    
    def _generate_response_pattern(
        self,
        query: str,
        kb_context: str,
        calendar_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response based on query pattern matching"""
        
        query_lower = query.lower()
        
        # Timetable/schedule queries - directly return the relevant content
        if any(word in query_lower for word in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "lab", "lecture", "class", "breakfast", "lunch", "dinner", "timetable", "schedule", "event", "meeting", "january"]):
            # Extract just the matched event entries, not the full context
            if kb_context and len(kb_context.strip()) > 50:
                # Parse kb_context to find events by date
                lines = kb_context.split('\n')
                matched_events = []
                current_date = None
                
                # Extract day and date numbers from query for filtering
                query_day = None
                query_dates = []
                
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                for day in days:
                    if day in query_lower:
                        query_day = day.capitalize()
                        break
                
                # Extract numbers (dates) from query
                import re
                numbers = re.findall(r'\b(\d{1,2})\b', query)
                query_dates = [int(n) for n in numbers if 1 <= int(n) <= 31]
                
                # Process lines
                for line in lines:
                    # Detect date header lines
                    if any(day in line for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']) and 'January' in line:
                        current_date = line.strip()
                    
                    # Check if line is an event
                    if '|' in line and any(keyword in line.lower() for keyword in ['lecture', 'lab', 'meeting', 'study', 'practice', 'office', 'group', 'tutorial', 'club', 'project', 'internship', 'leisure', 'planning', 'assignment']):
                        # Filter by query criteria
                        should_include = False
                        
                        if query_dates and current_date:
                            # Check if current date matches any query dates
                            for date_num in query_dates:
                                if f", {date_num}," in current_date or f"{date_num}, 2026" in current_date:
                                    should_include = True
                                    break
                        elif query_day and current_date:
                            # Check if day matches
                            if query_day in current_date:
                                should_include = True
                        elif not query_dates and not query_day:
                            # No specific filter, include all
                            should_include = True
                        
                        if should_include:
                            matched_events.append(line.strip())
                
                if matched_events:
                    if query_dates:
                        return f"January {query_dates[0]} Events:\n" + '\n'.join(matched_events)
                    elif query_day:
                        return f"{query_day} Events:\n" + '\n'.join(matched_events)
                    return '\n'.join(matched_events)
                
                return "No events found matching your query."
        
        # Work hours/schedule questions
        if any(word in query_lower for word in ["work hours", "office hours", "when"]):
            if any(phrase in kb_context.lower() for phrase in ["work hours", "9:00", "5:30", "office day"]):
                return (
                    "Standard work hours: 9:00 AM to 5:30 PM, Monday to Friday.\n"
                    "Remote work: Tuesdays and Thursdays\n"
                    "Office days: Monday, Wednesday, Friday"
                )
        
        # Leave/time off questions
        if any(word in query_lower for word in ["leave", "vacation", "days off", "absent", "holiday"]):
            if "annual leave" in kb_context:
                return (
                    "Annual leave: 20 days per year\n"
                    "Sick leave: 10 days per year\n"
                    "Maternity/paternity: 3 months with full salary"
                )
        
        # Travel questions
        if any(word in query_lower for word in ["travel", "trip", "flight", "hotel", "accommodation"]):
            if "travel" in kb_context:
                return (
                    "2-week advance request required\n"
                    "Manager approval needed\n"
                    "Hotel: 4-star max ($250/night)\n"
                    "Flights: Economy domestic, Business class for international >6 hours\n"
                    "Daily allowance: $80 per day"
                )
        
        # Internship questions
        if any(word in query_lower for word in ["intern", "internship", "graduate", "trainee"]):
            if "internship" in kb_context:
                return (
                    "Internship program details:\n"
                    "- Duration: 3-6 months\n"
                    "- Minimum commitment: 20 hours/week\n"
                    "- Monthly stipend: $1,500 USD\n"
                    "- Bonus: $500 upon completion\n"
                    "- Start dates: January, April, July, October"
                )
        
        # Calendar/meeting questions
        if any(word in query_lower for word in ["meeting", "calendar", "event", "appointment"]):
            if calendar_data and calendar_data.get("success"):
                events = calendar_data.get("data", {}).get("events", [])
                if events:
                    event_list = "\n".join([f"- {e['title']} on {e['start']}" for e in events[:5]])
                    return f"Your upcoming events:\n{event_list}"
                else:
                    return "You have no upcoming events in the next 7 days."
        
        # Fallback: try to extract just event lines from KB context
        if kb_context and len(kb_context.strip()) > 50:
            lines = kb_context.split('\n')
            event_lines = []
            for line in lines:
                if '|' in line and any(keyword in line.lower() for keyword in ['lecture', 'lab', 'meeting', 'study', 'practice', 'office', 'group', 'tutorial', 'club', 'project', 'internship', 'leisure', 'planning', 'assignment', 'exercise']):
                    event_lines.append(line.strip())
            
            if event_lines:
                return '\n'.join(event_lines[:10])  # Return top 10 matching lines
            
            # If no event lines found, return clean summary
            return "Information found in knowledge base but not directly matching your query."
        
        # Default response
        return (
            "I couldn't find specific information matching your query. "
            "Try asking about: events, meetings, lectures, labs, or your schedule."
        )
    
    def _determine_tools_needed(self, query: str) -> List[str]:
        """Determine which MCP tools are needed"""
        tools_needed = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["event", "meeting", "calendar", "schedule", "appointment"]):
            tools_needed.append("calendar_tools")
        
        if any(word in query_lower for word in ["when", "time", "available", "free"]):
            tools_needed.append("calendar_tools")
        
        if any(word in query_lower for word in ["policy", "rule", "leave", "travel", "code of conduct"]):
            tools_needed.append("knowledge_base")
        
        return tools_needed if tools_needed else ["knowledge_base"]
    
    def process_query(self, query: str) -> AgentResponse:
        """
        Process user query using RAG and MCP tools
        
        Args:
            query: User question
        
        Returns:
            AgentResponse with answer and sources
        """
        logger.info(f"Processing query: {query}")
        
        # Determine tools needed
        tools_needed = self._determine_tools_needed(query)
        
        # Retrieve from knowledge base
        kb_context, kb_results = self._retrieve_knowledge_base(query)
        
        sources = []
        calendar_data = None
        tools_used = ["rag_retrieval"]
        
        # Add KB sources
        for result in kb_results:
            sources.append({
                "type": "knowledge_base",
                "source": result["metadata"]["source"],
                "similarity": result["similarity_score"]
            })
        
        # Use calendar tools if needed and available
        if "calendar_tools" in tools_needed and self._check_mcp_server():
            tools_used.append("calendar_search")
            calendar_data = self._search_calendar_events(query)
            if calendar_data and calendar_data.get("success"):
                sources.append({
                    "type": "calendar",
                    "tool": "search_events",
                    "events_found": calendar_data.get("data", {}).get("count", 0)
                })
        
        # Generate answer
        answer = self._generate_answer(query, kb_context, calendar_data)
        
        # Create response
        response = AgentResponse(
            query=query,
            answer=answer,
            sources=sources,
            tools_used=tools_used,
            calendar_data=calendar_data,
            kb_context=kb_context
        )
        
        # Add to history
        self.conversation_history.append(response)
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return [r.to_dict() for r in self.conversation_history]


def interactive_session():
    """Run interactive agent session"""
    print("="*70)
    print("Agentic AI Calendar Assistant")
    print("="*70)
    print("\nInitializing agent...")
    
    try:
        agent = MainAgent()
        print("[OK] Agent initialized successfully")
        print(f"[OK] RAG System loaded with stats: {agent.rag.get_stats()}")
        
        # Check MCP server
        if agent._check_mcp_server():
            print("[OK] MCP Server is available")
        else:
            print("[!] MCP Server not available (will use RAG only)")
        
        print("\n" + "="*70)
        print("Enter your questions (type 'quit' to exit)")
        print("="*70)
        
        while True:
            try:
                query = input("\n> Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n[*] Processing...")
                response = agent.process_query(query)
                
                print("\n" + response.answer)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"Error: {str(e)}")
    
    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")


def demo_queries():
    """Run demo with predefined queries"""
    print("="*70)
    print("Agentic AI Calendar Assistant - Demo Mode")
    print("="*70)
    
    try:
        agent = MainAgent()
        print("[OK] Agent initialized")
        
        demo_questions = [
            "What are the standard work hours?",
            "How many days of annual leave do I get?",
            "What is the company's travel policy?",
            "Tell me about internship compensation",
            "What are my upcoming meetings?",
            "Search for any meetings in my calendar",
            "What is the code of conduct?",
            "Can you summarize the project guidelines?"
        ]
        
        print("\nRunning demo with sample queries...\n")
        
        for i, query in enumerate(demo_questions, 1):
            print("="*70)
            print(f"Query {i}: {query}")
            print("="*70)
            
            response = agent.process_query(query)
            
            print(f"\nAnswer:\n{response.answer}")
            print(f"\nTools used: {', '.join(response.tools_used)}")
            print(f"Sources: {len(response.sources)} found")
            
            print("\n" + "-"*70)
    
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        logger.error(f"Demo error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_queries()
    else:
        interactive_session()
