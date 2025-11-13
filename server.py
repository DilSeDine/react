import asyncio
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from videosdk.agents import (Agent, AgentSession, MCPServerHTTP,
                             MCPServerStdio, RealTimePipeline, function_tool)
from videosdk.plugins.google import GeminiLiveConfig, GeminiRealtime

load_dotenv()

# Get port from environment with better debugging
port = int(os.getenv("PORT", 10000))
print(f"Starting server on port: {port}")
print(f"PORT environment variable: {os.getenv('PORT')}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, # More common for public APIs not relying on browser cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global store for active agent sessions and conversation tracking ---
# We'll map meeting_id to the AgentSession instance
active_sessions: Dict[str, AgentSession] = {}
# Track conversations for evaluation
conversation_logs: Dict[str, List[Dict]] = {}
# Store evaluation results
evaluation_results: Dict[str, Dict] = {}
# --- ---

def extract_evaluation_insights_static(conversation_text: str) -> Optional[Dict]:
    """Static function to extract structured evaluation insights from conversation text"""
    insights = {}
    
    # Define regex patterns for extracting evaluation insights
    patterns = {
        'communication_score': r'COMMUNICATION_SCORE:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
        'business_case_analysis': r'BUSINESS_CASE_ANALYSIS:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
        'leadership_potential': r'LEADERSHIP_POTENTIAL:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
        'team_dynamics_skills': r'TEAM_DYNAMICS_SKILLS:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
        'market_strategy_knowledge': r'MARKET_STRATEGY_KNOWLEDGE:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
        'client_management_experience': r'CLIENT_MANAGEMENT_EXPERIENCE:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
        'overall_cultural_fit': r'OVERALL_CULTURAL_FIT:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
        'recommendation_status': r'RECOMMENDATION_STATUS:\s*(.*?)(?=\n|$)',
        'improvement_areas': r'IMPROVEMENT_AREAS:\s*(.*?)(?=\n|$)',
        'notable_strengths': r'NOTABLE_STRENGTHS:\s*(.*?)(?=\n|$)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, conversation_text, re.IGNORECASE | re.DOTALL)
        if match:
            if key in ['communication_score', 'business_case_analysis', 'leadership_potential', 
                      'team_dynamics_skills', 'market_strategy_knowledge', 'client_management_experience', 
                      'overall_cultural_fit']:
                insights[key] = {
                    'score': int(match.group(1)),
                    'feedback': match.group(2).strip()
                }
            else:
                insights[key] = match.group(1).strip()
    
    return insights if insights else None

class MyVoiceAgent(Agent):
    def __init__(self, system_prompt: str, personality: str):
        # mcp_script = Path(__file__).parent / "mcp_studio.py"
        mcp_script_weather = Path(__file__).parent / "mcp_weather.py"
        # mcp_servers = [
        #     MCPServerStdio(
        #     command=sys.executable,
        #     args=[str(mcp_script_weather)],
        #     client_session_timeout_seconds=30
        # ),
        #     MCPServerHTTP(
        #         url=os.getenv("ZAPIER_WEBHOOK_URL")
        #     )
        # ]
        super().__init__(
            instructions=system_prompt,
            # mcp_servers=mcp_servers
        )
        self.personality = personality

    async def on_enter(self) -> None:
        meeting_id = self.session.context.get("meetingId", "unknown")
        # Initialize conversation log for this meeting
        if meeting_id not in conversation_logs:
            conversation_logs[meeting_id] = []
        
        welcome_message = f"Hey, How can I help you today?"
        await self.session.say(welcome_message)
        
        # Log the welcome message
        conversation_logs[meeting_id].append({
            "speaker": "agent",
            "message": welcome_message,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def on_exit(self) -> None:
        meeting_id = self.session.context.get("meetingId", "unknown")
        goodbye_message = "Goodbye!"
        await self.session.say(goodbye_message)
        
        # Log the goodbye message
        if meeting_id in conversation_logs:
            conversation_logs[meeting_id].append({
                "speaker": "agent",
                "message": goodbye_message,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Process evaluation when agent exits
            await self.process_interview_evaluation(meeting_id)

    async def on_user_speech(self, transcript: str) -> None:
        """Track user speech for evaluation"""
        meeting_id = self.session.context.get("meetingId", "unknown")
        if meeting_id in conversation_logs:
            conversation_logs[meeting_id].append({
                "speaker": "user",
                "message": transcript,
                "timestamp": asyncio.get_event_loop().time()
            })

    async def on_agent_speech(self, text: str) -> None:
        """Track agent speech for evaluation"""
        meeting_id = self.session.context.get("meetingId", "unknown")
        if meeting_id in conversation_logs:
            conversation_logs[meeting_id].append({
                "speaker": "agent", 
                "message": text,
                "timestamp": asyncio.get_event_loop().time()
            })

    async def process_interview_evaluation(self, meeting_id: str) -> None:
        """Process the conversation and extract evaluation insights"""
        if meeting_id not in conversation_logs:
            return
            
        conversation = conversation_logs[meeting_id]
        full_conversation = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in conversation])
        
        # Extract evaluation insights using regex patterns
        evaluation = self.extract_evaluation_insights(full_conversation)
        
        if evaluation:
            evaluation_results[meeting_id] = {
                "evaluation": evaluation,
                "conversation": conversation,
                "processed_at": asyncio.get_event_loop().time()
            }
            print(f"[{meeting_id}] Evaluation completed and stored.")

    def extract_evaluation_insights(self, conversation_text: str) -> Optional[Dict]:
        """Extract structured evaluation insights from conversation text"""
        insights = {}
        
        # Define regex patterns for extracting evaluation insights
        patterns = {
            'communication_score': r'COMMUNICATION_SCORE:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
            'business_case_analysis': r'BUSINESS_CASE_ANALYSIS:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
            'leadership_potential': r'LEADERSHIP_POTENTIAL:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
            'team_dynamics_skills': r'TEAM_DYNAMICS_SKILLS:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
            'market_strategy_knowledge': r'MARKET_STRATEGY_KNOWLEDGE:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
            'client_management_experience': r'CLIENT_MANAGEMENT_EXPERIENCE:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
            'overall_cultural_fit': r'OVERALL_CULTURAL_FIT:\s*\[(\d+)\]\s*(.*?)(?=\n|$)',
            'recommendation_status': r'RECOMMENDATION_STATUS:\s*(.*?)(?=\n|$)',
            'improvement_areas': r'IMPROVEMENT_AREAS:\s*(.*?)(?=\n|$)',
            'notable_strengths': r'NOTABLE_STRENGTHS:\s*(.*?)(?=\n|$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, conversation_text, re.IGNORECASE | re.DOTALL)
            if match:
                if key in ['communication_score', 'business_case_analysis', 'leadership_potential', 
                          'team_dynamics_skills', 'market_strategy_knowledge', 'client_management_experience', 
                          'overall_cultural_fit']:
                    insights[key] = {
                        'score': int(match.group(1)),
                        'feedback': match.group(2).strip()
                    }
                else:
                    insights[key] = match.group(1).strip()
        
        return insights if insights else None
        

    @function_tool
    async def end_call(self) -> None:
        """End the call upon request by the user"""
        await self.session.say("Goodbye!")
        await asyncio.sleep(1)
        await self.session.leave()
  

class MeetingReqConfig(BaseModel):
    meeting_id: str
    token: str
    model: str
    voice: str
    personality: str
    temperature: float
    system_prompt: str
    topP: float
    topK: float


class LeaveAgentReqConfig(BaseModel): # For the leave endpoint
    meeting_id: str

class GetEvaluationRequest(BaseModel):
    meeting_id: str

async def server_operations(req: MeetingReqConfig):
    print(f"req body : {req}")
    meeting_id = req.meeting_id
    print(f"[{meeting_id}] Initializing agent operations...")
    

    # Use all values from the request
    model = GeminiRealtime(
        model=req.model,
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(
            voice=req.voice,
            response_modalities=["AUDIO"],
            temperature=req.temperature,
            top_p=req.topP,
            top_k=int(req.topK),
        )
    )

    pipeline = RealTimePipeline(model=model)

    # Pass system_prompt and personality in the context if your agent uses them
    session = AgentSession(
        agent=MyVoiceAgent(req.system_prompt, req.personality),
        pipeline=pipeline,
        context={
            "meetingId": meeting_id,
            "name": "Gemini Agent",
            "videosdk_auth": req.token,
        }
    )

    active_sessions[meeting_id] = session
    print(f"[{meeting_id}] Agent session stored. Current active sessions: {list(active_sessions.keys())}")

    try:
        print(f"[{meeting_id}] Agent attempting to start...")
        await session.start()
        print(f"[{meeting_id}] Agent session.start() completed normally.")
    except Exception as ex:
        print(f"[{meeting_id}] [ERROR] in agent session: {ex}")
        if active_sessions.get(meeting_id) is session:
            active_sessions.pop(meeting_id, None)
            try:
                if hasattr(session, 'leave') and session.leave is not None:
                    await session.leave()
            except Exception as leave_ex:
                print(f"[{meeting_id}] [ERROR] during cleanup after failed start: {leave_ex}")
    finally:
        print(f"[{meeting_id}] Server operations completed for this session.")


@app.get("/join-agent")
async def join_agent_info():
    return {
        "error": "Method Not Allowed",
        "message": "This endpoint requires a POST request with JSON data",
        "usage": "POST /join-agent",
        "required_fields": {
            "meeting_id": "string",
            "token": "string", 
            "model": "string",
            "voice": "string",
            "personality": "string",
            "temperature": "float",
            "system_prompt": "string",
            "topP": "float",
            "topK": "float"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }

@app.post("/join-agent")
async def join_agent(req: MeetingReqConfig, bg_tasks: BackgroundTasks):
    if req.meeting_id in active_sessions:
        # Optional: decide how to handle re-joining an already active meeting
        # For now, let's allow it, the new agent will replace the old one in `active_sessions`
        # The old background task will eventually complete and clean itself up.
        print(f"Agent joining meeting {req.meeting_id} which might already have an active agent. A new one will be started.")

    bg_tasks.add_task(server_operations, req)
    return {"message": f"AI agent joining process initiated for meeting {req.meeting_id}"}


# --- NEW/MODIFIED ENDPOINT ---
@app.get("/leave-agent")
async def leave_agent_info():
    return {
        "error": "Method Not Allowed",
        "message": "This endpoint requires a POST request with JSON data",
        "usage": "POST /leave-agent",
        "required_fields": {
            "meeting_id": "string"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }

@app.post("/leave-agent")
async def leave_agent(req: LeaveAgentReqConfig):
    meeting_id = req.meeting_id
    print(f"[{meeting_id}] Received /leave-agent request.")

    session = active_sessions.get(meeting_id, None)  # Don't pop yet, just get reference
    
    # STEP 1: Process evaluation FIRST before any session cleanup
    evaluation_data = None
    conversation_data = None
    
    print(f"[{meeting_id}] Processing evaluation before session cleanup...")
    
    # Try to process evaluation using the agent instance first
    if session and hasattr(session, 'agent') and session.agent:
        try:
            await session.agent.process_interview_evaluation(meeting_id)
            print(f"[{meeting_id}] Evaluation processed using agent instance.")
        except Exception as e:
            print(f"[{meeting_id}] Error processing evaluation with agent: {e}")
    
    # Check for evaluation results from agent processing
    if meeting_id in evaluation_results:
        evaluation_data = evaluation_results[meeting_id]["evaluation"]
        conversation_data = evaluation_results[meeting_id]["conversation"]
        print(f"[{meeting_id}] Evaluation results found from agent processing.")
    elif meeting_id in conversation_logs:
        conversation_data = conversation_logs[meeting_id]
        print(f"[{meeting_id}] Conversation data found, processing evaluation with static method...")
        
        # Use static method to process evaluation
        full_conversation = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in conversation_data])
        evaluation = extract_evaluation_insights_static(full_conversation)
        
        if evaluation:
            evaluation_results[meeting_id] = {
                "evaluation": evaluation,
                "conversation": conversation_data,
                "processed_at": asyncio.get_event_loop().time()
            }
            evaluation_data = evaluation
            print(f"[{meeting_id}] Evaluation processed with static method and stored.")
        else:
            print(f"[{meeting_id}] No evaluation insights found in conversation.")
    else:
        print(f"[{meeting_id}] No conversation data found for evaluation.")
    
    # STEP 2: Now that evaluation is complete, safely cleanup session
    if session:
        print(f"[{meeting_id}] Starting session cleanup after evaluation processing...")
        
        # Remove from active sessions
        active_sessions.pop(meeting_id, None)
        print(f"[{meeting_id}] Session removed from active_sessions.")
        
        # Try to close session gracefully
        try:
            if hasattr(session, 'leave') and callable(getattr(session, 'leave', None)):
                leave_result = session.leave()
                if leave_result is not None:
                    await leave_result
                    print(f"[{meeting_id}] Session.leave() completed successfully.")
                else:
                    print(f"[{meeting_id}] Session.leave() returned None, session likely already closed.")
            else:
                print(f"[{meeting_id}] Session does not have a callable leave method.")
        except Exception as e:
            print(f"[{meeting_id}] Error during session.leave(): {e}")
        
        return {
            "status": "removed",
            "meeting_id": meeting_id,
            "message": f"Session for meeting {meeting_id} has been removed after processing evaluation.",
            "evaluation": evaluation_data,
            "conversation": conversation_data,
            "has_evaluation": evaluation_data is not None,
            "conversation_length": len(conversation_data) if conversation_data else 0,
            "processing_order": "evaluation_first_then_cleanup"
        }
    else:
        print(f"[{meeting_id}] No session found in active_sessions.")
        
        # Still check for evaluation/conversation data even if session not found
        evaluation_data = None
        conversation_data = None
        
        if meeting_id in evaluation_results:
            evaluation_data = evaluation_results[meeting_id]["evaluation"]
            conversation_data = evaluation_results[meeting_id]["conversation"]
            print(f"[{meeting_id}] Found existing evaluation results.")
        elif meeting_id in conversation_logs:
            conversation_data = conversation_logs[meeting_id]
            print(f"[{meeting_id}] Found conversation, manually processing evaluation...")
            
            # Manually process evaluation using static method approach
            full_conversation = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in conversation_data])
            evaluation = extract_evaluation_insights_static(full_conversation)
            
            if evaluation:
                evaluation_results[meeting_id] = {
                    "evaluation": evaluation,
                    "conversation": conversation_data,
                    "processed_at": asyncio.get_event_loop().time()
                }
                evaluation_data = evaluation
                print(f"[{meeting_id}] Evaluation manually processed and stored.")
        
        return {
            "status": "not_found",
            "meeting_id": meeting_id,
            "message": f"No active session found for meeting {meeting_id}, but checking for data.",
            "evaluation": evaluation_data,
            "conversation": conversation_data,
            "has_evaluation": evaluation_data is not None,
            "conversation_length": len(conversation_data) if conversation_data else 0
        }
# --- END NEW/MODIFIED ENDPOINT ---

# Optional: Keep these endpoints for debugging or manual access if needed
@app.get("/debug/get-evaluation")
async def debug_get_evaluation_info():
    return {
        "message": "Debug endpoint - use POST with meeting_id",
        "active_evaluations": list(evaluation_results.keys()),
        "active_conversations": list(conversation_logs.keys())
    }


@app.get("/")
async def root():
    return {
        "message": "VideoSDK AI Agent Demo Server",
        "status": "running",
        "endpoints": {
            "test": "GET /test - Test if server is running",
            "join_agent": "POST /join-agent - Join an AI agent to a meeting",
            "leave_agent": "POST /leave-agent - Remove agent and get evaluation results",
            "debug_evaluation": "GET /debug/get-evaluation - Debug endpoint for evaluation data"
        },
        "docs": "Visit /docs for interactive API documentation",
        "stats": {
            "active_sessions": len(active_sessions),
            "total_conversations": len(conversation_logs),
            "completed_evaluations": len(evaluation_results)
        }
    }

@app.get("/test")
async def test():
    return {"message": "Server is running!"}


if __name__ == "__main__":
    # Print configuration for debugging
    print(f"Server starting with host=0.0.0.0, port={port}")
    print(f"Environment variables:")
    print(f"  PORT: {os.getenv('PORT')}")
    print(f"  GOOGLE_API_KEY: {'Set' if os.getenv('GOOGLE_API_KEY') else 'Not set'}")
    print(f"  VIDEOSDK_API_KEY: {'Set' if os.getenv('VIDEOSDK_API_KEY') else 'Not set'}")
    
    # Use 0.0.0.0 to bind to all interfaces for deployment platforms like Render
    # Disable reload in production
    is_development = os.getenv('RENDER') is None
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=is_development)
