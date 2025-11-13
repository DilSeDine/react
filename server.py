import asyncio
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from videosdk.agents import (Agent, AgentSession, MCPServerHTTP,
                             MCPServerStdio, RealTimePipeline, function_tool)
from videosdk.plugins.google import GeminiLiveConfig, GeminiRealtime
from videosdk.plugins.simli import SimliAvatar, SimliConfig

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

# --- Global store for active agent sessions ---
# We'll map meeting_id to the AgentSession instance
active_sessions: Dict[str, AgentSession] = {}
# --- ---

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
        await self.session.say(f"Hey, How can I help you today?")
    
    async def on_exit(self) -> None:
        await self.session.say("Goodbye!")
        

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
    enable_avatar: bool = False  # Optional flag to enable Simli avatar
    face_id: str = "d2a5c7c6-fed9-4f55-bcb3-062f7cd20103"  # Default face ID


class LeaveAgentReqConfig(BaseModel): # For the leave endpoint
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

    # Initialize Simli Avatar if enabled
    pipeline = None
    if req.enable_avatar and os.getenv("SIMLI_API_KEY"):
        print(f"[{meeting_id}] Initializing Simli Avatar with face ID: {req.face_id}")
        simli_config = SimliConfig(
            apiKey=os.getenv("SIMLI_API_KEY"),
            faceId=req.face_id,
            maxSessionLength=1800,
            maxIdleTime=600,
        )
        simli_avatar = SimliAvatar(
            config=simli_config,
            is_trinity_avatar=True,
        )
        pipeline = RealTimePipeline(model=model, avatar=simli_avatar)
        print(f"[{meeting_id}] Simli Avatar initialized successfully")
    else:
        pipeline = RealTimePipeline(model=model)
        if req.enable_avatar:
            print(f"[{meeting_id}] Avatar requested but SIMLI_API_KEY not found - using voice-only mode")

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
            "topK": "float",
            "enable_avatar": "boolean (optional)",
            "face_id": "string (optional)"
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

    session = active_sessions.pop(meeting_id, None)

    if session:
        print(f"[{meeting_id}] Session removed from active_sessions.")
        return {
            "status": "removed",
            "meeting_id": meeting_id,
            "message": f"Session for meeting {meeting_id} has been removed."
        }
    else:
        print(f"[{meeting_id}] No session found in active_sessions.")
        return {
            "status": "not_found",
            "meeting_id": meeting_id,
            "message": f"No session found for meeting {meeting_id}."
        }
# --- END NEW/MODIFIED ENDPOINT ---


@app.get("/")
async def root():
    return {
        "message": "VideoSDK AI Agent Demo Server",
        "status": "running",
        "endpoints": {
            "test": "GET /test - Test if server is running",
            "join_agent": "POST /join-agent - Join an AI agent to a meeting",
            "leave_agent": "POST /leave-agent - Remove an agent from a meeting"
        },
        "docs": "Visit /docs for interactive API documentation"
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
    print(f"  SIMLI_API_KEY: {'Set' if os.getenv('SIMLI_API_KEY') else 'Not set'}")
    
    # Use 0.0.0.0 to bind to all interfaces for deployment platforms like Render
    # Disable reload in production
    is_development = os.getenv('RENDER') is None
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=is_development)
