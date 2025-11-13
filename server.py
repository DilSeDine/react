import asyncio
import json
import os
import sys
import traceback
import uuid
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

load_dotenv()

port = int(os.getenv("PORT", 8000)) # Use environment variable for port, default to 8000

# Initialize app without database lifespan
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
    def __init__(self, system_prompt: str, personality: str, meeting_id: str, candidate_data: Dict = None):
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
        self.connection_lost = False
        self.meeting_id = meeting_id
        self.session_id = str(uuid.uuid4())
        self.candidate_data = candidate_data or {}
        self.question_count = 0
        self.conversation_history = []
        self.current_phase = "introduction"
        self.is_finalized = False
        
    async def create_database_session(self):
        """Print interview session data instead of saving to database"""
        try:
            session_data = {
                "session_id": self.session_id,
                "meeting_id": self.meeting_id,
                "access_code": self.candidate_data.get("accessCode", ""),
                "candidate_info": self.candidate_data.get("candidate", {}),
                "job_info": self.candidate_data.get("job", {}),
                "company_info": self.candidate_data.get("company", {}),
                "model": "gemini-2.0-flash-live-001",
                "voice": "Puck",  # You can make this dynamic
                "temperature": 0.8,  # You can make this dynamic
                "personality": self.personality
            }
            
            print(f"=== CREATING INTERVIEW SESSION ===")
            print(f"Session Data: {json.dumps(session_data, indent=2)}")
            print(f"Created session: {self.session_id}")
            
        except Exception as e:
            print(f"Error creating session: {e}")
  
    async def log_conversation(self, speaker: str, text: str):
        """Print conversation data instead of saving to database"""
        try:
            transcript_data = {
                "session_id": self.session_id,
                "speaker": speaker,
                "text": text,
                "timestamp": asyncio.get_event_loop().time()
            }
            print(f"=== CONVERSATION LOG ===")
            print(f"Speaker: {speaker}")
            print(f"Text: {text}")
            print(f"Session ID: {self.session_id}")
        except Exception as e:
            print(f"Error logging conversation: {e}")

    async def log_question_and_evaluation(self, question: str, answer: str, evaluation: Dict):
        """Print question, answer, and evaluation data instead of saving to database"""
        try:
            qa_data = {
                "session_id": self.session_id,
                "question_number": self.question_count,
                "phase": self.current_phase,
                "question": question,
                "answer": answer,
                "difficulty": evaluation.get("difficulty", "medium"),
                "category": evaluation.get("category", "technical"),
                "technical_accuracy": evaluation.get("technical_accuracy", 5),
                "communication_clarity": evaluation.get("communication_clarity", 5),
                "depth_of_knowledge": evaluation.get("depth_of_knowledge", 5),
                "relevance": evaluation.get("relevance", 5),
                "overall_score": evaluation.get("overall_score", 5),
                "confidence_level": evaluation.get("confidence_level", "medium"),
                "strengths": evaluation.get("strengths", []),
                "areas_for_improvement": evaluation.get("areas_for_improvement", []),
                "detailed_feedback": evaluation.get("detailed_feedback", ""),
                "next_difficulty": evaluation.get("next_difficulty", "same"),
                "evaluation_approach": evaluation.get("evaluation_approach", ""),
                "interviewer_notes": evaluation.get("interviewer_notes", "")
            }
            
            print(f"=== QUESTION & ANSWER EVALUATION ===")
            print(f"Q&A Data: {json.dumps(qa_data, indent=2)}")
            
        except Exception as e:
            print(f"Error logging Q&A: {e}")

    async def on_enter(self) -> None:
        try:
            # Create database session
            await self.create_database_session()
            
            # Check if this is a personalized interview with candidate information
            if "**Candidate Name:**" in self.instructions:
                # Extract candidate name from the system prompt
                lines = self.instructions.split('\n')
                candidate_name = "there"
                for line in lines:
                    if "**Candidate Name:**" in line:
                        candidate_name = line.split("**Candidate Name:**")[1].strip()
                        break
                greeting = f"Hello! I'm your technical interviewer for today. I'm excited to learn about your background and assess your expertise. Shall we begin?"
                await self.session.say(greeting)
                await self.log_conversation("interviewer", greeting)
            elif self.personality == "Technical Interviewer":
                greeting = "Hello! I'm your technical interviewer for today. I'm excited to learn about your background and assess your expertise in Java programming and Machine Learning. Shall we begin?"
                await self.session.say(greeting)
                await self.log_conversation("interviewer", greeting)
            else:
                await self.session.say(f"Hey, How can I help you today?")
        except Exception as e:
            print(f"Error in on_enter: {e}")
            # Fallback greeting
            try:
                await self.session.say("Hello! I'm your interviewer for today. Are you ready to begin?")
            except:
                print("Failed to send fallback greeting")
    
    async def on_exit(self) -> None:
        try:
            print(f"[{self.meeting_id}] on_exit called, session exists: {self.session is not None}")
            goodbye_msg = "Thank you for your time today. The interview has been completed and your responses have been recorded. You should hear back from our team soon. Have a great day!"
            
            # Try to send goodbye message if session still exists
            if self.session:
                try:
                    await self.session.say(goodbye_msg)
                    await self.log_conversation("interviewer", goodbye_msg)
                    print(f"[{self.meeting_id}] Goodbye message sent successfully")
                except Exception as session_error:
                    print(f"[{self.meeting_id}] Failed to send goodbye message: {session_error}")
            else:
                print(f"[{self.meeting_id}] Session is None, cannot send goodbye message")
            
            # Finalize interview evaluation in database
            await self.finalize_interview_evaluation()
            
        except Exception as e:
            print(f"Error in on_exit: {e}")
    
    async def finalize_interview_evaluation(self):
        """Calculate and store overall interview evaluation"""
        if self.is_finalized:
            print(f"Interview already finalized for session: {self.session_id}")
            return
            
        try:
            self.is_finalized = True
            # This would typically analyze all Q&As to generate overall scores
            # For now, providing a basic implementation
            evaluation_data = {
                "overall_score": 7,  # This should be calculated from individual Q&A scores
                "recommendation": "hire",  # This should be determined based on performance
                "technical_competency": 7,
                "communication_skills": 8,
                "problem_solving": 7,
                "cultural_fit": 8,
                "leadership_potential": 6,
                "key_strengths": ["Good communication", "Strong technical foundation"],
                "areas_for_development": ["More hands-on experience with advanced topics"],
                "interview_summary": "Candidate demonstrated solid technical knowledge and good communication skills.",
                "hiring_recommendation_reason": "Strong technical foundation with good potential for growth",
                "total_questions": self.question_count,
                "duration_minutes": 45  # This could be calculated from start/end times
            }
            
            print(f"=== FINAL INTERVIEW EVALUATION ===")
            print(f"Evaluation Data: {json.dumps(evaluation_data, indent=2)}")
            print(f"Finalized interview evaluation for session: {self.session_id}")
            
        except Exception as e:
            print(f"Error finalizing interview evaluation: {e}")
            self.is_finalized = False  # Reset flag if finalization failed
    
    async def on_error(self, error) -> None:
        """Handle errors that occur during the session"""
        print(f"[{self.meeting_id}] Agent session error: {error}")
        self.connection_lost = True
        
        # Check if it's a Google API service unavailable error
        if "service is currently unavailable" in str(error).lower() or "1011" in str(error):
            print(f"[{self.meeting_id}] Google Gemini Live API service unavailable - this is expected behavior")
            # Don't try to recover from Google API service issues
            await self.handle_service_unavailable()
            return
        
        # Log the error to database for other types of errors
        try:
            await self.log_conversation("system", f"ERROR: {str(error)[:200]}")
            if self.session:
                await self.session.say("I experienced a brief connection issue, but I'm back now. Let's continue with our interview.")
        except Exception as log_error:
            print(f"[{self.meeting_id}] Could not log error or inform user: {log_error}")
        
        # Try to recover for non-service errors
        await self.attempt_recovery()
    
    async def handle_service_unavailable(self):
        """Handle Google API service unavailable gracefully"""
        try:
            print(f"[{self.meeting_id}] Handling Google API service unavailable - finalizing interview gracefully")
            # The interview should already be finalized, but ensure it's done
            if not self.is_finalized:
                await self.finalize_interview_evaluation()
            # Don't try to send messages as the API is down
        except Exception as e:
            print(f"[{self.meeting_id}] Error in handle_service_unavailable: {e}")
    
    async def attempt_recovery(self):
        """Attempt to recover from connection errors"""
        try:
            print(f"Attempting to recover session for meeting: {self.meeting_id}")
            # Mark this in the database for tracking
            await self.log_conversation("system", "Attempting session recovery")
            
            # The session will be handled by the pipeline's error recovery mechanisms
            # We just need to ensure our state is consistent
            self.connection_lost = False
            
        except Exception as e:
            print(f"Recovery attempt failed: {e}")
            await self.log_conversation("system", f"Recovery failed: {e}")
        

    @function_tool
    async def evaluate_response(self, question: str, answer: str, category: str = "technical") -> None:
        """Evaluate candidate response and log to database"""
        try:
            self.question_count += 1
            
            # Simple evaluation logic - in production, this could use AI for more sophisticated evaluation
            evaluation = self.analyze_response(question, answer, category)
            
            await self.log_question_and_evaluation(question, answer, evaluation)
            await self.log_conversation("candidate", answer)
            
            print(f"Evaluated response for question {self.question_count}: Score {evaluation['overall_score']}/10")
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
    
    def analyze_response(self, question: str, answer: str, category: str) -> Dict:
        """Analyze and score candidate response"""
        # Basic analysis - in production, this could use NLP/AI for more sophisticated evaluation
        answer_length = len(answer.split())
        
        # Simple scoring based on answer length and keywords
        technical_accuracy = min(10, max(1, answer_length // 10))  # Basic length-based scoring
        communication_clarity = min(10, max(1, (answer_length // 5) + 3))
        depth_of_knowledge = min(10, max(1, answer_length // 15))
        relevance = 8 if any(keyword in answer.lower() for keyword in ['java', 'programming', 'code', 'algorithm', 'machine learning', 'model']) else 5
        
        overall_score = round((technical_accuracy + communication_clarity + depth_of_knowledge + relevance) / 4)
        
        # Determine strengths and areas for improvement
        strengths = []
        areas_for_improvement = []
        
        if communication_clarity >= 7:
            strengths.append("Clear communication")
        else:
            areas_for_improvement.append("Could provide more detailed explanations")
            
        if depth_of_knowledge >= 7:
            strengths.append("Good technical depth")
        else:
            areas_for_improvement.append("Could demonstrate deeper technical understanding")
            
        if relevance >= 7:
            strengths.append("Relevant technical knowledge")
        else:
            areas_for_improvement.append("Could focus more on relevant technical concepts")
        
        # Determine next difficulty level
        next_difficulty = "same"
        if overall_score >= 8:
            next_difficulty = "harder"
        elif overall_score <= 4:
            next_difficulty = "easier"
        
        return {
            "difficulty": "medium",  # Default difficulty
            "category": category,
            "technical_accuracy": technical_accuracy,
            "communication_clarity": communication_clarity,
            "depth_of_knowledge": depth_of_knowledge,
            "relevance": relevance,
            "overall_score": overall_score,
            "confidence_level": "medium" if overall_score >= 6 else "low",
            "strengths": strengths,
            "areas_for_improvement": areas_for_improvement,
            "detailed_feedback": f"Response scored {overall_score}/10. Answer length: {answer_length} words.",
            "next_difficulty": next_difficulty,
            "evaluation_approach": "Automated scoring based on response length, clarity, and technical keyword detection",
            "interviewer_notes": f"Question #{self.question_count} in {self.current_phase} phase"
        }

    @function_tool
    async def end_call(self) -> None:
        """End the call upon request by the user"""
        if hasattr(self, '_end_call_executed') and self._end_call_executed:
            print(f"[{self.meeting_id}] end_call already executed, skipping")
            return
            
        try:
            self._end_call_executed = True
            print(f"[{self.meeting_id}] end_call started, session exists: {self.session is not None}")
            
            goodbye_msg = "Thank you for your time today. The interview has been completed and your responses have been recorded. You should hear back from our team soon. Have a great day!"
            
            # Check if session exists before trying to use it
            if self.session:
                try:
                    await self.session.say(goodbye_msg)
                    await self.log_conversation("interviewer", goodbye_msg)
                    print(f"[{self.meeting_id}] Goodbye message sent successfully in end_call")
                except Exception as say_error:
                    print(f"[{self.meeting_id}] Failed to send goodbye in end_call: {say_error}")
                
                await self.finalize_interview_evaluation()
                await asyncio.sleep(1)
                
                try:
                    await self.session.leave()
                    print(f"[{self.meeting_id}] Session left successfully")
                except Exception as leave_error:
                    print(f"[{self.meeting_id}] Failed to leave session: {leave_error}")
                    # Set session to None if leave fails to prevent further attempts
                    self.session = None
            else:
                print(f"[{self.meeting_id}] Session is None in end_call, cannot send goodbye message")
                # Still finalize the evaluation even if session is gone
                await self.finalize_interview_evaluation()
                
        except Exception as e:
            print(f"[{self.meeting_id}] Error ending call: {e}")
            self._end_call_executed = False  # Reset flag if there was an error
  

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
    candidate_data: Dict = None  # Optional candidate data from access code verification


class LeaveAgentReqConfig(BaseModel): # For the leave endpoint
    meeting_id: str

async def server_operations(req: MeetingReqConfig):
    print(f"req body : {req}")
    meeting_id = req.meeting_id
    print(f"[{meeting_id}] Initializing agent operations...")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Use all values from the request with enhanced configuration
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
                agent=MyVoiceAgent(req.system_prompt, req.personality, meeting_id, req.candidate_data),
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
                print(f"[{meeting_id}] Agent attempting to start... (Attempt {retry_count + 1}/{max_retries})")
                await session.start()
                print(f"[{meeting_id}] Agent session.start() completed normally.")
                break  # Success, exit retry loop
                
            except Exception as ex:
                retry_count += 1
                print(f"[{meeting_id}] [ERROR] in agent session (Attempt {retry_count}/{max_retries}): {ex}")
                print(f"[{meeting_id}] Full traceback: {traceback.format_exc()}")
                
                # Clean up failed session
                if active_sessions.get(meeting_id) is session:
                    active_sessions.pop(meeting_id, None)
                try:
                    if hasattr(session, 'leave') and session.leave is not None:
                        await session.leave()
                except Exception as leave_ex:
                    print(f"[{meeting_id}] [ERROR] during cleanup after failed start: {leave_ex}")
                
                # If this was the last attempt, re-raise the error
                if retry_count >= max_retries:
                    print(f"[{meeting_id}] Max retries ({max_retries}) reached. Giving up.")
                    raise ex
                
                # Wait before retrying
                print(f"[{meeting_id}] Waiting 2 seconds before retry...")
                await asyncio.sleep(2)
        
        except Exception as setup_ex:
            retry_count += 1
            print(f"[{meeting_id}] [ERROR] during setup (Attempt {retry_count}/{max_retries}): {setup_ex}")
            print(f"[{meeting_id}] Full setup traceback: {traceback.format_exc()}")
            
            # Clean up if session was created
            if meeting_id in active_sessions:
                active_sessions.pop(meeting_id, None)
            
            # If this was the last attempt, re-raise the error
            if retry_count >= max_retries:
                print(f"[{meeting_id}] Max setup retries ({max_retries}) reached. Giving up.")
                raise setup_ex
            
            # Wait before retrying
            print(f"[{meeting_id}] Waiting 3 seconds before setup retry...")
            await asyncio.sleep(3)


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


@app.get("/test")
async def test():
    return {"message": "Server is running!"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    }






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
