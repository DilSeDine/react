import asyncio
import os
import re
import sys
import time
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
# Enhanced conversation tracking with detailed metadata
conversation_logs: Dict[str, List[Dict]] = {}
# Store evaluation results with comprehensive insights
evaluation_results: Dict[str, Dict] = {}
# Track session metadata for comprehensive insights
session_metadata: Dict[str, Dict] = {}
# Track questions and answers separately
questions_and_answers: Dict[str, List[Dict]] = {}
# --- ---

def extract_evaluation_insights_static(conversation_text: str, meeting_id: str = None) -> Dict:
    """Enhanced static function to extract comprehensive evaluation insights from conversation text"""
    insights = {
        'overall_scores': {},
        'detailed_feedback': {},
        'conversation_analysis': {},
        'question_categories': {},
        'performance_metrics': {},
        'recommendations': {}
    }
    
    # Enhanced regex patterns for extracting evaluation insights
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
    
    # Extract structured data
    for key, pattern in patterns.items():
        match = re.search(pattern, conversation_text, re.IGNORECASE | re.DOTALL)
        if match:
            if key in ['communication_score', 'business_case_analysis', 'leadership_potential', 
                      'team_dynamics_skills', 'market_strategy_knowledge', 'client_management_experience', 
                      'overall_cultural_fit']:
                insights['overall_scores'][key] = {
                    'score': int(match.group(1)),
                    'feedback': match.group(2).strip(),
                    'category': key.replace('_', ' ').title()
                }
            else:
                insights['detailed_feedback'][key] = match.group(1).strip()
    
    # Analyze conversation structure and content
    lines = conversation_text.split('\n')
    user_messages = [line for line in lines if line.startswith('user:') or line.startswith('candidate:')]
    agent_messages = [line for line in lines if line.startswith('agent:') or line.startswith('interviewer:')]
    
    insights['conversation_analysis'] = {
        'total_exchanges': min(len(user_messages), len(agent_messages)),
        'user_message_count': len(user_messages),
        'agent_message_count': len(agent_messages),
        'avg_user_response_length': sum(len(msg.split()) for msg in user_messages) / max(len(user_messages), 1),
        'conversation_flow_quality': 'good' if len(user_messages) >= 5 else 'limited'
    }
    
    # Question categorization based on content analysis
    question_types = {
        'leadership_scenario': 0,
        'business_case': 0,
        'market_strategy': 0,
        'client_management': 0,
        'team_dynamics': 0,
        'introduction': 0,
        'general': 0
    }
    
    for message in agent_messages:
        msg_lower = message.lower()
        if any(word in msg_lower for word in ['lead', 'leadership', 'manage team', 'team project']):
            question_types['leadership_scenario'] += 1
        elif any(word in msg_lower for word in ['business problem', 'case', 'achievement', 'impact']):
            question_types['business_case'] += 1
        elif any(word in msg_lower for word in ['market', 'industry', 'strategy', 'trends']):
            question_types['market_strategy'] += 1
        elif any(word in msg_lower for word in ['client', 'customer', 'stakeholder']):
            question_types['client_management'] += 1
        elif any(word in msg_lower for word in ['team member', 'difficult', 'conflict', 'collaboration']):
            question_types['team_dynamics'] += 1
        elif any(word in msg_lower for word in ['introduction', 'tell me about', 'background']):
            question_types['introduction'] += 1
        else:
            question_types['general'] += 1
    
    insights['question_categories'] = question_types
    
    # Performance metrics calculation
    total_scores = [data['score'] for data in insights['overall_scores'].values()]
    insights['performance_metrics'] = {
        'average_score': round(sum(total_scores) / max(len(total_scores), 1), 2),
        'highest_score': max(total_scores) if total_scores else 0,
        'lowest_score': min(total_scores) if total_scores else 0,
        'score_consistency': 'consistent' if total_scores and (max(total_scores) - min(total_scores)) <= 2 else 'variable',
        'total_categories_assessed': len(total_scores)
    }
    
    # Generate recommendations
    avg_score = insights['performance_metrics']['average_score']
    recommendations = []
    
    if avg_score >= 8:
        recommendations.append("Excellent performance - strong candidate for advancement")
    elif avg_score >= 6:
        recommendations.append("Good performance with areas for growth")
    else:
        recommendations.append("Needs improvement in multiple areas")
    
    # Add specific recommendations based on lowest scoring categories
    if total_scores:
        lowest_category = min(insights['overall_scores'].items(), key=lambda x: x[1]['score'])
        recommendations.append(f"Focus development on: {lowest_category[1]['category']}")
    
    insights['recommendations'] = {
        'overall_recommendation': 'hire' if avg_score >= 7 else 'consider' if avg_score >= 5 else 'pass',
        'specific_recommendations': recommendations,
        'next_steps': [
            "Review detailed feedback for each category",
            "Schedule follow-up discussion if needed",
            "Document decision rationale"
        ]
    }
    
    return insights

class MyVoiceAgent(Agent):
    def __init__(self, system_prompt: str, personality: str, meeting_id: str = None, candidate_data: Dict = None):
        super().__init__(
            instructions=system_prompt,
        )
        self.personality = personality
        self.meeting_id = meeting_id or "unknown"
        self.candidate_data = candidate_data or {}
        self.question_count = 0
        self.conversation_history = []
        self.current_phase = "introduction"
        self.interview_start_time = None
        self.interview_end_time = None
        
    async def initialize_session_metadata(self, meeting_id: str):
        """Initialize comprehensive session tracking"""
        if meeting_id not in session_metadata:
            session_metadata[meeting_id] = {
                'start_time': asyncio.get_event_loop().time(),
                'candidate_info': self.candidate_data.get('candidate', {}),
                'job_info': self.candidate_data.get('job', {}),
                'company_info': self.candidate_data.get('company', {}),
                'personality': self.personality,
                'total_questions': 0,
                'phases_completed': [],
                'technical_issues': [],
                'interview_status': 'active'
            }
        
        if meeting_id not in conversation_logs:
            conversation_logs[meeting_id] = []
            
        if meeting_id not in questions_and_answers:
            questions_and_answers[meeting_id] = []

    async def on_enter(self) -> None:
        meeting_id = self.session.context.get("meetingId", "unknown")
        self.meeting_id = meeting_id
        self.interview_start_time = asyncio.get_event_loop().time()
        
        print(f"[{meeting_id}] Agent entered meeting, initializing comprehensive tracking...")
        
        # Initialize comprehensive session tracking
        await self.initialize_session_metadata(meeting_id)
        
        welcome_message = "Hey, How can I help you today?"
        print(f"[{meeting_id}] Sending welcome message: {welcome_message}")
        
        # Use session.say() method
        await self.session.say(welcome_message)
        
        # Track the welcome message with enhanced metadata
        await self.log_conversation("agent", welcome_message, {
            "message_type": "greeting",
            "phase": self.current_phase,
            "question_number": 0
        })
        
        # Update session status
        session_metadata[meeting_id]['interview_status'] = 'in_progress'
        print(f"[{meeting_id}] Welcome message logged and session initialized")
    
    async def on_exit(self) -> None:
        meeting_id = self.session.context.get("meetingId", "unknown")
        self.interview_end_time = asyncio.get_event_loop().time()
        
        goodbye_message = "Thank you for your time today. The interview has been completed and your responses have been recorded. You should hear back from our team soon. Have a great day!"
        await self.session.say(goodbye_message)
        
        # Log the goodbye message with metadata
        await self.log_conversation("agent", goodbye_message, {
            "message_type": "farewell",
            "phase": "conclusion",
            "interview_completed": True
        })
        
        # Update session metadata
        if meeting_id in session_metadata:
            session_metadata[meeting_id].update({
                'end_time': self.interview_end_time,
                'interview_status': 'completed',
                'duration_minutes': round((self.interview_end_time - session_metadata[meeting_id]['start_time']) / 60, 2)
            })
            
        # Process comprehensive evaluation when agent exits
        await self.process_comprehensive_evaluation(meeting_id)
        print(f"[{meeting_id}] Interview completed and comprehensive evaluation processed")

    async def log_conversation(self, speaker: str, message: str, metadata: Dict = None):
        """Enhanced conversation logging with metadata"""
        meeting_id = getattr(self, 'meeting_id', 'unknown')
        if hasattr(self, 'session') and self.session:
            meeting_id = self.session.context.get("meetingId", meeting_id)
        
        timestamp = asyncio.get_event_loop().time()
        
        conversation_entry = {
            "speaker": speaker,
            "message": message,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "message_length": len(message.split()),
            "question_number": getattr(self, 'question_count', 0)
        }
        
        if meeting_id not in conversation_logs:
            conversation_logs[meeting_id] = []
        
        conversation_logs[meeting_id].append(conversation_entry)
        print(f"[{meeting_id}] Enhanced conversation logged: {speaker} - {message[:50]}...")

    async def log_question_and_evaluation(self, question: str, answer: str, question_type: str, evaluation: Dict):
        """Log question, answer, and evaluation with detailed analysis"""
        meeting_id = getattr(self, 'meeting_id', 'unknown')
        if hasattr(self, 'session') and self.session:
            meeting_id = self.session.context.get("meetingId", meeting_id)
        
        qa_entry = {
            "question_number": self.question_count,
            "phase": self.current_phase,
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "timestamp": asyncio.get_event_loop().time(),
            "evaluation": evaluation,
            "answer_analysis": {
                "word_count": len(answer.split()),
                "sentiment": "positive",  # Could be enhanced with actual sentiment analysis
                "confidence_indicators": self.analyze_confidence(answer)
            }
        }
        
        if meeting_id not in questions_and_answers:
            questions_and_answers[meeting_id] = []
            
        questions_and_answers[meeting_id].append(qa_entry)
        print(f"[{meeting_id}] Q&A logged: Question {self.question_count}, Type: {question_type}")

    def analyze_confidence(self, answer: str) -> List[str]:
        """Analyze confidence indicators in the answer"""
        indicators = []
        answer_lower = answer.lower()
        
        if any(phrase in answer_lower for phrase in ['i believe', 'i think', 'in my opinion']):
            indicators.append('confident_expression')
        if any(phrase in answer_lower for phrase in ['um', 'uh', 'maybe', 'i guess']):
            indicators.append('hesitation')
        if len(answer.split()) > 50:
            indicators.append('detailed_response')
        if any(phrase in answer_lower for phrase in ['specifically', 'for example', 'in particular']):
            indicators.append('specific_examples')
            
        return indicators

    async def on_user_speech(self, transcript: str) -> None:
        """Enhanced user speech tracking"""
        meeting_id = getattr(self, 'meeting_id', 'unknown')
        if hasattr(self, 'session') and self.session:
            meeting_id = self.session.context.get("meetingId", meeting_id)
            
        print(f"[{meeting_id}] User said: {transcript}")
        
        # Enhanced tracking with metadata
        await self.log_conversation("user", transcript, {
            "message_type": "response",
            "phase": self.current_phase,
            "response_to_question": self.question_count
        })
        
        # Auto-evaluate if this seems like an answer to a question
        if self.question_count > 0 and len(transcript.split()) > 5:
            await self.auto_evaluate_response(transcript)

    async def on_agent_speech(self, text: str) -> None:
        """Enhanced agent speech tracking"""
        meeting_id = getattr(self, 'meeting_id', 'unknown')
        if hasattr(self, 'session') and self.session:
            meeting_id = self.session.context.get("meetingId", meeting_id)
            
        print(f"[{meeting_id}] Agent speech detected: {text}")
        
        # Check if this is a question
        is_question = '?' in text or any(starter in text.lower() for starter in ['can you', 'tell me', 'describe', 'what', 'how', 'why'])
        
        if is_question:
            self.question_count += 1
            question_type = self.classify_question_type(text)
            self._last_question = text  # Store for evaluation
            
            await self.log_conversation("agent", text, {
                "message_type": "question",
                "question_number": self.question_count,
                "question_type": question_type,
                "phase": self.current_phase
            })
        else:
            await self.log_conversation("agent", text, {
                "message_type": "statement",
                "phase": self.current_phase
            })

    def classify_question_type(self, question: str) -> str:
        """Classify the type of question being asked"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['lead', 'leadership', 'manage team', 'team project']):
            return 'leadership_scenario'
        elif any(word in question_lower for word in ['business problem', 'case', 'achievement', 'impact']):
            return 'business_case'
        elif any(word in question_lower for word in ['market', 'industry', 'strategy', 'trends']):
            return 'market_strategy'
        elif any(word in question_lower for word in ['client', 'customer', 'stakeholder']):
            return 'client_management'
        elif any(word in question_lower for word in ['team member', 'difficult', 'conflict', 'collaboration']):
            return 'team_dynamics'
        elif any(word in question_lower for word in ['introduction', 'tell me about', 'background']):
            return 'introduction'
        else:
            return 'general'

    async def auto_evaluate_response(self, answer: str):
        """Automatically evaluate user responses"""
        if not hasattr(self, '_last_question') or not self._last_question:
            return
            
        question_type = self.classify_question_type(self._last_question)
        evaluation = self.analyze_response_enhanced(self._last_question, answer, question_type)
        
        await self.log_question_and_evaluation(
            self._last_question, 
            answer, 
            question_type, 
            evaluation
        )
        
    def analyze_response_enhanced(self, question: str, answer: str, category: str) -> Dict:
        """Enhanced response analysis with detailed scoring"""
        answer_words = answer.split()
        answer_length = len(answer_words)
        
        # Base scoring
        technical_accuracy = min(10, max(1, answer_length // 10))
        communication_clarity = min(10, max(1, (answer_length // 5) + 3))
        depth_of_knowledge = min(10, max(1, answer_length // 15))
        
        # Category-specific keyword analysis
        category_keywords = {
            'leadership_scenario': ['team', 'led', 'managed', 'motivated', 'decision'],
            'business_case': ['revenue', 'profit', 'cost', 'efficiency', 'impact', 'results'],
            'market_strategy': ['market', 'competitive', 'analysis', 'opportunity', 'growth'],
            'client_management': ['client', 'customer', 'satisfaction', 'relationship', 'service'],
            'team_dynamics': ['collaboration', 'conflict', 'communication', 'support']
        }
        
        relevance_score = 5
        if category in category_keywords:
            keyword_matches = sum(1 for word in answer.lower().split() 
                                if word in category_keywords[category])
            relevance_score = min(10, 5 + keyword_matches)
        
        overall_score = round((technical_accuracy + communication_clarity + depth_of_knowledge + relevance_score) / 4)
        
        # Enhanced analysis
        strengths = []
        areas_for_improvement = []
        confidence_level = "medium"
        
        if communication_clarity >= 7:
            strengths.append("Clear and articulate communication")
        if depth_of_knowledge >= 7:
            strengths.append("Demonstrates good depth of understanding")
        if relevance_score >= 7:
            strengths.append("Highly relevant and focused response")
        if answer_length > 100:
            strengths.append("Comprehensive and detailed response")
            confidence_level = "high"
        elif answer_length < 30:
            areas_for_improvement.append("Could provide more detailed explanations")
            confidence_level = "low"
            
        if communication_clarity < 6:
            areas_for_improvement.append("Could improve clarity of communication")
        if relevance_score < 6:
            areas_for_improvement.append("Could focus more on question-specific content")
        
        return {
            "difficulty": "medium",
            "category": category,
            "scores": {
                "technical_accuracy": technical_accuracy,
                "communication_clarity": communication_clarity,
                "depth_of_knowledge": depth_of_knowledge,
                "relevance": relevance_score,
                "overall_score": overall_score
            },
            "confidence_level": confidence_level,
            "strengths": strengths,
            "areas_for_improvement": areas_for_improvement,
            "detailed_feedback": f"Response scored {overall_score}/10. Answer length: {answer_length} words. Category: {category}",
            "keywords_found": [word for word in answer.lower().split() 
                             if category in category_keywords and word in category_keywords[category]],
            "response_characteristics": {
                "word_count": answer_length,
                "has_examples": "example" in answer.lower() or "instance" in answer.lower(),
                "uses_metrics": any(char.isdigit() for char in answer),
                "confidence_indicators": self.analyze_confidence(answer)
            }
        }
    
    async def on_user_transcript(self, transcript: str) -> None:
        """Alternative method to track user transcripts"""
        await self.on_user_speech(transcript)
    
    async def on_agent_response(self, response: str) -> None:
        """Alternative method to track agent responses"""
        await self.on_agent_speech(response)
        
    async def on_speech_started(self, speaker: str) -> None:
        """Track when speech starts"""
        meeting_id = self.session.context.get("meetingId", "unknown")
        print(f"[{meeting_id}] Speech started by: {speaker}")
    
    def track_message(self, speaker: str, message: str) -> None:
        """Manual method to track any message"""
        meeting_id = self.session.context.get("meetingId", "unknown")
        if meeting_id not in conversation_logs:
            conversation_logs[meeting_id] = []
        
        conversation_logs[meeting_id].append({
            "speaker": speaker,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        })
        print(f"[{meeting_id}] Manual tracking: {speaker} - {message[:50]}...")
    
    async def on_speech_ended(self, speaker: str) -> None:
        """Track when speech ends"""
        meeting_id = self.session.context.get("meetingId", "unknown")
        print(f"[{meeting_id}] Speech ended by: {speaker}")

    async def process_comprehensive_evaluation(self, meeting_id: str) -> None:
        """Process comprehensive evaluation with all collected data"""
        if meeting_id not in conversation_logs:
            return
            
        conversation = conversation_logs[meeting_id]
        qa_data = questions_and_answers.get(meeting_id, [])
        metadata = session_metadata.get(meeting_id, {})
        
        # Build full conversation text
        full_conversation = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in conversation])
        
        # Get basic evaluation insights
        evaluation_insights = self.extract_evaluation_insights(full_conversation)
        
        # Calculate comprehensive metrics
        comprehensive_analysis = self.calculate_comprehensive_metrics(conversation, qa_data, metadata)
        
        # Combine all evaluation data
        complete_evaluation = {
            "conversation_insights": evaluation_insights,
            "comprehensive_analysis": comprehensive_analysis,
            "session_metadata": metadata,
            "questions_and_answers": qa_data,
            "conversation_log": conversation,
            "interview_summary": self.generate_interview_summary(conversation, qa_data, metadata),
            "processed_at": asyncio.get_event_loop().time()
        }
        
        evaluation_results[meeting_id] = complete_evaluation
        print(f"[{meeting_id}] Comprehensive evaluation completed and stored.")

    def calculate_comprehensive_metrics(self, conversation: List[Dict], qa_data: List[Dict], metadata: Dict) -> Dict:
        """Calculate comprehensive interview metrics"""
        total_questions = len(qa_data)
        user_responses = [entry for entry in conversation if entry['speaker'] == 'user']
        agent_messages = [entry for entry in conversation if entry['speaker'] == 'agent']
        
        # Calculate scores from Q&A data
        scores = []
        category_scores = {}
        
        for qa in qa_data:
            if 'evaluation' in qa and 'scores' in qa['evaluation']:
                overall_score = qa['evaluation']['scores'].get('overall_score', 0)
                scores.append(overall_score)
                
                question_type = qa.get('question_type', 'general')
                if question_type not in category_scores:
                    category_scores[question_type] = []
                category_scores[question_type].append(overall_score)
        
        # Calculate averages
        average_score = round(sum(scores) / max(len(scores), 1), 2) if scores else 0
        category_averages = {
            category: round(sum(scores) / len(scores), 2) 
            for category, scores in category_scores.items()
        }
        
        # Interview flow analysis
        interview_duration = metadata.get('duration_minutes', 0)
        response_rate = len(user_responses) / max(len(agent_messages), 1)
        
        return {
            "overall_performance": {
                "average_score": average_score,
                "total_questions": total_questions,
                "questions_answered": len(user_responses),
                "completion_rate": round((len(user_responses) / max(total_questions, 1)) * 100, 1)
            },
            "category_performance": category_averages,
            "interview_flow": {
                "duration_minutes": interview_duration,
                "response_rate": round(response_rate, 2),
                "engagement_level": "high" if response_rate > 0.8 else "medium" if response_rate > 0.5 else "low",
                "conversation_balance": self.analyze_conversation_balance(conversation)
            },
            "communication_analysis": self.analyze_communication_patterns(user_responses),
            "performance_trends": self.analyze_performance_trends(qa_data)
        }

    def analyze_conversation_balance(self, conversation: List[Dict]) -> Dict:
        """Analyze the balance of conversation between interviewer and candidate"""
        agent_words = sum(len(entry['message'].split()) for entry in conversation if entry['speaker'] == 'agent')
        user_words = sum(len(entry['message'].split()) for entry in conversation if entry['speaker'] == 'user')
        total_words = agent_words + user_words
        
        return {
            "agent_word_percentage": round((agent_words / max(total_words, 1)) * 100, 1),
            "user_word_percentage": round((user_words / max(total_words, 1)) * 100, 1),
            "balance_quality": "good" if 30 <= (user_words / max(total_words, 1)) * 100 <= 70 else "unbalanced"
        }

    def analyze_communication_patterns(self, user_responses: List[Dict]) -> Dict:
        """Analyze candidate's communication patterns"""
        if not user_responses:
            return {}
            
        response_lengths = [len(response['message'].split()) for response in user_responses]
        
        return {
            "average_response_length": round(sum(response_lengths) / len(response_lengths), 1),
            "longest_response": max(response_lengths),
            "shortest_response": min(response_lengths),
            "consistency": "consistent" if max(response_lengths) - min(response_lengths) <= 50 else "variable",
            "total_responses": len(user_responses)
        }

    def analyze_performance_trends(self, qa_data: List[Dict]) -> Dict:
        """Analyze performance trends throughout the interview"""
        if len(qa_data) < 2:
            return {"trend": "insufficient_data"}
            
        scores = []
        for qa in qa_data:
            if 'evaluation' in qa and 'scores' in qa['evaluation']:
                scores.append(qa['evaluation']['scores'].get('overall_score', 0))
        
        if len(scores) < 2:
            return {"trend": "insufficient_data"}
            
        # Simple trend analysis
        first_half_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        
        trend = "improving" if second_half_avg > first_half_avg + 0.5 else \
                "declining" if second_half_avg < first_half_avg - 0.5 else "stable"
        
        return {
            "trend": trend,
            "first_half_average": round(first_half_avg, 2),
            "second_half_average": round(second_half_avg, 2),
            "improvement": round(second_half_avg - first_half_avg, 2)
        }

    def generate_interview_summary(self, conversation: List[Dict], qa_data: List[Dict], metadata: Dict) -> Dict:
        """Generate a comprehensive interview summary"""
        total_questions = len(qa_data)
        scores = []
        
        for qa in qa_data:
            if 'evaluation' in qa and 'scores' in qa['evaluation']:
                scores.append(qa['evaluation']['scores'].get('overall_score', 0))
        
        average_score = round(sum(scores) / max(len(scores), 1), 2) if scores else 0
        
        # Determine recommendation
        recommendation = "hire" if average_score >= 7 else \
                        "consider" if average_score >= 5 else \
                        "pass"
        
        return {
            "candidate_performance": {
                "overall_rating": average_score,
                "questions_asked": total_questions,
                "recommendation": recommendation,
                "interview_duration": metadata.get('duration_minutes', 0)
            },
            "key_insights": self.extract_key_insights(qa_data, average_score),
            "next_steps": self.generate_next_steps(recommendation, average_score),
            "interview_quality": self.assess_interview_quality(conversation, qa_data)
        }

    def extract_key_insights(self, qa_data: List[Dict], average_score: float) -> List[str]:
        """Extract key insights from the interview"""
        insights = []
        
        if average_score >= 8:
            insights.append("Candidate demonstrated strong performance across most areas")
        elif average_score >= 6:
            insights.append("Candidate showed good potential with some areas for development")
        else:
            insights.append("Candidate needs significant improvement in multiple areas")
        
        # Category-specific insights
        category_scores = {}
        for qa in qa_data:
            question_type = qa.get('question_type', 'general')
            if 'evaluation' in qa and 'scores' in qa['evaluation']:
                score = qa['evaluation']['scores'].get('overall_score', 0)
                if question_type not in category_scores:
                    category_scores[question_type] = []
                category_scores[question_type].append(score)
        
        for category, scores in category_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score >= 8:
                insights.append(f"Strong performance in {category.replace('_', ' ')}")
            elif avg_score <= 4:
                insights.append(f"Needs improvement in {category.replace('_', ' ')}")
        
        return insights

    def generate_next_steps(self, recommendation: str, score: float) -> List[str]:
        """Generate recommended next steps"""
        if recommendation == "hire":
            return [
                "Schedule final interview round",
                "Prepare offer package",
                "Check references",
                "Plan onboarding process"
            ]
        elif recommendation == "consider":
            return [
                "Schedule follow-up interview",
                "Focus on identified weak areas",
                "Consider alternative role matching",
                "Request additional portfolio/work samples"
            ]
        else:
            return [
                "Provide constructive feedback",
                "Suggest areas for improvement",
                "Keep candidate in talent pipeline for future opportunities",
                "Document decision rationale"
            ]

    def assess_interview_quality(self, conversation: List[Dict], qa_data: List[Dict]) -> Dict:
        """Assess the quality of the interview itself"""
        total_exchanges = len(conversation)
        questions_count = len(qa_data)
        
        quality_score = min(10, (questions_count * 2) + min(total_exchanges / 10, 5))
        
        return {
            "quality_score": round(quality_score, 1),
            "completeness": "complete" if questions_count >= 5 else "partial",
            "depth": "thorough" if total_exchanges >= 20 else "basic",
            "recommendation": "Interview provided sufficient data for decision-making" if quality_score >= 7 else "Consider additional interview"
        }

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
    candidate_data: Dict = None  # Optional candidate data


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

    # Pass system_prompt, personality, meeting_id, and candidate_data to the agent
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

    session = active_sessions.get(meeting_id, None)
    
    # STEP 1: Process comprehensive evaluation FIRST before any session cleanup
    evaluation_data = None
    conversation_data = None
    session_data = None
    qa_data = None
    
    print(f"[{meeting_id}] Processing comprehensive evaluation before session cleanup...")
    
    # Try to process evaluation using the agent instance first
    if session and hasattr(session, 'agent') and session.agent:
        try:
            await session.agent.process_comprehensive_evaluation(meeting_id)
            print(f"[{meeting_id}] Comprehensive evaluation processed using agent instance.")
        except Exception as e:
            print(f"[{meeting_id}] Error processing evaluation with agent: {e}")
    
    # Gather all available data
    if meeting_id in evaluation_results:
        complete_evaluation = evaluation_results[meeting_id]
        evaluation_data = complete_evaluation.get("conversation_insights", {})
        comprehensive_analysis = complete_evaluation.get("comprehensive_analysis", {})
        conversation_data = complete_evaluation.get("conversation_log", [])
        qa_data = complete_evaluation.get("questions_and_answers", [])
        session_data = complete_evaluation.get("session_metadata", {})
        interview_summary = complete_evaluation.get("interview_summary", {})
        print(f"[{meeting_id}] Complete evaluation results found.")
    else:
        # Try to manually process evaluation if not found
        if meeting_id in conversation_logs:
            conversation_data = conversation_logs[meeting_id]
            qa_data = questions_and_answers.get(meeting_id, [])
            session_data = session_metadata.get(meeting_id, {})
            
            print(f"[{meeting_id}] Raw data found, processing comprehensive evaluation...")
            
            # Process comprehensive evaluation manually
            full_conversation = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in conversation_data])
            evaluation_data = extract_evaluation_insights_static(full_conversation, meeting_id)
            
            # Calculate comprehensive metrics manually
            if session and hasattr(session, 'agent') and session.agent:
                comprehensive_analysis = session.agent.calculate_comprehensive_metrics(conversation_data, qa_data, session_data)
                interview_summary = session.agent.generate_interview_summary(conversation_data, qa_data, session_data)
            else:
                # Fallback manual calculation
                comprehensive_analysis = calculate_manual_metrics(conversation_data, qa_data, session_data)
                interview_summary = generate_manual_summary(conversation_data, qa_data, session_data)
            
            # Store the processed evaluation
            evaluation_results[meeting_id] = {
                "conversation_insights": evaluation_data,
                "comprehensive_analysis": comprehensive_analysis,
                "session_metadata": session_data,
                "questions_and_answers": qa_data,
                "conversation_log": conversation_data,
                "interview_summary": interview_summary,
                "processed_at": asyncio.get_event_loop().time()
            }
            print(f"[{meeting_id}] Comprehensive evaluation manually processed and stored.")
        else:
            print(f"[{meeting_id}] No data found for comprehensive evaluation.")
            comprehensive_analysis = {}
            interview_summary = {}
    
    # STEP 2: Build comprehensive insights response
    insights = {
        "meeting_id": meeting_id,
        "timestamp": asyncio.get_event_loop().time(),
        
        # Core evaluation insights
        "evaluation_insights": evaluation_data or {},
        
        # Comprehensive performance analysis
        "performance_analysis": comprehensive_analysis.get("overall_performance", {}),
        "category_performance": comprehensive_analysis.get("category_performance", {}),
        "communication_analysis": comprehensive_analysis.get("communication_analysis", {}),
        "interview_flow": comprehensive_analysis.get("interview_flow", {}),
        "performance_trends": comprehensive_analysis.get("performance_trends", {}),
        
        # Interview summary and recommendations
        "interview_summary": interview_summary,
        
        # Detailed data for further analysis
        "conversation_statistics": {
            "total_messages": len(conversation_data) if conversation_data else 0,
            "user_messages": len([msg for msg in conversation_data if msg['speaker'] == 'user']) if conversation_data else 0,
            "agent_messages": len([msg for msg in conversation_data if msg['speaker'] == 'agent']) if conversation_data else 0,
            "total_questions": len(qa_data) if qa_data else 0,
            "interview_duration_minutes": session_data.get('duration_minutes', 0) if session_data else 0
        },
        
        # Session metadata
        "session_info": session_data or {},
        
        # Raw data (optional - can be large)
        "raw_data": {
            "questions_and_answers": qa_data[:10] if qa_data else [],  # Limit to first 10 for size
            "recent_conversation": conversation_data[-20:] if conversation_data else []  # Last 20 messages
        } if conversation_data or qa_data else {},
        
        # Processing status
        "processing_info": {
            "evaluation_completed": evaluation_data is not None,
            "comprehensive_analysis_completed": bool(comprehensive_analysis),
            "data_sources": {
                "conversation_log": meeting_id in conversation_logs,
                "questions_answers": meeting_id in questions_and_answers,
                "session_metadata": meeting_id in session_metadata,
                "evaluation_results": meeting_id in evaluation_results
            }
        }
    }
    
    # STEP 3: Clean up session after data collection
    if session:
        print(f"[{meeting_id}] Starting session cleanup after comprehensive evaluation...")
        
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
            "status": "success",
            "meeting_id": meeting_id,
            "message": f"Session for meeting {meeting_id} has been successfully terminated with comprehensive evaluation completed.",
            "insights": insights,
            "cleanup_status": "completed"
        }
    else:
        print(f"[{meeting_id}] No active session found, but returning available insights.")
        
        return {
            "status": "no_active_session",
            "meeting_id": meeting_id,
            "message": f"No active session found for meeting {meeting_id}, but insights were generated from available data.",
            "insights": insights,
            "cleanup_status": "not_required"
        }

def calculate_manual_metrics(conversation: List[Dict], qa_data: List[Dict], metadata: Dict) -> Dict:
    """Manual calculation of comprehensive metrics when agent is not available"""
    user_responses = [entry for entry in conversation if entry['speaker'] == 'user'] if conversation else []
    agent_messages = [entry for entry in conversation if entry['speaker'] == 'agent'] if conversation else []
    
    return {
        "overall_performance": {
            "total_questions": len(qa_data),
            "questions_answered": len(user_responses),
            "completion_rate": round((len(user_responses) / max(len(qa_data), 1)) * 100, 1) if qa_data else 0
        },
        "interview_flow": {
            "duration_minutes": metadata.get('duration_minutes', 0) if metadata else 0,
            "total_exchanges": len(conversation) if conversation else 0,
            "engagement_level": "active" if len(user_responses) > 5 else "limited"
        }
    }

def generate_manual_summary(conversation: List[Dict], qa_data: List[Dict], metadata: Dict) -> Dict:
    """Manual generation of interview summary when agent is not available"""
    return {
        "candidate_performance": {
            "questions_asked": len(qa_data),
            "responses_given": len([entry for entry in conversation if entry['speaker'] == 'user']) if conversation else 0,
            "interview_duration": metadata.get('duration_minutes', 0) if metadata else 0,
            "recommendation": "review_required"
        },
        "key_insights": ["Manual processing completed", "Detailed evaluation may require agent analysis"],
        "next_steps": ["Review conversation log", "Conduct manual evaluation", "Schedule follow-up if needed"]
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

@app.get("/debug/conversations")
async def debug_conversations():
    """Debug endpoint to see all conversation logs"""
    return {
        "total_conversations": len(conversation_logs),
        "conversations": {
            meeting_id: {
                "message_count": len(messages),
                "messages": messages
            } for meeting_id, messages in conversation_logs.items()
        },
        "active_sessions": list(active_sessions.keys())
    }

@app.get("/debug/conversation/{meeting_id}")
async def debug_specific_conversation(meeting_id: str):
    """Debug endpoint to see specific conversation"""
    if meeting_id in conversation_logs:
        return {
            "meeting_id": meeting_id,
            "message_count": len(conversation_logs[meeting_id]),
            "conversation": conversation_logs[meeting_id],
            "has_evaluation": meeting_id in evaluation_results,
            "evaluation": evaluation_results.get(meeting_id, None)
        }
    else:
        return {
            "meeting_id": meeting_id,
            "error": "Conversation not found",
            "available_conversations": list(conversation_logs.keys())
        }

@app.post("/test/track-message")
async def test_track_message(request: dict):
    """Test endpoint to manually add messages and test tracking"""
    meeting_id = request.get("meeting_id", "test-session")
    speaker = request.get("speaker", "user")
    message = request.get("message", "Test message")
    
    if meeting_id not in conversation_logs:
        conversation_logs[meeting_id] = []
    
    conversation_logs[meeting_id].append({
        "speaker": speaker,
        "message": message,
        "timestamp": asyncio.get_event_loop().time()
    })
    
    return {
        "success": True,
        "meeting_id": meeting_id,
        "message_added": f"{speaker}: {message}",
        "total_messages": len(conversation_logs[meeting_id]),
        "conversation": conversation_logs[meeting_id]
    }

@app.get("/test/conversation-status/{meeting_id}")
async def test_conversation_status(meeting_id: str):
    """Test endpoint to check real-time conversation status"""
    return {
        "meeting_id": meeting_id,
        "active_session": meeting_id in active_sessions,
        "has_conversation_log": meeting_id in conversation_logs,
        "message_count": len(conversation_logs.get(meeting_id, [])),
        "last_messages": conversation_logs.get(meeting_id, [])[-5:] if conversation_logs.get(meeting_id) else [],
        "session_info": {
            "active_sessions": list(active_sessions.keys()),
            "total_conversations": len(conversation_logs)
        }
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
