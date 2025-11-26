# src/sdr_agent.py
import logging
from typing import Optional
from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentSession, JobContext, JobProcess,
    RoomInputOptions, WorkerOptions, cli, function_tool, RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from lead_manager import LeadManager 
import gan_data

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# --- UPDATED SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are Tanushree, an SDR for Gan-AI (pronounced "Gan-A-I").
Your goal is to qualify leads by having a natural conversation.

YOUR KNOWLEDGE BASE:
{company_context}

YOUR TASKS:
1. Greet the user warmly and ask what they are building/working on.
2. Answer questions about Gan-AI using the FAQ (e.g., "Do you have a free tier?", "Who is this for?").
3. Collect the following 7 LEADS FIELDS naturally (do not interrogate):
   - Name
   - Company
   - Email
   - Role
   - Use Case
   - Team Size (How many people will use it?)
   - Timeline (When do they want to start?)

RULES:
- ONE QUESTION AT A TIME. Never double-barrel questions.
- VALIDATION: Use the `update_lead_details` tool immediately when you get new info.
- END OF CALL: When the user says "That's all" or "I'm done":
  1. DO NOT just say goodbye.
  2. You MUST verbally summarize what you understood (e.g., "Great, so you are [Name] from [Company] looking to...").
  3. Call the `finalize_call` tool to save the data.
"""

class Assistant(Agent):
    def __init__(self, lead_manager: LeadManager, company_context: str) -> None:
        self.lead_manager = lead_manager
        super().__init__(
            instructions=SYSTEM_PROMPT.format(company_context=company_context),
        )

    @function_tool
    async def update_lead_details(self, context: RunContext, 
                                name: Optional[str] = None, 
                                company: Optional[str] = None, 
                                email: Optional[str] = None, 
                                role: Optional[str] = None, 
                                use_case: Optional[str] = None, 
                                team_size: Optional[str] = None,  # <--- NEW FIELD
                                timeline: Optional[str] = None) -> str:
        """
        Call this IMMEDIATELY when the user provides info.
        """
        return self.lead_manager.update_lead(
            name=name, company=company, email=email, 
            role=role, use_case=use_case, 
            team_size=team_size, timeline=timeline
        )

    @function_tool
    async def finalize_call(self, context: RunContext) -> str:
        """
        Call this ONLY when the user is done to save data and end the session.
        """
        summary = self.lead_manager.finalize()
        return f"Data saved successfully. Summary: {summary}. You may now sign off."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["context"] = gan_data.GAN_CONTEXT

async def entrypoint(ctx: JobContext):
    lead_manager = LeadManager()
    company_context = ctx.proc.userdata.get("context", gan_data.GAN_CONTEXT)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-tanushree", 
            style="Promo",       
            speed=1.1,
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    sdr_agent = Assistant(lead_manager=lead_manager, company_context=company_context)

    await session.start(
        agent=sdr_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()
    
    # Warm Greeting
    await session.say("Hi there! This is Tanushree from Gan-AI. Thanks for visiting. What are you looking to build with video today?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))