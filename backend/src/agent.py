import logging
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    JobContext, 
    WorkerOptions, 
    cli, 
    RoomInputOptions,
    JobProcess,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation

logger = logging.getLogger("tutor-agent")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")

# --- 1. GLOBAL CONTENT LOADER ---

def load_tutor_content():
    """Load tutor content from JSON file."""
    content_path = Path(__file__).parent.parent / "shared-data" / "day4_tutor_content.json"
    try:
        with open(content_path, "r") as f:
            content = json.load(f)
            return content
    except FileNotFoundError:
        logger.error(f"Content file not found: {content_path}.")
        return [{"id": "variables", "title": "Variables", "summary": "Variables store values.", "sample_question": "What is a variable?"}]
    
TUTOR_CONTENT = load_tutor_content()

def get_concepts_list() -> str:
    return ", ".join(concept["title"] for concept in TUTOR_CONTENT)

def get_concept_by_keyword(keyword: str) -> Optional[dict]:
    keyword_lower = keyword.lower()
    for concept in TUTOR_CONTENT:
        if (keyword_lower in concept["title"].lower() or 
            keyword_lower in concept["id"].lower() or 
            concept["id"] == keyword_lower):
            return concept
    return None

# --- 2. MULTI-AGENT ARCHITECTURE SETUP ---

@dataclass
class UserData:
    """Stores data and agents to be shared across the session"""
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None

    def summarize(self) -> str:
        return f"Current concepts available: {get_concepts_list()}"

RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    """
    Base class that handles the complex logic of switching between agents
    and preserving the chat history context.
    """
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        userdata: UserData = self.session.userdata
        
        # FIX: Check if connected before accessing local_participant
        if userdata.ctx and userdata.ctx.room and userdata.ctx.room.connection_state == "connected":
            try:
                # Tag the participant in the room so we know who is active
                await userdata.ctx.room.local_participant.set_attributes({"agent": agent_name})
            except Exception as e:
                logger.warning(f"Could not set agent attribute: {e}")

        # --- Context Preservation Logic ---
        chat_ctx = self.chat_ctx.copy()

        if userdata.prev_agent:
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # Inject the system prompt for this specific agent
        chat_ctx.add_message(
            role="system",
            content=f"{self.instructions}" 
        )
        await self.update_chat_ctx(chat_ctx)
        
        # Say hello immediately upon entering
        await self.session.generate_reply()

    def _truncate_chat_ctx(
        self,
        items: list,
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list:
        """Truncate the chat context to keep only recent relevant messages."""
        def _valid_item(item) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in ["function_call", "function_call_output"]:
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Agent:
        """The magic function that performs the handoff"""
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.personas[name]
        userdata.prev_agent = current_agent
        
        logger.info(f"Transferring from {current_agent.__class__.__name__} to {name}")
        return next_agent

# --- 3. DEFINING THE SPECIFIC AGENTS ---

class CoordinatorAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are the Coordinator.
            Welcome users to the coding tutor!
            Available Modes:
            1. Learn (Teacher Matthew will explain concepts)
            2. Quiz (Alicia will test you)
            3. Teach-Back (Ken will listen to your explanation)
            
            Ask the user what they want to do. If they choose a mode, use the transfer tools.
            """,
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(voice="en-US-matthew", style="Conversation"),
            vad=silero.VAD.load()
        )

    @function_tool
    async def start_learn_mode(self, context: RunContext_T) -> Agent:
        """Switch to Learn Mode (Matthew)."""
        await self.session.say("Great choice! I'll hand you over to Matthew for the lesson.")
        return await self._transfer_to_agent("learn", context)

    @function_tool
    async def start_quiz_mode(self, context: RunContext_T) -> Agent:
        """Switch to Quiz Mode (Alicia)."""
        await self.session.say("Time to test your knowledge! Here is Alicia.")
        return await self._transfer_to_agent("quiz", context)
    
    @function_tool
    async def start_teach_back_mode(self, context: RunContext_T) -> Agent:
        """Switch to Teach-Back Mode (Ken)."""
        await self.session.say("Teaching is the best way to learn. Ken is ready to listen.")
        return await self._transfer_to_agent("teach_back", context)


class LearnAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are Matthew (Learn Mode). 
            Explain these concepts: {get_concepts_list()}.
            Use analogies. Keep it brief. 
            If the user wants to quiz or stop, ask the coordinator.""",
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(voice="en-US-matthew", style="Conversation"),
            vad=silero.VAD.load()
        )

    @function_tool
    async def get_concept_details(self, context: RunContext_T, keyword: str) -> str:
        """Get details to explain a concept."""
        concept = get_concept_by_keyword(keyword)
        if concept: return f"Summary: {concept['summary']}"
        return "Concept not found."

    @function_tool
    async def return_to_coordinator(self, context: RunContext_T) -> Agent:
        """Go back to the main menu/coordinator."""
        return await self._transfer_to_agent("coordinator", context)


class QuizAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are Alicia (Quiz Mode). 
            Ask questions about: {get_concepts_list()}.
            Verify their answer. be energetic!""",
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(voice="en-US-alicia", style="Conversation"),
            vad=silero.VAD.load()
        )

    @function_tool
    async def get_quiz_question(self, context: RunContext_T, keyword: str) -> str:
        """Get a specific question for a concept."""
        concept = get_concept_by_keyword(keyword)
        if concept: return f"Question: {concept['sample_question']}"
        return "Concept not found."

    @function_tool
    async def return_to_coordinator(self, context: RunContext_T) -> Agent:
        """Go back to the main menu."""
        return await self._transfer_to_agent("coordinator", context)


class TeachBackAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""You are Ken (Teach-Back Mode). 
            Ask the user to explain a concept from: {get_concepts_list()}.
            Listen to them, then grade their explanation.""",
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(voice="en-US-ken", style="Conversation"),
            vad=silero.VAD.load()
        )

    @function_tool
    async def get_target_concept(self, context: RunContext_T, keyword: str) -> str:
        """Check what the user should be explaining."""
        concept = get_concept_by_keyword(keyword)
        if concept: return f"They should explain: {concept['summary']}"
        return "Concept not found."

    @function_tool
    async def return_to_coordinator(self, context: RunContext_T) -> Agent:
        """Go back to the main menu."""
        return await self._transfer_to_agent("coordinator", context)


# --- 4. PREWARM & ENTRYPOINT ---

def prewarm(proc: JobProcess):
    """Preload VAD model."""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # 1. Initialize the shared data structure
    userdata = UserData(ctx=ctx)

    # 2. Create instances of all agents
    coordinator = CoordinatorAgent()
    learn_agent = LearnAgent()
    quiz_agent = QuizAgent()
    teach_back_agent = TeachBackAgent()

    # 3. Register them in the dictionary
    userdata.personas.update({
        "coordinator": coordinator,
        "learn": learn_agent,
        "quiz": quiz_agent,
        "teach_back": teach_back_agent
    })

    # 4. Create the session with UserData
    session = AgentSession[UserData](userdata=userdata)
    
    # --- CRITICAL FIX: Connect FIRST, then Start ---
    # We must connect first so on_enter can access the participant attributes
    await ctx.connect()

    # 5. Start with the Coordinator
    await session.start(
        agent=coordinator,
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))