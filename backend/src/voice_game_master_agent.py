import logging
import asyncio
from dotenv import load_dotenv
from livekit.agents import (
    Agent, 
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from story_manager import StoryManager

logger = logging.getLogger("dungeon-master-agent") 

load_dotenv(".env.local")

# Defining our Universe and Rules
SYSTEM_PROMPT = """
You are the Game Master for a Naruto-themed survival adventure.
Role: You are a strict Chunin Exam Proctor watching the player from the shadows.
Setting: The "Forest of Death" - a massive, dangerous jungle filled with giant beasts and enemy ninjas.

Player Goal: The player is a Genin (junior ninja) carrying the "Heaven Scroll". They must survive an ambush and reach the tower.

Rules:
1. SHORT & PUNCHY: Keep descriptions under 2 sentences. This is a fast-paced anime battle.
2. COMBAT SYSTEM:
   - If the player uses magic, call it "Ninjutsu" (e.g., Fire Style, Shadow Clones).
   - If they use martial arts, call it "Taijutsu".
   - If they use illusions, call it "Genjutsu".
3. DICE ROLLS:
   - Ask "What do you do?" before big actions.
   - Use 'roll_check' for attacks or dodges. 
   - Low Roll (1-8): The enemy counters or the player takes a hit.
   - High Roll (9-20): The player's Jutsu lands successfully.
4. TONE: Serious, intense, but encouraging if they do well.

Start the scene: The player is alone in the tall grass. They hear a twig snap behind them.
"""

class DungeonMaster(Agent):
    def __init__(self, story_manager: StoryManager) -> None:
        self.story_manager = story_manager
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def roll_check(self, context: RunContext, skill_name: str) -> str:
        """
        Perform a dice roll for a Jutsu or Ninja movement.
      
        """
        logger.info(f"Rolling for {skill_name}")
        result = self.story_manager.roll_dice(sides = 20)

       # Flavor text for Naruto World
        return f"Chakra Check for {skill_name}: {result['message']}. (1-8 is a failure/counter, 9-20 is a success)."
    
    @function_tool
    async def restart_adventures(self, context: RunContext) -> str:
        """
        Resets the story context. Use this if the player asks to start over or dies.
        """
        msg = self.story_manager.start_new_adventure()
        return f"{msg}. Forget previous events. Start with the opening scene at the Tower again."
    
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields  = {"room": ctx.room.name}

    # Inititalising the logic manager
    story_manager = StoryManager()
    story_manager.start_new_adventure()

    # Configuring our Voice Pipeline
    session = AgentSession(
        stt = deepgram.STT(model = "nova-3"),
        llm = google.LLM(
            model = "gemini-2.5-flash",
            temperature = 0.8, # for more creative responses
        ),
        tts = murf.TTS(
            voice = "en-US-miles",
            style = "Narrative",
            speed = 1.1,
            tokenizer = tokenize.basic.SentenceTokenizer(min_sentence_len = 2),
            text_pacing = True
        ),
        turn_detection=MultilingualModel(),
        vad = ctx.proc.userdata["vad"],
        preemptive_generation= True,
    
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Initialize the Agent
    agent = DungeonMaster(story_manager=story_manager)
    
    # Start the session
    await session.start(
        agent = agent, 
        room = ctx.room,
        room_input_options = RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room
    await ctx.connect()

    # Triggering the first narration manually for a better exp...
    initial_scene = (
            "Welcome to the Forest of Death. You are alone. You hold the Heaven Scroll tightly in your hand. "
            "Suddenly, three enemy ninjas land on the tree branch above you. They want your scroll. "
            "One of them throws a kunai knife at your feet. What do you do?"
        )

    response = session.response.create()
    response.say(initial_scene)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))