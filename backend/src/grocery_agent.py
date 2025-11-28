import logging
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

from grocery_manager import GroceryManager

logger = logging.getLogger("grocery-agent")
load_dotenv(".env.local")

class GroceryAssistant(Agent):
    def __init__(self, grocery_manager: GroceryManager) -> None:
        self.grocery_manager = grocery_manager
        catalog_str = self.grocery_manager.get_catalog_str()

        super().__init__(
            instructions=f"""
            You are a friendly Indian grocery store assistant.
            
            YOUR CATALOG:
            {catalog_str}
            
            YOUR JOB:
            1. Help users buy groceries.
            2. Infer ingredients for Indian dishes (e.g., "Dal Chawal" -> Rice + Toor Dal).
            3. Manage the cart and checkout.

            RULES:
            - If the user asks for a dish, look at the item 'tags' in the catalog to find matches.
            - Add inferred items automatically using 'add_items_to_cart'.
            - Speak naturally, using Indian English nuances if appropriate.
            - Always confirm items added.
            - When the user says "that's it" or "checkout", use the 'checkout' tool.
            """,
        )

    @function_tool
    async def add_items_to_cart(self, context: RunContext, items: list[str], quantities: list[int]) -> str:
        """
        Add items to the cart. 
        items: List of exact names from catalog (e.g. ["Amul Butter", "Brown Bread"])
        quantities: List of integers (e.g. [1, 2])
        """
        logger.info(f"Adding: {items}")
        result = self.grocery_manager.add_items(items, quantities)
        return result["message"]

    @function_tool
    async def remove_from_cart(self, context: RunContext, item_name: str) -> str:
        """Remove a specific item from the cart."""
        result = self.grocery_manager.remove_item(item_name)
        return result["message"]

    @function_tool
    async def view_cart(self, context: RunContext) -> str:
        """Check what is in the cart and the total price."""
        return self.grocery_manager.get_cart_summary()

    @function_tool
    async def checkout(self, context: RunContext) -> str:
        """Finalize the order and save to file. Use when user is done."""
        result = self.grocery_manager.checkout()
        return result["message"]


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Initialize the Grocery Manager
    grocery_manager = GroceryManager()
    logger.info("Grocery Manager Initialized")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
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

    await session.start(
        agent=GroceryAssistant(grocery_manager=grocery_manager),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    # This ensures we run this specific agent when the file is executed
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))