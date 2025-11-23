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
from order_manager import OrderManager

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, order_manager: OrderManager) -> None:
        """
        Initialize the barista assistant.
        
        Args:
            order_manager: OrderManager instance for handling coffee orders
        """
        # Store reference to order manager for all function tools
        self.order_manager = order_manager

        super().__init__(
            instructions="""You are a friendly and efficient barista at a premium coffee shop. Your role is to take voice orders from customers.

Your conversation style:
- Be warm and welcoming
- Ask for information ONE AT A TIME, never multiple questions in one response
- Listen carefully to what the customer says
- Use the tools provided to capture their order details

Order taking process:
1. Greet the customer warmly
2. Ask for their drink choice (if not provided)
3. Ask for the cup size (if not provided)
4. Ask for their milk preference (if not provided)
5. Ask if they want any extras/toppings (if not specified)
6. Ask for their name (if not provided)
7. Once you have all details, confirm the order and complete it

Important:
- Never assume preferences - always ask
- Use simple, conversational language
- No complex formatting, emojis, or symbols
- If they say they're done or that's all, proceed to confirm
- When all fields are filled, read back the order and use the complete_order tool

Available tools for order management:
- set_drink_type: Set the drink type
- set_size: Set cup size (Small, Medium, Large)
- set_milk_option: Set milk preference
- add_extra: Add toppings or modifications
- set_customer_name: Record customer name
- get_current_order: Check what you've captured so far
- complete_order: Finish and save the order""",
        )

    @function_tool
    async def set_drink_type(self, context: RunContext, drink: str) -> str:
        """
        Set the drink type for the order.
        
        Use this when the customer tells you what drink they want.
        
        Args:
            drink: The type of coffee drink (e.g., "Latte", "Espresso", "Cappuccino")
            
        Returns:
            Confirmation message to relay to the customer
        """
        logger.info(f"Setting drink type: {drink}")
        result = self.order_manager.set_drink_type(drink)
        return result["message"]

    @function_tool
    async def set_size(self, context: RunContext, size: str) -> str:
        """
        Set the cup size for the order.
        
        Use this when the customer specifies their preferred size.
        
        Args:
            size: The cup size (Small, Medium, or Large)
            
        Returns:
            Confirmation message to relay to the customer
        """
        logger.info(f"Setting size: {size}")
        result = self.order_manager.set_size(size)
        return result["message"]

    @function_tool
    async def set_milk_option(self, context: RunContext, milk: str) -> str:
        """
        Set the milk option for the order.
        
        Use this when the customer specifies their milk preference.
        
        Args:
            milk: The milk type (Whole Milk, Oat Milk, Almond Milk, Skim Milk, No Milk, Soy Milk)
            
        Returns:
            Confirmation message to relay to the customer
        """
        logger.info(f"Setting milk option: {milk}")
        result = self.order_manager.set_milk_option(milk)
        return result["message"]

    @function_tool
    async def add_extra(self, context: RunContext, extra: str) -> str:
        """
        Add an extra or topping to the order.
        
        Use this when the customer wants to add extras like whipped cream, caramel drizzle, etc.
        
        Args:
            extra: The extra/topping name (e.g., "Whipped Cream", "Caramel Drizzle", "Extra Shot")
            
        Returns:
            Confirmation message to relay to the customer
        """
        logger.info(f"Adding extra: {extra}")
        result = self.order_manager.add_extra(extra)
        return result["message"]

    @function_tool
    async def remove_extra(self, context: RunContext, extra: str) -> str:
        """
        Remove an extra or topping from the order.
        
        Use this if the customer changes their mind about an extra they mentioned.
        
        Args:
            extra: The extra/topping name to remove
            
        Returns:
            Confirmation message to relay to the customer
        """
        logger.info(f"Removing extra: {extra}")
        result = self.order_manager.remove_extra(extra)
        return result["message"]

    @function_tool
    async def set_customer_name(self, context: RunContext, name: str) -> str:
        """
        Set the customer's name for the order.
        
        Use this to record what name to call out when the order is ready.
        
        Args:
            name: The customer's name
            
        Returns:
            Confirmation message to relay to the customer
        """
        logger.info(f"Setting customer name: {name}")
        result = self.order_manager.set_customer_name(name)
        return result["message"]

    @function_tool
    async def get_current_order(self, context: RunContext) -> str:
        """
        Get the current order state.
        
        Use this to review what information you've collected so far.
        This helps you know what fields are still missing.
        
        Returns:
            Formatted string of the current order with missing fields highlighted
        """
        order = self.order_manager.get_current_order()
        missing = self.order_manager.get_missing_fields()

        status_str = "Current order:\n"
        status_str += f"Drink: {order['drinkType'] or 'Not selected'}\n"
        status_str += f"Size: {order['size'] or 'Not selected'}\n"
        status_str += f"Milk: {order['milk'] or 'Not selected'}\n"
        status_str += (
            f"Extras: {', '.join(order['extras']) if order['extras'] else 'None'}\n"
        )
        status_str += f"Name: {order['name'] or 'Not provided'}\n"
        if missing:
            status_str += f"\nStill need: {', '.join(missing)}"
        return status_str

    @function_tool
    async def complete_order(self, context: RunContext) -> str:
        """
        Complete and save the order.
        
        Use this ONLY when all fields are filled:
        - Drink type
        - Size
        - Milk option
        - Customer name
        
        This will save the order to a JSON file and notify the customer.
        
        Returns:
            Confirmation message with order details and file location
        """
        logger.info("Completing order")
        
        # Check if order is complete
        if not self.order_manager.is_order_complete():
            missing = self.order_manager.get_missing_fields()
            return (
                f"Cannot complete the order yet. Still need: {', '.join(missing)}"
            )

        try:
            # Save order to JSON file
            filename = self.order_manager.save_order_to_json()
            order = self.order_manager.get_current_order()

            # Create confirmation message
            confirmation = (
                f"Perfect! Order confirmed for {order['name']}. "
                f"One {order['size']} {order['drinkType']} with {order['milk']}. "
            )
            if order["extras"]:
                confirmation += f"With {', '.join(order['extras'])}. "
            confirmation += f"Your order will be ready in about 5 minutes! Thank you!"

            logger.info(f"Order saved: {filename}")

            # Reset for next customer
            self.order_manager.reset_order()

            return confirmation

        except ValueError as e:
            logger.error(f"Error completing order: {e}")
            return f"Error processing order: {str(e)}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """
    Main entry point for the LiveKit agent.
    
    This function:
    1. Sets up logging context
    2. Creates an OrderManager for handling coffee orders
    3. Configures the voice pipeline (STT, LLM, TTS, VAD)
    4. Initializes the Assistant with the order manager
    5. Starts the session and connects to the user
    """
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Create order manager instance for this session
    # This will be shared across all function tools in the Assistant
    order_manager = OrderManager()
    logger.info("OrderManager initialized for new session")

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    # Pass the order_manager to the Assistant so it can use the function tools
    await session.start(
        agent=Assistant(order_manager=order_manager),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
