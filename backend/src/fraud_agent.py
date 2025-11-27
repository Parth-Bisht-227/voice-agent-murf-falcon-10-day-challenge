import logging
import sqlite3
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from livekit.agents import (
    Agent, 
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("fraud_agent")
logger.setLevel(logging.INFO)

# --- DATABASE MANAGER ---

class FraudManager:
    """
    Handles all interactions with the SQLite database.
    """
    def __init__(self, db_name="bank_fraud.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        """Creates the table and seeds mock data if empty."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fraud_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT COLLATE NOCASE,
                security_identifier TEXT,
                card_ending TEXT,
                merchant TEXT,
                amount TEXT,
                timestamp TEXT,
                location TEXT,
                security_question TEXT,
                security_answer TEXT,
                status TEXT,
                notes TEXT
            )
        ''')

        # Check if db is empty, and seed mock data
        cursor.execute('SELECT COUNT(*) FROM fraud_cases')
        if cursor.fetchone()[0] == 0:
            logger.info("Seeding db with mock data...")
            mock_data = (
                "Samuel", "8812", "2424", "Amazon", "$1,250.00",
                "2024-08-01 14:23:55", "New York, USA",
                "What is your pet's name?", "Shiro",
                "pending_review", ""
            )
            # Fixed column names in INSERT statement
            cursor.execute('''
                INSERT INTO fraud_cases (
                    username, security_identifier, card_ending, merchant,
                    amount, timestamp, location, security_question,
                    security_answer, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', mock_data)
            conn.commit()
        conn.close()

    def get_case_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieves a fraud case by username."""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Fixed SQL typo: SELEC -> SELECT
        cursor.execute("SELECT * FROM fraud_cases WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def update_case_status(self, case_id: int, status: str, notes: str) -> str:
        """Updates the status of a case, after confirming."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE fraud_cases SET status = ?, notes = ? WHERE id = ?",
            (status, notes, case_id)
        )
        conn.commit()
        conn.close()
        return f"Case {case_id} updated to {status}."


# --- THE AGENT ---

class FraudAssistant(Agent):
    def __init__(self, fraud_manager: FraudManager) -> None:
        self.fraud_manager = fraud_manager     
        super().__init__(
            instructions="""You are a Fraud Security Specialist at Apex Bank. 
Your goal is to verify a suspicious transaction with the customer securely.

YOUR KNOWLEDGE BASE:
- You have access to a database of fraud cases. 
- You do NOT know who the user is initially. You must ask.

THE FLOW:
1. **Identify**: Ask the user for their First Name.
2. **Fetch**: Use `get_case_details` to retrieve their file.
   - If no file is found, apologize and end the call.
3. **Verify**: The file contains a `security_question` and `security_answer`. 
   - Ask the user the security question.
   - Compare their spoken answer to the `security_answer` in the file.
   - If they match (fuzzy match is okay), proceed. If they fail twice, end the call.
4. **Review**: Read the transaction details (Merchant, Amount, Location) clearly.
5. **Decide**: Ask if they authorized this transaction.
   - If YES: Use `update_case_outcome` to mark as 'safe'.
   - If NO: Use `update_case_outcome` to mark as 'fraudulent' and assure them the card is blocked.
6. **Close**: Thank them and end the call.

TONE: Professional, Calm, Secure, Efficient.
            """,
        )

    @function_tool
    async def get_case_details(self, context: RunContext, username: str) -> str:
        """
        Look up the customer's fraud file by their name.
        """
        logger.info(f"Looking up case for: {username}")
        # Fixed method call to match the corrected class method name
        case = self.fraud_manager.get_case_by_username(username)

        if not case:
            return "No active fraud case found for this user."
        
        return json.dumps(case)
    
    @function_tool
    async def update_case_outcome(self, context: RunContext, case_id: int, status: str, notes: str) -> str:
        """
        Finalize the call by updating the database.
        """
        logger.info(f"Updating case {case_id} to {status} with notes: {notes}")
        result = self.fraud_manager.update_case_status(case_id, status, notes)
        return result


# --- MAIN SETUP ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    fraud_manager = FraudManager()
    logger.info("FraudManager initialized.")

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
        agent=FraudAssistant(fraud_manager=fraud_manager),
        room=ctx.room,
    )

    await ctx.connect()
    
    # agent = session.agent
    # if agent:
    #     pass 

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))