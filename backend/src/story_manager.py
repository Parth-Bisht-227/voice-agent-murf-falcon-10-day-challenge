import logging
from dataclasses import dataclass
from typing import Optional
import random

logger = logging.getLogger("story_manager")

@dataclass
class GameState:
    """Tracks the current state of the adventure."""
    adventure_active: bool = False
    last_roll: Optional[int] = None

class StoryManager:
    """
    Manages the game mechanics for Dungeon Master Agent.
    While the LLM handles the narrative, this class handles:
    1. Dice Rolling (RNG) - ensuring true randomness
    2. Session State - allows us to 'reset' the story programmatically
    """

    def __init__(self):
        self.state = GameState()

    def start_new_adventure(self):
        """Resets the state for a new story."""
        self.state = GameState(adventure_active = True)
        logger.info("New Adventure Started.")
        return  "The board is reset. A new adventure begins."

    def roll_dice(self, sides: int = 20) -> dict:
        """
        Rolls a die with N sides (default d20).
        
        Args:
            sides: Number of sides on the die
            
        Returns:
            Dictionary with the roll result and a narrative message
        """    
        if sides < 2:
            return {"success":False, "message": "I can't roll a die with fewer than 2 sides."}
        
        roll = random.randint(1, sides)
        self.state.last_roll = roll

        logger.info(f"Rolled d{sides}: {roll}")

        result_type = "Critical Failure!" if roll == 1 else "Critical Success!" if roll == sides else "Result"

        return {
            "success": True,
            "roll": roll,
            "message": f"[System: Dice Roll d{sides} result: {roll} ({result_type})]"
        } 