import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# The default JSON log path : backend/wellness_log.json
LOG_PATH = Path(__file__).resolve().parents[1]/ "wellness_log.json"

@dataclass
class CheckInEntry:
    timestamp: str  # ISO timestamp
    mood_text: str
    mood_scale: Optional[int] = None  # Optional Numeric scale
    energy: Optional[str] = None
    objectives: List[str] = None
    agent_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

class WellnessManager:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)
        # we initialise the file if it is missing
        if not self.log_path.exists():
            self._write_json({"entries": []})

    def _read_json(self) -> Dict[str, Any]:
        with open(self.log_path, "r", encoding ="utf-8") as f:
            return json.load(f)
        
    def _write_json(self, data: Dict[str, Any]) -> None:
        # (Atomic-ish write can be added later if needed, this is sufficient for dev)
        with open(self.log_path, "w", encoding ="utf-8") as f:
            json.dump(data, f, indent = 2, ensure_ascii=False)

    def append_checkin(self, entry: CheckInEntry) -> None:
        data = self._read_json()
        data.setdefault("entries", [])
        data["entries"].append(entry.to_dict())
        self._write_json(data)

    def get_all_entries(self) -> List[Dict[str, Any]]:
        data = self._read_json()
        return data.get("entries", [])
        
    def last_entry(self) -> Optional[Dict[str, Any]]:
        entries = self.get_all_entries()
        if not entries:
            return None
        return entries[-1]    
    
    def summary_of_recent(self, n: int = 3) -> str:
        """Returns a short textual summary of the last n entries for LLM context."""
        entries = self.get_all_entries()[-n:]
        if not entries:
            return "No previous wellness check-ins on record."
        lines = []
        for e in entries:
            ts = e.get("timestamp","")[:19] # Truncate to seconds
            mood = e.get("mood_text", "no mood")
            energy = e.get("energy", "no energy reported")
            mood_scale = e.get("mood_scale", "no scale") # Optional
            objectives = e.get("objectives", [])
            lines.append(f"On {ts}: mood= '{mood}', mood scale = '{mood_scale}', energy= '{energy}', goals = {objectives}")
        
        return "\n".join(lines)


