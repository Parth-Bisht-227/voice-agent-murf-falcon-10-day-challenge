# src/lead_manager.py
import json
import logging
import os

logger = logging.getLogger("lead_manager")

class LeadManager:
    def __init__(self):
        # Added 'team_size' as requested
        self.current_lead = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None, 
            "timeline": None
        }

    def update_lead(self, **kwargs):
        """Updates lead data and returns instructions for the Agent."""
        updated_fields = []
        
        for key, value in kwargs.items():
            if key in self.current_lead and value is not None:
                self.current_lead[key] = value
                updated_fields.append(key)
        
        # Immediate JSON dump (As per requirement: "Store as user responds")
        self._append_to_json(final=False)

        missing = self.get_missing_fields()
        
        if not missing:
            return "SUCCESS: All fields collected. You can now move to the summary phase."
        
        return f"Saved. Missing fields: {', '.join(missing)}. Please ask for ONE of these next."

    def get_missing_fields(self):
        return [k for k, v in self.current_lead.items() if v is None]

    def generate_summary(self):
        """Creates a natural language summary of the collected data."""
        data = self.current_lead
        # Fill None with 'Unknown' for the summary to read smoothly
        d = {k: (v if v else "unknown") for k, v in data.items()}
        
        summary = (f"Lead: {d['name']} from {d['company']} ({d['role']}). "
                   f"Needs Gan.ai for {d['use_case']} with a team of {d['team_size']}. "
                   f"Timeline: {d['timeline']}.")
        return summary

    def _append_to_json(self, final=False):
        """Helper to write to disk"""
        filename = "leads_db.json"
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        # If it's a final save, we append. 
        # For intermediate saves, we might just log or overwrite a temp file. 
        # Here we will just append for simplicity of the demo.
        if final:
            data.append(self.current_lead)
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
        return filename

    def finalize(self):
        """Final save and return summary string."""
        fname = self._append_to_json(final=True)
        return self.generate_summary()