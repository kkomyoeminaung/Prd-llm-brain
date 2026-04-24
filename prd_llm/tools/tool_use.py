"""
Tool Use / Function Calling logic
"""
import json

class ToolRegistry:
    def __init__(self):
        self.tools = {
            "calculator": "Calculate math",
            "search": "Search web",
            "python": "Run code"
        }
    
    def execute(self, name, args):
        print(f"[Tool] Running {name} with {args}")
        return f"Executed {name} successfully."

class ToolUsingPRDLLM:
    def __init__(self, model, registry):
        self.model = model
        self.registry = registry
        
    def generate_with_tools(self, prompt):
        # Simulated tool call detection
        if "calculate" in prompt.lower():
            res = self.registry.execute("calculator", {"expr": prompt})
            return f"The calculation result is: {res}"
        return f"Simulated response to: {prompt}"
