#!/usr/bin/env python3
"""
AI Backend for VS Code Extension
"""
import sys
import json
import asyncio
import os
from pathlib import Path

class AIBackend:
    def __init__(self):
        self.running = True
    
    async def run(self):
        """Main loop for processing requests from extension."""
        print("🤖 AI Backend started", flush=True)
        
        while self.running:
            try:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    continue
                
                # Parse request
                try:
                    request = json.loads(line.strip())
                    response = await self.process_request(request)
                    
                    # Send response
                    print(json.dumps(response), flush=True)
                    
                except json.JSONDecodeError:
                    print(json.dumps({"error": "Invalid JSON"}), flush=True)
                    
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)
    
    async def process_request(self, request):
        """Process AI request."""
        request_type = request.get("type", "chat")
        message = request.get("message", "")
        context = request.get("context", "")
        
        if request_type == "chat":
            return await self.chat(message, context)
        elif request_type == "explain":
            return await self.explain(context)
        elif request_type == "debug":
            return await self.debug(context)
        elif request_type == "generate":
            return await self.generate(message)
        else:
            return {"error": f"Unknown request type: {request_type}"}
    
    async def chat(self, message, context):
        """Chat with AI."""
        # Simple rule-based responses
        responses = {
            "hello": "Hello! I'm your VS Code AI assistant. How can I help?",
            "explain": f"I can explain that code. Context length: {len(context)} chars",
            "debug": "Let me help debug that. What's the error?",
            "default": f"I understand: {message}. Code context: {len(context)} chars"
        }
        
        msg_lower = message.lower()
        for key, response in responses.items():
            if key in msg_lower:
                return {"response": response}
        
        return {"response": responses["default"]}
    
    async def explain(self, code):
        """Explain code."""
        lines = code.count('\n') + 1
        return {
            "response": f"Code analysis:\n- Lines: {lines}\n- Characters: {len(code)}\n- Language detected: {self.detect_language(code)}\n\nI can explain this code in detail. What specific part would you like explained?"
        }
    
    async def debug(self, code):
        """Debug code."""
        # Simple debug suggestions
        suggestions = []
        
        if 'print(' in code or 'console.log' in code:
            suggestions.append("Add more print/console.log statements to trace execution")
        
        if 'try:' not in code and 'catch' not in code:
            suggestions.append("Consider adding try-catch/exception handling")
        
        if len(code.split('\n')) > 50:
            suggestions.append("Consider breaking code into smaller functions")
        
        suggestion_text = "\n".join([f"- {s}" for s in suggestions]) if suggestions else "- No specific issues detected"
        
        return {
            "response": f"Debug suggestions:\n{suggestion_text}\n\nFor specific errors, share the error message."
        }
    
    async def generate(self, prompt):
        """Generate code."""
        languages = {
            "python": "Python",
            "javascript": "JavaScript",
            "react": "React",
            "html": "HTML",
            "css": "CSS"
        }
        
        detected_lang = None
        for lang in languages:
            if lang in prompt.lower():
                detected_lang = languages[lang]
                break
        
        lang = detected_lang or "Python"
        
        return {
            "response": f"I'll generate {lang} code for: {prompt}\n\nHere's a template to get started..."
        }
    
    def detect_language(self, code):
        """Simple language detection."""
        if 'def ' in code or 'import ' in code:
            return "Python"
        elif 'function ' in code or 'const ' in code or 'let ' in code:
            return "JavaScript"
        elif '<html' in code or '<div' in code:
            return "HTML"
        elif '{' in code and ':' in code and ';' in code:
            return "CSS"
        else:
            return "Unknown"

async def main():
    backend = AIBackend()
    await backend.run()

if __name__ == "__main__":
    asyncio.run(main())