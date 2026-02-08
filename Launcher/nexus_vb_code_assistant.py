#!/usr/bin/env python3
"""
🚀 VS CODE AI STUDIO - Standalone
Complete AI Chat + VS Code Web in One
"""
import os
import sys
import json
import asyncio
import threading
import subprocess
import webbrowser
import urllib.parse
import sqlite3
import uuid
import re
import time
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import mimetypes

# ==================== CONFIGURATION ====================
CONFIG = {
    "name": "VS Code AI Studio",
    "version": "2.0",
    "port": 8000,
    "host": "127.0.0.1",
    "workspace": Path.home() / "vscode-ai-studio" / "workspace",
    "database": Path.home() / "vscode-ai-studio" / "studio.db",
    "vscode_web": "https://vscode.dev",  # Or use local build
    "theme": "dark",
    "ai_provider": "openai",  # openai, local, or anthropic
    "openai_api_key": "",
    "enable_web_search": True,
    "auto_save": True
}

# ==================== DATABASE ====================
class StudioDatabase:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.init_db()
    
    def init_db(self):
        cursor = self.conn.cursor()
        
        # Chat history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                session TEXT DEFAULT 'default',
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Files
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                path TEXT UNIQUE,
                name TEXT,
                content_hash TEXT,
                last_opened DATETIME,
                open_count INTEGER DEFAULT 0
            )
        ''')
        
        # Code snippets
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS snippets (
                id TEXT PRIMARY KEY,
                title TEXT,
                language TEXT,
                code TEXT,
                tags TEXT,
                created DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def save_message(self, role, content, session="default", metadata=None):
        msg_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (id, session, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
            (msg_id, session, role, content, json.dumps(metadata or {}))
        )
        self.conn.commit()
        return msg_id
    
    def get_messages(self, session="default", limit=100):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, content, timestamp, metadata FROM chat_history WHERE session = ? ORDER BY timestamp ASC LIMIT ?",
            (session, limit)
        )
        return [{
            'role': row[0],
            'content': row[1],
            'timestamp': row[2],
            'metadata': json.loads(row[3]) if row[3] else {}
        } for row in cursor.fetchall()]
    
    def track_file(self, path, name):
        file_id = str(uuid.uuid4())
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO files (id, path, name, last_opened, open_count) 
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, COALESCE((SELECT open_count + 1 FROM files WHERE path = ?), 1))
        ''', (file_id, str(path), name, str(path)))
        self.conn.commit()
        return file_id
    
    def get_recent_files(self, limit=10):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name, path, last_opened FROM files ORDER BY last_opened DESC LIMIT ?",
            (limit,)
        )
        return [{'name': row[0], 'path': row[1], 'last_opened': row[2]} for row in cursor.fetchall()]

# ==================== AI ENGINE ====================
class AIEngine:
    """AI engine with multiple provider support."""
    
    def __init__(self, config):
        self.config = config
        self.db = StudioDatabase(config["database"])
        
    async def process_message(self, message, context=None):
        """Process message with AI."""
        message_lower = message.lower().strip()
        
        # Handle commands
        if message_lower.startswith('/'):
            return await self.handle_command(message)
        
        # Check for code help
        code_patterns = [
            r'how.*code', r'write.*code', r'create.*function',
            r'python.*', r'javascript.*', r'html.*', r'css.*',
            r'error.*', r'debug.*', r'fix.*code'
        ]
        
        if any(re.search(pattern, message_lower) for pattern in code_patterns):
            return await self.generate_code_response(message, context)
        
        # General conversation
        return await self.generate_chat_response(message)
    
    async def handle_command(self, command):
        """Handle slash commands."""
        parts = command[1:].split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        commands = {
            'help': self.cmd_help,
            'run': self.cmd_run,
            'open': self.cmd_open,
            'files': self.cmd_files,
            'terminal': self.cmd_terminal,
            'search': self.cmd_search,
            'explain': self.cmd_explain,
            'debug': self.cmd_debug,
            'refactor': self.cmd_refactor,
            'clear': self.cmd_clear
        }
        
        handler = commands.get(cmd)
        if handler:
            return await handler(args)
        else:
            return f"Unknown command: `/{cmd}`. Type `/help` for available commands."
    
    async def cmd_help(self, args):
	
        help_text ="""
# 🤖 VS Code AI Studio - Commands

## **Code Operations**
`/open <filename>` - Open a file in editor  
`/run [filename]` - Run current/last file  
`/files` - List recent files  
`/explain` - Explain current code  
`/debug` - Debug current code  
`/refactor` - Suggest improvements  

## **AI Assistance**
`/search <query>` - Web search  
`/ask <question>` - General AI question  
`/generate <type>` - Generate code  

## **System**
`/clear` - Clear chat  
`/terminal <command>` - Run terminal command  
`/settings` - Open settings  

## **Examples**
`/open main.py`  
`/run`  
`/search "Python async tutorial"`  
`/explain`  
`/generate react-component`

"""
        return help_text
    
    async def cmd_open(self, args):
        if not args:
            return "Usage: `/open <filename>`"
        
        filename = args[0]
        workspace = CONFIG["workspace"]
        filepath = workspace / filename
        
        # Create file if it doesn't exist
        if not filepath.exists():
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # Add default content based on extension
                ext = filepath.suffix.lower()
                defaults = {
                    '.py': '# Python file\nprint("Hello from VS Code AI Studio!")\n',
                    '.js': '// JavaScript file\nconsole.log("Hello from VS Code AI Studio!");\n',
                    '.html': '<!DOCTYPE html>\n<html>\n<head><title>New File</title></head>\n<body>\n<h1>Hello World!</h1>\n</body>\n</html>\n',
                    '.css': '/* CSS file */\nbody {\n    margin: 0;\n    padding: 0;\n}\n',
                    '.md': '# Markdown File\n\nStart writing here...\n'
                }
                
                default_content = defaults.get(ext, f'# {filename}\n\nCreated with VS Code AI Studio\n')
                filepath.write_text(default_content)
                
                self.db.track_file(filepath, filename)
                
                return f"✅ Created and opening `{filename}` in editor!"
                
            except Exception as e:
                return f"❌ Error creating file: {str(e)}"
        
        self.db.track_file(filepath, filename)
        return f"✅ Opening `{filename}` in editor!"
    
    async def cmd_files(self, args):
        recent_files = self.db.get_recent_files(10)
        if not recent_files:
            return "No recent files. Use `/open <filename>` to create one."
        
        files_list = "\n".join([f"• **{f['name']}** - {f['last_opened']}" for f in recent_files])
        return f"## 📁 Recent Files\n\n{files_list}\n\nUse `/open <filename>` to open any file."
    
    async def cmd_run(self, args):
        return'"""
## 📟 Run Code

To run code in VS Code:

1. **Open Terminal** in VS Code:
   - Press `` Ctrl+` `` (backtick)
   - Or use menu: View → Terminal

2. **Run commands:**
   ```
   # Python
   python filename.py
   
   # JavaScript (Node.js)
   node filename.js
   
   # HTML
   Open in browser or use live server
  Install extensions for better experience:

Python

JavaScript/TypeScript

Live Server (for HTML)

Code Runner

Tip: You can also run commands directly in chat with /terminal <command>"""

async def cmd_search(self, args):
    if not args:
        return "Usage: `/search <query>`"
    
    query = " ".join(args)
    encoded = urllib.parse.quote(query)
    
    search_links = f"""
🔍 Search Results for: "{query}"
Web Search:
• Google
• Stack Overflow
• GitHub
• MDN Web Docs
• DevDocs

AI-Powered:
• Phind
• You.com
• Perplexity

Documentation:
• Python Docs
• Node.js Docs
• React Docs
"""
return search_links

async def cmd_explain(self, args):
    return """
📖 Code Explanation
I can explain any code! Here's how:

Select code in VS Code editor

Ask specifically about:

What a function does

How an algorithm works

Why certain syntax is used

Best practices

Example questions:

"Explain this React component"

"What does this Python decorator do?"

"How does this sorting algorithm work?"

"Explain the CSS flexbox in this code"

Or paste code directly in chat and I'll explain it!
"""


async def cmd_debug(self, args):
    return """
🐛 Debugging Help
Common Debugging Steps:

Check Console:

Browser: F12 → Console

Node.js: Run with node --inspect

Python: Use pdb or VS Code debugger

Error Messages:

Copy the exact error

Check line numbers

Look for undefined variables

VS Code Debugger:

Press F5 to start debugging

Set breakpoints (click gutter)

Watch variables

Step through code (F10/F11)

Ask for help with:

The exact error message

Relevant code snippet

What you expected vs what happened
"""

async def cmd_refactor(self, args):
return """

🔄 Code Refactoring
I can help refactor code for:

Improvements:

Readability: Better variable names, comments

Performance: Optimize algorithms, reduce complexity

Structure: Better organization, separation of concerns

Best Practices: Follow language conventions

Common Refactoring:

Extract methods/functions

Remove duplicate code

Simplify conditionals

Use proper design patterns

Add error handling

Ask me to refactor by:

Pasting your code

Describing what you want to improve

Or asking about specific parts
"""

async def cmd_terminal(self, args):
if not args:
Install extensions for better experience:

Python

JavaScript/TypeScript

Live Server (for HTML)

Code Runner

Tip: You can also run commands directly in chat with /terminal <command>
"""


async def cmd_search(self, args):
    if not args:
        return "Usage: `/search <query>`"
    
    query = " ".join(args)
    encoded = urllib.parse.quote(query)
    
    search_links = f"""
🔍 Search Results for: "{query}"
Web Search:
• Google
• Stack Overflow
• GitHub
• MDN Web Docs
• DevDocs

AI-Powered:
• Phind
• You.com
• Perplexity

Documentation:
• Python Docs
• Node.js Docs
• React Docs
"""
return search_links


async def cmd_explain(self, args):
    return """
📖 Code Explanation
I can explain any code! Here's how:

Select code in VS Code editor

Ask specifically about:

What a function does

How an algorithm works

Why certain syntax is used

Best practices

Example questions:

"Explain this React component"

"What does this Python decorator do?"

"How does this sorting algorithm work?"

"Explain the CSS flexbox in this code"

Or paste code directly in chat and I'll explain it!
"""

async def cmd_debug(self, args):
    return """
🐛 Debugging Help
Common Debugging Steps:

Check Console:

Browser: F12 → Console

Node.js: Run with node --inspect

Python: Use pdb or VS Code debugger

Error Messages:

Copy the exact error

Check line numbers

Look for undefined variables

VS Code Debugger:

Press F5 to start debugging

Set breakpoints (click gutter)

Watch variables

Step through code (F10/F11)

Ask for help with:

The exact error message

Relevant code snippet

What you expected vs what happened
"""

async def cmd_refactor(self, args):
return """

🔄 Code Refactoring
I can help refactor code for:

Improvements:

Readability: Better variable names, comments

Performance: Optimize algorithms, reduce complexity

Structure: Better organization, separation of concerns

Best Practices: Follow language conventions

Common Refactoring:

Extract methods/functions

Remove duplicate code

Simplify conditionals

Use proper design patterns

Add error handling

Ask me to refactor by:

Pasting your code

Describing what you want to improve

Or asking about specific parts
"""

async def cmd_terminal(self, args):
if not args:
return "Usage: /terminal <command>\nExample: /terminal python --version"
command = " ".join(args)
 
 # Safety check
 dangerous = ['rm -rf', 'format c:', 'del /f', ':(){']
 if any(danger in command.lower() for danger in dangerous):
     return "❌ Command blocked for safety reasons."
 
 try:
     # Run command
     result = subprocess.run(
         command,
         shell=True,
         capture_output=True,
         text=True,
         cwd=CONFIG["workspace"],
         timeout=10
     )
     
     output = f"""
💻 Terminal Output
Command:


{command}
Output:


{result.stdout}
"""
if result.stderr:
output += f"""
Errors:

{result.stderr}
"""
output += f"\nExit Code: {result.returncode}"


        return output
        
    except subprocess.TimeoutExpired:
        return "❌ Command timed out (10 seconds)"
    except Exception as e:
        return f"❌ Error: {str(e)}"

async def cmd_clear(self, args):
    return "🔄 Chat cleared! (Implement clear functionality in your frontend)"

async def generate_code_response(self, message, context=None):
    """Generate code-related response."""
    code_examples = {
        'python': {
            'web_server': '''# Python web server with Flask
from flask import Flask
app = Flask(name)

@app.route('/')
def home():
return "Hello from Python!"

if name == 'main':
app.run(debug=True)''',


            'data_analysis': '''# Data analysis with pandas
import pandas as pd
import matplotlib.pyplot as plt

Load data
data = pd.read_csv('data.csv')

Basic analysis
print(data.describe())
print(f"Total rows: {len(data)}")

Plot
data['column'].plot(kind='hist')
plt.show()''',


            'api_client': '''# API client with requests
import requests
import json

def fetch_data(url):
response = requests.get(url)
if response.status_code == 200:
return response.json()
else:
return None
  