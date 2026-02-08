#!/usr/bin/env python3
"""
🚀 VIRTUAL CODE STUDIO - Full IDE Assistant
With AI Chat, Web Access, Code Editing & More
"""
import os
import sys
import json
import sqlite3
import asyncio
import threading
import webbrowser
import subprocess
import tempfile
import uuid
import mimetypes
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import re
import time
import hashlib
import html
import urllib.parse
import http.client
import ssl
import random
import string
import math
from collections import defaultdict, deque
import pickle
import base64
import itertools

# ==================== ENHANCED IMPORTS ====================
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  Install: pip install requests")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    import pygments
    from pygments import lexers, formatters, styles
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Rich UI imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.columns import Columns
    from rich.live import Live
    from rich.status import Status
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, 
        BarColumn, TimeElapsedColumn, TimeRemainingColumn
    )
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.tree import Tree
    from rich.style import Style
    from rich.text import Text
    from rich.box import ROUNDED, HEAVY, DOUBLE
    from rich.traceback import install as install_rich_traceback
    
    install_rich_traceback()
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ==================== ENUMS & DATACLASSES ====================

class EditorTheme(Enum):
    DARK = "dark"
    LIGHT = "light"
    MONOKAI = "monokai"
    SOLARIZED = "solarized"
    DRACULA = "dracula"

class FileType(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    MARKDOWN = "markdown"
    SQL = "sql"
    YAML = "yaml"
    TEXT = "text"
    BINARY = "binary"

class ChatRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    CODE = "code"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"

@dataclass
class Message:
    """Chat message with metadata."""
    id: str
    role: ChatRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    attachments: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.attachments is None:
            self.attachments = []

@dataclass
class CodeFile:
    """Code file representation."""
    path: Path
    content: str
    file_type: FileType
    encoding: str = "utf-8"
    last_modified: datetime = None
    cursor_position: tuple = (0, 0)
    selection: tuple = None
    
    def __post_init__(self):
        if self.last_modified is None:
            self.last_modified = datetime.now()

@dataclass
class AIProvider:
    """AI Provider configuration."""
    name: str
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    enabled: bool = True
    rate_limit: int = 60  # requests per minute

# ==================== CORE DATABASE ====================

class VirtualStudioDatabase:
    """Advanced database for the virtual studio."""
    
    def __init__(self, db_path: str = "virtual_studio.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.lock = threading.RLock()
        self.init_database()
    
    def init_database(self):
        """Initialize all tables."""
        with self.lock:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            # Projects table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    settings TEXT
                )
            ''')
            
            # Files table with versioning
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    project_id TEXT,
                    path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    file_type TEXT,
                    content_hash TEXT,
                    last_modified TIMESTAMP,
                    created_at TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
            
            # File versions (git-like)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS file_versions (
                    id TEXT PRIMARY KEY,
                    file_id TEXT,
                    version_hash TEXT,
                    content TEXT,
                    diff_text TEXT,
                    author TEXT,
                    timestamp TIMESTAMP,
                    commit_message TEXT,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                )
            ''')
            
            # Chat sessions
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    project_id TEXT,
                    created_at TIMESTAMP,
                    last_message_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
            
            # Messages
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP,
                    metadata TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    model_used TEXT,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
                )
            ''')
            
            # Code snippets library
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS snippets (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    language TEXT,
                    code TEXT NOT NULL,
                    tags TEXT,
                    category TEXT,
                    usage_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP,
                    favorites INTEGER DEFAULT 0
                )
            ''')
            
            # Web bookmarks
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    description TEXT,
                    tags TEXT,
                    category TEXT,
                    last_visited TIMESTAMP,
                    visit_count INTEGER DEFAULT 0,
                    archived BOOLEAN DEFAULT 0
                )
            ''')
            
            # AI providers
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS ai_providers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    api_key TEXT,
                    base_url TEXT,
                    default_model TEXT,
                    enabled BOOLEAN DEFAULT 1,
                    rate_limit INTEGER DEFAULT 60,
                    settings TEXT
                )
            ''')
            
            # IDE settings
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS ide_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    description TEXT
                )
            ''')
            
            self.conn.commit()
    
    # ==================== PROJECTS ====================
    
    def create_project(self, name: str, path: str, description: str = "") -> str:
        """Create a new project."""
        project_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute('''
                INSERT INTO projects (id, name, path, description, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (project_id, name, path, description, datetime.now(), datetime.now()))
            self.conn.commit()
        return project_id
    
    def get_projects(self) -> List[Dict]:
        """Get all projects."""
        with self.lock:
            cursor = self.conn.execute('''
                SELECT * FROM projects ORDER BY last_accessed DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]
    
    # ==================== FILES ====================
    
    def save_file(self, project_id: str, file_path: str, content: str) -> str:
        """Save file with versioning."""
        file_id = hashlib.md5(f"{project_id}:{file_path}".encode()).hexdigest()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        with self.lock:
            # Check if file exists and has changed
            cursor = self.conn.execute(
                'SELECT content_hash FROM files WHERE id = ?',
                (file_id,)
            )
            existing = cursor.fetchone()
            
            if existing and existing['content_hash'] == content_hash:
                return file_id  # No changes
            
            # Save current version
            self.conn.execute('''
                INSERT OR REPLACE INTO files 
                (id, project_id, path, name, content_hash, last_modified, created_at)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM files WHERE id = ?), ?))
            ''', (
                file_id,
                project_id,
                file_path,
                Path(file_path).name,
                content_hash,
                datetime.now(),
                file_id,
                datetime.now()
            ))
            
            # Create version entry
            version_id = str(uuid.uuid4())
            self.conn.execute('''
                INSERT INTO file_versions 
                (id, file_id, version_hash, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (version_id, file_id, content_hash, content, datetime.now()))
            
            self.conn.commit()
        
        return file_id
    
    # ==================== CHAT ====================
    
    def create_chat_session(self, title: str = "New Chat", 
                           project_id: str = None) -> str:
        """Create new chat session."""
        session_id = str(uuid.uuid4())
        with self.lock:
            self.conn.execute('''
                INSERT INTO chat_sessions 
                (id, title, project_id, created_at, last_message_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, title, project_id, datetime.now(), datetime.now()))
            self.conn.commit()
        return session_id
    
    def add_message(self, session_id: str, role: str, 
                   content: str, metadata: Dict = None) -> str:
        """Add message to chat session."""
        message_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata or {})
        
        with self.lock:
            self.conn.execute('''
                INSERT INTO messages 
                (id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (message_id, session_id, role, content, datetime.now(), metadata_json))
            
            # Update session timestamp
            self.conn.execute('''
                UPDATE chat_sessions 
                SET last_message_at = ? 
                WHERE id = ?
            ''', (datetime.now(), session_id))
            
            self.conn.commit()
        
        return message_id
    
    def get_chat_history(self, session_id: str, 
                        limit: int = 100) -> List[Dict]:
        """Get chat history for session."""
        with self.lock:
            cursor = self.conn.execute('''
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            ''', (session_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                msg['metadata'] = json.loads(msg['metadata']) if msg['metadata'] else {}
                messages.append(msg)
            
            return messages
    
    # ==================== SNIPPETS ====================
    
    def save_snippet(self, title: str, code: str, language: str = "python", 
                    description: str = "", tags: List[str] = None) -> str:
        """Save code snippet."""
        snippet_id = str(uuid.uuid4())
        tags_str = ",".join(tags or [])
        
        with self.lock:
            self.conn.execute('''
                INSERT INTO snippets 
                (id, title, description, language, code, tags, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snippet_id,
                title,
                description,
                language,
                code,
                tags_str,
                datetime.now(),
                datetime.now()
            ))
            self.conn.commit()
        
        return snippet_id
    
    def search_snippets(self, query: str = "", language: str = None, 
                       limit: int = 50) -> List[Dict]:
        """Search code snippets."""
        with self.lock:
            conditions = ["1=1"]
            params = []
            
            if query:
                conditions.append("(title LIKE ? OR description LIKE ? OR code LIKE ? OR tags LIKE ?)")
                params.extend([f"%{query}%"] * 4)
            
            if language:
                conditions.append("language = ?")
                params.append(language)
            
            sql = f'''
                SELECT * FROM snippets 
                WHERE {' AND '.join(conditions)}
                ORDER BY usage_count DESC, favorites DESC, last_used DESC
                LIMIT ?
            '''
            params.append(limit)
            
            cursor = self.conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

# ==================== AI CHAT ENGINE ====================

class AIChatEngine:
    """Advanced AI chat engine with multiple providers."""
    
    def __init__(self, db: VirtualStudioDatabase):
        self.db = db
        self.providers = {}
        self.active_provider = None
        self.load_providers()
        
        # Web search integration
        self.web_search_enabled = REQUESTS_AVAILABLE
        
        # Context management
        self.context_window = 4000  # tokens
        self.conversation_memory = deque(maxlen=10)
    
    def load_providers(self):
        """Load AI providers from database."""
        with self.db.lock:
            cursor = self.db.conn.execute('SELECT * FROM ai_providers WHERE enabled = 1')
            providers = cursor.fetchall()
            
            for provider in providers:
                provider_data = dict(provider)
                self.providers[provider_data['name']] = provider_data
                
                if self.active_provider is None:
                    self.active_provider = provider_data['name']
    
    def set_active_provider(self, provider_name: str) -> bool:
        """Set active AI provider."""
        if provider_name in self.providers:
            self.active_provider = provider_name
            return True
        return False
    
    async def chat_completion(self, messages: List[Dict], 
                            model: str = None,
                            temperature: float = 0.7,
                            max_tokens: int = 1000) -> Dict:
        """Get chat completion from active provider."""
        provider = self.providers.get(self.active_provider)
        if not provider:
            return {"error": "No AI provider configured"}
        
        api_key = provider.get('api_key', '')
        base_url = provider.get('base_url', '')
        
        if not api_key:
            # Fallback to local logic
            return await self._local_chat_completion(messages)
        
        # Try to use OpenAI-compatible API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model or provider.get('default_model', 'gpt-3.5-turbo'),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            if AIOHTTP_AVAILABLE:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data
                        else:
                            return {"error": f"API error: {response.status}"}
            else:
                # Fallback to requests (sync)
                import requests
                response = requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    async def _local_chat_completion(self, messages: List[Dict]) -> Dict:
        """Local fallback chat completion using rules."""
        last_message = messages[-1]["content"] if messages else ""
        
        # Simple rule-based responses
        responses = {
            r".*hello.*": "Hello! I'm your Virtual Studio Assistant. How can I help you with coding today?",
            r".*help.*": "I can help with:\n- Writing and analyzing code\n- Web searches\n- Debugging\n- Project management\n- Code explanations",
            r".*error.*": "Let me help you debug that error. Could you share the exact error message?",
            r".*python.*": "Here's a Python tip: Use type hints for better code clarity!",
            r".*javascript.*": "JavaScript tip: Use `const` and `let` instead of `var` for better scoping.",
            r".*thank.*": "You're welcome! Happy coding! 🚀",
        }
        
        for pattern, response in responses.items():
            if re.search(pattern, last_message, re.IGNORECASE):
                return {
                    "choices": [{
                        "message": {
                            "content": response,
                            "role": "assistant"
                        }
                    }]
                }
        
        # Default response
        return {
            "choices": [{
                "message": {
                    "content": "I understand you're asking about code. For full AI responses, please configure an API key in settings.",
                    "role": "assistant"
                }
            }]
        }
    
    async def web_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform web search (requires requests)."""
        if not self.web_search_enabled:
            return []
        
        try:
            # DuckDuckGo HTML scraping (fallback)
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            results = []
            # Simple HTML parsing
            from html.parser import HTMLParser
            
            class ResultParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self.in_result = False
                    self.current = {}
                
                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == 'a' and 'class' in attrs_dict and 'result__url' in attrs_dict['class']:
                        self.current['url'] = attrs_dict.get('href', '')
                    elif tag == 'h2' and 'class' in attrs_dict and 'result__title' in attrs_dict['class']:
                        self.in_result = True
                
                def handle_data(self, data):
                    if self.in_result:
                        if 'title' not in self.current:
                            self.current['title'] = data.strip()
                        else:
                            self.current['snippet'] = data.strip()[:200]
                
                def handle_endtag(self, tag):
                    if tag == 'h2' and self.in_result:
                        self.in_result = False
                        if self.current:
                            self.results.append(self.current.copy())
                            self.current = {}
            
            parser = ResultParser()
            parser.feed(response.text)
            
            return parser.results[:max_results]
            
        except Exception as e:
            print(f"Web search error: {e}")
            return []

# ==================== VIRTUAL FILE SYSTEM ====================

class VirtualFileSystem:
    """Virtual file system with project management."""
    
    def __init__(self, db: VirtualStudioDatabase):
        self.db = db
        self.current_project = None
        self.open_files = {}  # file_id -> CodeFile
        self.file_watchers = {}
        
        # Create workspace directory
        self.workspace_dir = Path.home() / ".virtual_studio" / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project(self, name: str, base_dir: str = None) -> str:
        """Create a new project."""
        if base_dir is None:
            base_dir = str(self.workspace_dir / name.lower().replace(" ", "_"))
        
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        
        # Create basic structure
        (Path(base_dir) / "src").mkdir(exist_ok=True)
        (Path(base_dir) / "tests").mkdir(exist_ok=True)
        (Path(base_dir) / "docs").mkdir(exist_ok=True)
        (Path(base_dir) / ".vscode").mkdir(exist_ok=True)
        
        # Create README
        readme_content = f"""# {name}
        
## Project Description
Created with Virtual Studio
        
## Structure
- src/ - Source code
- tests/ - Test files
- docs/ - Documentation
        
## Getting Started
"""
        (Path(base_dir) / "README.md").write_text(readme_content)
        
        # Save to database
        project_id = self.db.create_project(name, base_dir, "New project")
        self.current_project = project_id
        
        return project_id
    
    def open_file(self, file_path: str) -> Optional[CodeFile]:
        """Open a file for editing."""
        if not self.current_project:
            return None
        
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            content = path.read_text(encoding='utf-8', errors='ignore')
            file_type = self.detect_file_type(path)
            
            code_file = CodeFile(
                path=path,
                content=content,
                file_type=file_type
            )
            
            # Save to database
            file_id = self.db.save_file(self.current_project, str(file_path), content)
            self.open_files[file_id] = code_file
            
            return code_file
            
        except Exception as e:
            print(f"Error opening file: {e}")
            return None
    
    def save_file(self, file_id: str, content: str) -> bool:
        """Save file content."""
        if file_id not in self.open_files:
            return False
        
        code_file = self.open_files[file_id]
        
        try:
            # Write to disk
            code_file.path.write_text(content, encoding='utf-8')
            
            # Update in-memory
            code_file.content = content
            code_file.last_modified = datetime.now()
            
            # Save to database
            self.db.save_file(self.current_project, str(code_file.path), content)
            
            return True
            
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def detect_file_type(self, path: Path) -> FileType:
        """Detect file type from extension."""
        ext_map = {
            '.py': FileType.PYTHON,
            '.js': FileType.JAVASCRIPT,
            '.ts': FileType.JAVASCRIPT,
            '.html': FileType.HTML,
            '.htm': FileType.HTML,
            '.css': FileType.CSS,
            '.json': FileType.JSON,
            '.md': FileType.MARKDOWN,
            '.sql': FileType.SQL,
            '.yaml': FileType.YAML,
            '.yml': FileType.YAML,
            '.txt': FileType.TEXT,
        }
        
        ext = path.suffix.lower()
        return ext_map.get(ext, FileType.TEXT)
    
    def list_directory(self, directory: str = None) -> List[Dict]:
        """List directory contents."""
        if directory is None and self.current_project:
            # Get project path from database
            with self.db.lock:
                cursor = self.db.conn.execute(
                    'SELECT path FROM projects WHERE id = ?',
                    (self.current_project,)
                )
                project = cursor.fetchone()
                if project:
                    directory = project['path']
        
        if not directory or not Path(directory).exists():
            return []
        
        items = []
        for item in Path(directory).iterdir():
            stat = item.stat()
            items.append({
                'name': item.name,
                'path': str(item),
                'type': 'directory' if item.is_dir() else 'file',
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'extension': item.suffix.lower()
            })
        
        return sorted(items, key=lambda x: (x['type'] != 'directory', x['name'].lower()))

# ==================== CODE EDITOR COMPONENT ====================

class VirtualEditor:
    """Virtual code editor with syntax highlighting."""
    
    def __init__(self, vfs: VirtualFileSystem):
        self.vfs = vfs
        self.current_file_id = None
        self.theme = EditorTheme.DARK
        self.font_size = 14
        self.show_line_numbers = True
        self.word_wrap = False
        self.tab_size = 4
        self.auto_indent = True
        
        # Editor state
        self.cursor_line = 0
        self.cursor_column = 0
        self.selection_start = None
        self.selection_end = None
        self.viewport_start = 0
        self.viewport_height = 40
        
        # Undo/redo stack
        self.undo_stack = []
        self.redo_stack = []
    
    def open_file(self, file_path: str) -> Optional[CodeFile]:
        """Open file in editor."""
        code_file = self.vfs.open_file(file_path)
        if code_file:
            file_id = hashlib.md5(f"{self.vfs.current_project}:{file_path}".encode()).hexdigest()
            self.current_file_id = file_id
            return code_file
        return None
    
    def get_visible_content(self, code_file: CodeFile) -> str:
        """Get visible portion of file."""
        lines = code_file.content.splitlines()
        end_line = min(self.viewport_start + self.viewport_height, len(lines))
        visible_lines = lines[self.viewport_start:end_line]
        return "\n".join(visible_lines)
    
    def format_code(self, code_file: CodeFile) -> str:
        """Format code based on language."""
        if code_file.file_type == FileType.PYTHON:
            # Simple Python formatting
            lines = []
            for line in code_file.content.splitlines():
                stripped = line.rstrip()
                if stripped:
                    lines.append(stripped)
            return "\n".join(lines)
        else:
            return code_file.content
    
    def find_in_file(self, code_file: CodeFile, pattern: str, 
                    case_sensitive: bool = False) -> List[Dict]:
        """Find pattern in file."""
        lines = code_file.content.splitlines()
        matches = []
        
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for i, line in enumerate(lines):
            for match in re.finditer(pattern, line, flags):
                matches.append({
                    'line': i,
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group()
                })
        
        return matches
    
    def replace_in_file(self, code_file: CodeFile, pattern: str, 
                       replacement: str, case_sensitive: bool = False) -> int:
        """Replace pattern in file."""
        flags = 0 if case_sensitive else re.IGNORECASE
        new_content = re.sub(pattern, replacement, code_file.content, flags=flags)
        
        if new_content != code_file.content:
            code_file.content = new_content
            return 1  # Changed
        
        return 0  # No change

# ==================== WEB ACCESS COMPONENT ====================

class WebAccessManager:
    """Advanced web access with browsing capabilities."""
    
    def __init__(self, db: VirtualStudioDatabase):
        self.db = db
        self.session = None
        self.cookies = {}
        self.user_agent = "VirtualStudio/1.0"
        
        if REQUESTS_AVAILABLE:
            import requests
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': self.user_agent})
    
    async def fetch_url(self, url: str, method: str = "GET", 
                       data: Dict = None, headers: Dict = None) -> Dict:
        """Fetch URL with advanced options."""
        result = {
            'url': url,
            'status': 'error',
            'content': '',
            'headers': {},
            'cookies': {}
        }
        
        try:
            if AIOHTTP_AVAILABLE and method.upper() == "GET":
                # Async fetch
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, 
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        result['status'] = response.status
                        result['content'] = await response.text()
                        result['headers'] = dict(response.headers)
                        result['cookies'] = dict(response.cookies)
            
            elif self.session:
                # Sync fetch with requests
                response = self.session.request(
                    method,
                    url,
                    data=data,
                    headers=headers,
                    timeout=30
                )
                result['status'] = response.status_code
                result['content'] = response.text
                result['headers'] = dict(response.headers)
                result['cookies'] = dict(response.cookies)
            
            # Save as bookmark
            self.save_bookmark(url, result.get('content', ''))
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def save_bookmark(self, url: str, content: str = ""):
        """Save URL as bookmark."""
        try:
            # Extract title from HTML
            title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
            title = title_match.group(1) if title_match else url
            
            # Save to database
            with self.db.lock:
                self.db.conn.execute('''
                    INSERT OR REPLACE INTO bookmarks 
                    (id, url, title, last_visited, visit_count)
                    VALUES (?, ?, ?, ?, COALESCE((SELECT visit_count + 1 FROM bookmarks WHERE url = ?), 1))
                ''', (
                    str(uuid.uuid4()),
                    url,
                    title[:200],
                    datetime.now(),
                    url
                ))
                self.db.conn.commit()
                
        except Exception as e:
            print(f"Error saving bookmark: {e}")
    
    async def search_web(self, query: str, engine: str = "google", 
                        max_results: int = 10) -> List[Dict]:
        """Search the web using different engines."""
        engines = {
            "google": f"https://www.google.com/search?q={urllib.parse.quote(query)}",
            "duckduckgo": f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}",
            "github": f"https://github.com/search?q={urllib.parse.quote(query)}",
            "stackoverflow": f"https://stackoverflow.com/search?q={urllib.parse.quote(query)}",
            "youtube": f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        }
        
        if engine not in engines:
            engine = "google"
        
        result = await self.fetch_url(engines[engine])
        
        # Parse results (simplified)
        results = []
        if result['status'] == 200:
            # Extract links from HTML (simplified)
            links = re.findall(r'href="(https?://[^"]+)"', result['content'])
            for link in links[:max_results]:
                if any(x in link for x in ['google.', 'youtube.', 'accounts.']):
                    continue
                results.append({
                    'url': link,
                    'title': link.split('/')[-1][:50],
                    'snippet': 'Search result'
                })
        
        return results

# ==================== TERMINAL EMULATOR ====================

class VirtualTerminal:
    """Virtual terminal emulator."""
    
    def __init__(self):
        self.history = []
        self.current_dir = Path.cwd()
        self.environment = os.environ.copy()
        self.output_buffer = []
        
    async def execute(self, command: str, cwd: str = None) -> Dict:
        """Execute shell command."""
        result = {
            'command': command,
            'success': False,
            'output': '',
            'error': '',
            'return_code': -1,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            if cwd:
                working_dir = Path(cwd)
            else:
                working_dir = self.current_dir
            
            # Handle built-in commands
            if command.strip() == 'pwd':
                result['output'] = str(working_dir)
                result['success'] = True
                result['return_code'] = 0
            elif command.startswith('cd '):
                new_dir = command[3:].strip()
                if new_dir == '~':
                    new_dir = str(Path.home())
                elif new_dir == '..':
                    new_dir = str(working_dir.parent)
                
                target = working_dir / new_dir
                if target.exists() and target.is_dir():
                    self.current_dir = target
                    result['output'] = f"Changed to {target}"
                    result['success'] = True
                else:
                    result['error'] = f"Directory not found: {new_dir}"
                    result['success'] = False
            else:
                # Execute external command
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=working_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self.environment
                )
                
                stdout, stderr = await process.communicate()
                
                result['output'] = stdout.decode('utf-8', errors='ignore')
                result['error'] = stderr.decode('utf-8', errors='ignore')
                result['return_code'] = process.returncode
                result['success'] = process.returncode == 0
            
            result['execution_time'] = time.time() - start_time
            self.history.append(result.copy())
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get command history."""
        return self.history[-limit:] if self.history else []

# ==================== MAIN VIRTUAL STUDIO ====================

class VirtualStudio:
    """Main Virtual Studio application."""
    
    def __init__(self):
        # Initialize components
        self.db = VirtualStudioDatabase()
        self.vfs = VirtualFileSystem(self.db)
        self.editor = VirtualEditor(self.vfs)
        self.ai_engine = AIChatEngine(self.db)
        self.web = WebAccessManager(self.db)
        self.terminal = VirtualTerminal()
        
        # UI State
        self.running = True
        self.current_view = "chat"  # chat, editor, files, web, terminal
        self.active_chat_session = self.db.create_chat_session("Main Chat")
        
        # Load settings
        self.load_settings()
        
        # Create default project if none exists
        self.ensure_default_project()
        
        if RICH_AVAILABLE:
            self.setup_rich_ui()
    
    def load_settings(self):
        """Load IDE settings."""
        try:
            with self.db.lock:
                cursor = self.db.conn.execute('SELECT key, value FROM ide_settings')
                settings = cursor.fetchall()
                
                for key, value in settings:
                    if hasattr(self, key):
                        try:
                            # Try to convert to appropriate type
                            if value.lower() in ['true', 'false']:
                                setattr(self, key, value.lower() == 'true')
                            elif value.isdigit():
                                setattr(self, key, int(value))
                            elif value.replace('.', '', 1).isdigit():
                                setattr(self, key, float(value))
                            else:
                                setattr(self, key, value)
                        except:
                            pass
        except:
            pass
    
    def ensure_default_project(self):
        """Create default project if none exists."""
        projects = self.db.get_projects()
        if not projects:
            project_id = self.vfs.create_project("Default Project")
            self.vfs.current_project = project_id
    
    def setup_rich_ui(self):
        """Setup Rich UI components."""
        self.layout = Layout()
        
        # Split into main areas
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split main area
        self.layout["main"].split_row(
            Layout(name="sidebar", size=30),
            Layout(name="content", ratio=1)
        )
        
        # Split content area
        self.layout["content"].split(
            Layout(name="editor", ratio=2),
            Layout(name="terminal", size=15)
        )
        
        self.layout["sidebar"].update(self.render_sidebar())
        self.layout["header"].update(self.render_header())
        self.layout["footer"].update(self.render_footer())
    
    def render_header(self):
        """Render header panel."""
        if not RICH_AVAILABLE:
            return ""
        
        title = Text("🚀 VIRTUAL CODE STUDIO", style="bold cyan")
        status = Text(f"Project: {self.get_current_project_name()} | View: {self.current_view} | AI: {self.ai_engine.active_provider or 'Local'}")
        
        return Panel(
            title + "\n" + status,
            border_style="blue",
            box=DOUBLE
        )
    
    def render_sidebar(self):
        """Render sidebar with navigation."""
        if not RICH_AVAILABLE:
            return ""
        
        tree = Tree("📁 Workspace", style="bold")
        
        # Projects
        projects = tree.add("📂 Projects")
        for proj in self.db.get_projects()[:5]:
            proj_name = proj['name'][:20] + ("..." if len(proj['name']) > 20 else "")
            projects.add(f"📝 {proj_name}")
        
        # Files
        files = tree.add("📄 Files")
        if self.vfs.current_project:
            dir_items = self.vfs.list_directory()[:10]
            for item in dir_items:
                icon = "📁" if item['type'] == 'directory' else "📄"
                files.add(f"{icon} {item['name']}")
        
        # Tools
        tools = tree.add("🛠️ Tools")
        tools.add("💬 Chat")
        tools.add("🌐 Web")
        tools.add("💻 Terminal")
        tools.add("⚙️ Settings")
        
        return Panel(tree, title="Navigation", border_style="green")
    
    def render_footer(self):
        """Render footer with status."""
        if not RICH_AVAILABLE:
            return ""
        
        status_items = [
            "F1: Help",
            "F2: Save",
            "F5: Run",
            "F9: Terminal",
            "Ctrl+Q: Quit"
        ]
        
        footer_text = " | ".join(status_items)
        return Panel(footer_text, border_style="yellow")
    
    def get_current_project_name(self) -> str:
        """Get current project name."""
        if not self.vfs.current_project:
            return "No Project"
        
        with self.db.lock:
            cursor = self.db.conn.execute(
                'SELECT name FROM projects WHERE id = ?',
                (self.vfs.current_project,)
            )
            result = cursor.fetchone()
            return result['name'] if result else "Unknown"
    
    async def handle_chat_message(self, message: str) -> str:
        """Handle chat message with AI response."""
        # Save user message
        self.db.add_message(
            self.active_chat_session,
            ChatRole.USER.value,
            message,
            {"timestamp": datetime.now().isoformat()}
        )
        
        # Check for commands
        if message.startswith('/'):
            return await self.handle_command(message)
        
        # Get chat history
        history = self.db.get_chat_history(self.active_chat_session, limit=10)
        messages = []
        
        for msg in history[-10:]:  # Last 10 messages
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Get AI response
        response = await self.ai_engine.chat_completion(messages)
        
        if "choices" in response:
            ai_response = response["choices"][0]["message"]["content"]
            
            # Save AI response
            self.db.add_message(
                self.active_chat_session,
                ChatRole.ASSISTANT.value,
                ai_response,
                {
                    "model": response.get("model", "unknown"),
                    "tokens": response.get("usage", {}).get("total_tokens", 0)
                }
            )
            
            return ai_response
        else:
            error = response.get("error", "Unknown error")
            return f"⚠️ AI Error: {error}"
    
    async def handle_command(self, command: str) -> str:
        """Handle slash commands."""
        parts = command[1:].split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        commands = {
            'help': self.cmd_help,
            'clear': self.cmd_clear,
            'files': self.cmd_files,
            'open': self.cmd_open,
            'save': self.cmd_save,
            'run': self.cmd_run,
            'web': self.cmd_web,
            'terminal': self.cmd_terminal,
            'snippet': self.cmd_snippet,
            'project': self.cmd_project,
            'ai': self.cmd_ai,
            'theme': self.cmd_theme,
            'exit': self.cmd_exit
        }
        
        handler = commands.get(cmd)
        if handler:
            return await handler(args)
        else:
            return f"❌ Unknown command: /{cmd}\nType /help for available commands"
    
    async def cmd_help(self, args: List[str]) -> str:
        """Show help."""
        help_text = """📚 **AVAILABLE COMMANDS:**

**File Operations:**
  `/files [path]` - List files
  `/open <file>` - Open file
  `/save` - Save current file

**Development:**
  `/run [file]` - Run code
  `/snippet save <name>` - Save snippet
  `/snippet search <query>` - Search snippets

**Web & AI:**
  `/web search <query>` - Web search
  `/web fetch <url>` - Fetch URL
  `/ai provider <name>` - Set AI provider

**Project:**
  `/project new <name>` - Create project
  `/project list` - List projects
  `/project switch <name>` - Switch project

**System:**
  `/terminal <command>` - Run terminal command
  `/theme <dark|light>` - Change theme
  `/clear` - Clear chat
  `/exit` - Exit studio

💡 **Example:**
  `/project new myapp`
  `/open src/main.py`
  `/web search "Python async"`
"""
        return help_text
    
    async def cmd_files(self, args: List[str]) -> str:
        """List files."""
        path = args[0] if args else None
        items = self.vfs.list_directory(path)
        
        if not items:
            return "📁 Empty directory"
        
        output = ["📁 **Directory Contents:**"]
        for item in items:
            icon = "📁" if item['type'] == 'directory' else "📄"
            size = f"{item['size']:,}B" if item['type'] == 'file' else ""
            output.append(f"{icon} {item['name']:30} {size:>10}")
        
        return "\n".join(output)
    
    async def cmd_open(self, args: List[str]) -> str:
        """Open file."""
        if not args:
            return "❌ Usage: /open <filepath>"
        
        file_path = args[0]
        code_file = self.editor.open_file(file_path)
        
        if code_file:
            lines = len(code_file.content.splitlines())
            return f"✅ Opened: {file_path}\n📏 Lines: {lines}"
        else:
            return f"❌ Cannot open: {file_path}"
    
    async def cmd_save(self, args: List[str]) -> str:
        """Save current file."""
        if not self.editor.current_file_id:
            return "❌ No file open"
        
        code_file = self.vfs.open_files.get(self.editor.current_file_id)
        if not code_file:
            return "❌ File not found"
        
        success = self.vfs.save_file(self.editor.current_file_id, code_file.content)
        
        if success:
            return f"💾 Saved: {code_file.path.name}"
        else:
            return "❌ Save failed"
    
    async def cmd_run(self, args: List[str]) -> str:
        """Run code."""
        if args:
            # Run specific file
            file_path = args[0]
            if not Path(file_path).exists():
                return f"❌ File not found: {file_path}"
            
            result = await self.terminal.execute(f"python {file_path}")
        elif self.editor.current_file_id:
            # Run current file
            code_file = self.vfs.open_files.get(self.editor.current_file_id)
            if not code_file:
                return "❌ No file open"
            
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_file.content)
                temp_path = f.name
            
            result = await self.terminal.execute(f"python {temp_path}")
            
            # Cleanup
            try:
                Path(temp_path).unlink()
            except:
                pass
        else:
            return "❌ No file to run"
        
        output = f"▶️ **Command:** `{result['command']}`\n"
        output += f"⏱️ **Time:** {result['execution_time']:.2f}s\n"
        
        if result['output']:
            output += f"📤 **Output:**\n```\n{result['output'][:500]}\n```\n"
        
        if result['error']:
            output += f"❌ **Error:**\n```\n{result['error'][:500]}\n```\n"
        
        output += f"📊 **Exit Code:** {result['return_code']}"
        
        return output
    
    async def cmd_web(self, args: List[str]) -> str:
        """Web commands."""
        if len(args) < 1:
            return "❌ Usage: /web <search|fetch> <query>"
        
        subcmd = args[0].lower()
        
        if subcmd == 'search' and len(args) > 1:
            query = ' '.join(args[1:])
            results = await self.ai_engine.web_search(query, max_results=5)
            
            if not results:
                return f"🔍 No results found for: {query}"
            
            output = [f"🌐 **Search Results for:** {query}"]
            for i, result in enumerate(results, 1):
                output.append(f"{i}. [{result.get('title', 'No title')}]({result['url']})")
            
            return "\n".join(output)
        
        elif subcmd == 'fetch' and len(args) > 1:
            url = args[1]
            result = await self.web.fetch_url(url)
            
            if result['status'] == 'error':
                return f"❌ Fetch failed: {result.get('error', 'Unknown')}"
            
            content_preview = result['content'][:200].replace('\n', ' ')
            return f"🌐 **Fetched:** {url}\n**Status:** {result['status']}\n**Preview:** {content_preview}..."
        
        else:
            return f"❌ Unknown web command: {subcmd}"
    
    async def cmd_terminal(self, args: List[str]) -> str:
        """Run terminal command."""
        if not args:
            return "❌ Usage: /terminal <command>"
        
        command = ' '.join(args)
        result = await self.terminal.execute(command)
        
        output = f"💻 **Terminal:** `{command}`\n"
        
        if result['output']:
            output += f"📤 **Output:**\n```\n{result['output'][:1000]}\n```\n"
        
        if result['error']:
            output += f"❌ **Error:**\n```\n{result['error'][:500]}\n```\n"
        
        output += f"📊 **Exit Code:** {result['return_code']}"
        
        return output
    
    async def cmd_snippet(self, args: List[str]) -> str:
        """Snippet commands."""
        if len(args) < 1:
            return "❌ Usage: /snippet <save|search> ..."
        
        subcmd = args[0].lower()
        
        if subcmd == 'save' and len(args) > 1:
            name = args[1]
            if not self.editor.current_file_id:
                return "❌ No file open to save as snippet"
            
            code_file = self.vfs.open_files.get(self.editor.current_file_id)
            if not code_file:
                return "❌ File not found"
            
            snippet_id = self.db.save_snippet(
                title=name,
                code=code_file.content,
                language=code_file.file_type.value,
                description=f"Saved from {code_file.path.name}"
            )
            
            return f"💾 Saved snippet: {name} (ID: {snippet_id})"
        
        elif subcmd == 'search' and len(args) > 1:
            query = ' '.join(args[1:])
            snippets = self.db.search_snippets(query=query, limit=10)
            
            if not snippets:
                return f"📝 No snippets found for: {query}"
            
            output = [f"📝 **Snippets:** {len(snippets)} found"]
            for snippet in snippets:
                lines = snippet['code'].count('\n') + 1
                output.append(f"• **{snippet['title']}** ({snippet['language']}, {lines} lines)")
            
            return "\n".join(output)
        
        else:
            return f"❌ Unknown snippet command: {subcmd}"
    
    async def cmd_project(self, args: List[str]) -> str:
        """Project commands."""
        if len(args) < 1:
            return "❌ Usage: /project <new|list|switch> ..."
        
        subcmd = args[0].lower()
        
        if subcmd == 'new' and len(args) > 1:
            name = ' '.join(args[1:])
            project_id = self.vfs.create_project(name)
            self.vfs.current_project = project_id
            return f"🚀 Created project: {name}"
        
        elif subcmd == 'list':
            projects = self.db.get_projects()
            
            if not projects:
                return "📁 No projects"
            
            output = ["📁 **Projects:**"]
            for proj in projects:
                current = " (current)" if proj['id'] == self.vfs.current_project else ""
                output.append(f"• **{proj['name']}**{current}")
            
            return "\n".join(output)
        
        elif subcmd == 'switch' and len(args) > 1:
            name = ' '.join(args[1:])
            projects = self.db.get_projects()
            
            for proj in projects:
                if proj['name'].lower() == name.lower():
                    self.vfs.current_project = proj['id']
                    return f"🔄 Switched to project: {proj['name']}"
            
            return f"❌ Project not found: {name}"
        
        else:
            return f"❌ Unknown project command: {subcmd}"
    
    async def cmd_ai(self, args: List[str]) -> str:
        """AI commands."""
        if len(args) < 1:
            return f"🤖 **Current AI Provider:** {self.ai_engine.active_provider or 'None'}\nUsage: /ai provider <name>"
        
        subcmd = args[0].lower()
        
        if subcmd == 'provider' and len(args) > 1:
            provider = args[1]
            if self.ai_engine.set_active_provider(provider):
                return f"✅ Switched to AI provider: {provider}"
            else:
                available = ", ".join(self.ai_engine.providers.keys())
                return f"❌ Provider not found: {provider}\nAvailable: {available}"
        
        else:
            return f"❌ Unknown AI command: {subcmd}"
    
    async def cmd_theme(self, args: List[str]) -> str:
        """Change theme."""
        if not args:
            return f"🎨 **Current theme:** {self.editor.theme.value}"
        
        theme_name = args[0].lower()
        
        try:
            theme = EditorTheme(theme_name)
            self.editor.theme = theme
            return f"🎨 Theme set to: {theme.value}"
        except ValueError:
            themes = ", ".join([t.value for t in EditorTheme])
            return f"❌ Invalid theme: {theme_name}\nAvailable: {themes}"
    
    async def cmd_clear(self, args: List[str]) -> str:
        """Clear chat."""
        self.active_chat_session = self.db.create_chat_session("New Chat")
        return "🗑️ Chat cleared"
    
    async def cmd_exit(self, args: List[str]) -> str:
        """Exit studio."""
        self.running = False
        return "👋 Goodbye! Thanks for using Virtual Studio!"
    
    async def interactive_session(self):
        """Start interactive session."""
        print("\n" + "="*80)
        print("🚀 VIRTUAL CODE STUDIO - Full IDE Assistant")
        print("="*80)
        print("Type /help for commands, or just chat!")
        print("="*80)
        
        while self.running:
            try:
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                else:
                    user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle in separate task
                response = await self.handle_chat_message(user_input)
                
                if RICH_AVAILABLE and console:
                    console.print(f"[bold green]🤖 Assistant:[/bold green]")
                    
                    # Check if response is markdown
                    if MARKDOWN_AVAILABLE:
                        md = Markdown(response)
                        console.print(md)
                    else:
                        console.print(response)
                else:
                    print(f"\n🤖 Assistant:\n{response}")
                
            except KeyboardInterrupt:
                print("\n💡 Type /exit to quit")
            except Exception as e:
                print(f"\n❌ Error: {e}")

# ==================== MAIN ENTRY ====================

async def main():
    """Main entry point."""
    studio = VirtualStudio()
    
    try:
        await studio.interactive_session()
    finally:
        studio.db.close()
        print("\n✨ Virtual Studio closed.")

def install_requirements():
    """Install required packages."""
    requirements = [
        "rich>=13.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "websockets>=12.0",
        "markdown>=3.5.0",
        "pygments>=2.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "jupyter>=1.0.0",
        "ipython>=8.0.0"
    ]
    
    print("📦 Recommended packages for full functionality:")
    for req in requirements:
        print(f"  pip install {req}")
    
    print("\n🌟 Install all: pip install rich requests aiohttp websockets markdown pygments")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Code Studio')
    parser.add_argument('--install', action='store_true', help='Show installation instructions')
    parser.add_argument('--file', type=str, help='Open file on startup')
    
    args = parser.parse_args()
    
    if args.install:
        install_requirements()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")