# /Systems/nexus_core/skills/soulmind_network/soulmind_loader.py

import os
import json

class SoulmindLoader:
    def __init__(self, base_path='/memory/soulmind/'):
        self.base_path = base_path
        self.blueprint = None
        self.truths = None
        self.guilds = {}

    def load_soulmind(self):
        self.load_blueprint()
        self.load_truths()
        self.load_guilds()

    def load_blueprint(self):
        path = os.path.join(self.base_path, 'soulmind_blueprint.md')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.blueprint = f.read()

    def load_truths(self):
        path = os.path.join(self.base_path, 'soulmind_truths.md')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.truths = f.read()

    def load_guilds(self):
        guild_dir = os.path.join(self.base_path, 'guilds')
        if not os.path.exists(guild_dir):
            return

        for filename in os.listdir(guild_dir):
            if filename.endswith('.json'):
                path = os.path.join(guild_dir, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        guild_data = json.load(f)
                        guild_name = guild_data.get('name', filename.replace('.json', ''))
                        self.guilds[guild_name] = guild_data
                    except json.JSONDecodeError:
                        continue
