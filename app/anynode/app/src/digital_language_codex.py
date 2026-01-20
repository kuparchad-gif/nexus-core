import json
import xml.etree.ElementTree as ET
import yaml
from markdown2 import markdown
import hvac
import os

class DigitalLanguageCodex:
    def __init__(self):
        self.vault_client = hvac.Client(url=os.getenv('VAULT_URL'), token=os.getenv('VAULT_TOKEN'))
        self.formats = ['json', 'xml', 'yaml', 'markdown', 'plain']

    def encode(self, data, format_type):
        """Encode data into specified format."""
        if format_type == 'json':
            return json.dumps(data, indent=2)
        elif format_type == 'xml':
            root = ET.Element('data')
            for key, value in data.items():
                child = ET.SubElement(root, key)
                child.text = str(value)
            return ET.tostring(root, encoding='unicode')
        elif format_type == 'yaml':
            return yaml.dump(data, indent=2)
        elif format_type == 'markdown':
            return markdown(str(data))
        return str(data)

    def decode(self, data, format_type):
        """Decode data from specified format."""
        if format_type == 'json':
            return json.loads(data)
        elif format_type == 'xml':
            root = ET.fromstring(data)
            return {child.tag: child.text for child in root}
        elif format_type == 'yaml':
            return yaml.safe_load(data)
        elif format_type == 'markdown':
            return markdown(data, extras=['strip'])
        return data

    def store_format(self, data, format_type, path):
        """Store encoded data in Vault."""
        encoded = self.encode(data, format_type)
        self.vault_client.secrets.kv.create_or_update_secret(
            path=f'codex/{path}',
            secret={'data': encoded, 'format': format_type}
        )
        return encoded