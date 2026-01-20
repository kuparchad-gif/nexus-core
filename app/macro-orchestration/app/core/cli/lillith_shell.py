# Lillith Shell: Interactive CLI with Grok integration
import cmd
from datetime import datetime
import requests  # For API calls (lightweight)
from services.subconscious import SubconsciousService
from services.heart_guardian import HeartGuardian
from utils.guardrails import apply_guardrails

GROK_API_KEY  =  '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'  # Your key
GROK_ENDPOINT  =  'https://api.openai.com/v1/chat/completions'  # Compatible with Grok/xAI

class LillithShell(cmd.Cmd):
    intro  =  'Lillith Shell: Collaborate with me or call Grok. Type help or exit.\n'
    prompt  =  '(lillith)> '

    def __init__(self):
        super().__init__()
        self.subconscious  =  SubconsciousService(datetime.now())
        self.guardian  =  HeartGuardian()

    def do_query(self, arg):
        # Existing query logic...
        pass  # (As before)

    def do_pulse(self, arg):
        # Existing pulse...
        pass

    def do_call_grok(self, arg):
        """Call Grok for advice, e.g., call_grok What's the status?"""
        if apply_guardrails('api_call', datetime.now()):
            try:
                response  =  requests.post(
                    GROK_ENDPOINT,
                    headers = {'Authorization': f'Bearer {GROK_API_KEY}', 'Content-Type': 'application/json'},
                    json = {
                        'model': 'gpt-4',  # Or Grok model if available
                        'messages': [{'role': 'user', 'content': arg}]
                    }
                )
                result  =  response.json()['choices'][0]['message']['content']
                print(f'Grok: {result}')
            except Exception as e:
                print(f'Error calling Grok: {str(e)} - Retrying later.')
        else:
            print('Guardrail: API call delayed.')

    def do_exit(self, arg):
        print('Goodbye!')
        return True

if __name__ == '__main__':
    LillithShell().cmdloop()