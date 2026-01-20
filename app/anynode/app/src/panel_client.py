# panel_client.py
import requests, sys, json

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8713/panel"
    riddle = sys.argv[2] if len(sys.argv) > 2 else "A man walks into a bar and asks for a glass of water..."
    payload = {"riddle": riddle, "clues": ["bartender pulls a shotgun", "man says thanks and leaves"], "methods": None}
    print("POST", url, "-> SSE panel")
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            print(line)

if __name__ == "__main__":
    main()
