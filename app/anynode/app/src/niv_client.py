# niv_client.py
import requests, sys

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "[REDACTED-URL]
    payload = {
        "template_type": "logical_reasoning",
        "input": {
            "context": "Demo",
            "premises": ["A -> B", "A"],
            "question": "Is B likely true?"
        },
        "template_path": "/mnt/data/nexus_thought_templates/json/logical_reasoning.json"
    }
    print("POST", url, "-> SSE")
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.strip() == "":
                continue
            print(line)

if __name__ == "__main__":
    main()

