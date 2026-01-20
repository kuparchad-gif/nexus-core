# plane_validator.py — Validate Nexus subject names
import re, sys

PATTERN = re.compile(
    r'^(svc|sys|cog|play)\.(sense|think|heal|archive)\.(cap\.register|cap\.request|cap\.reply|events|metrics)\.[A-Za-z0-9_\-\.>]+$'
)

def validate(subject: str) -> bool:
    return bool(PATTERN.match(subject))

if __name__ == "__main__":
    ok = True
    for s in sys.argv[1:]:
        print(s, "OK" if validate(s) else "INVALID")
        ok = ok and validate(s)
    sys.exit(0 if ok else 1)
