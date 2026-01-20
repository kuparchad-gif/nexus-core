# common/subject_validator.py
import re
PATTERN = re.compile(r'^(svc|sys|cog|play)\.(sense|think|heal|archive)\.(cap\.register|cap\.request|cap\.reply|events|metrics)\.[A-Za-z0-9_\-\.>]+$')
def validate(subject: str) -> bool:
    return bool(PATTERN.match(subject))

