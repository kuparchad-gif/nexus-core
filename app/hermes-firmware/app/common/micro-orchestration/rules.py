
import re, pathlib

CATEGORIES  =  [
  "UpperCognition/apps", "CognitionBridge/apps",
  "Reasoner-Conscious/apps", "Reasoner-Subconscious/apps",
  "SubconSwitch/apps", "CorpusCallosum/apps",
  "PerceptionEngine/apps", "CuriosityEngine/apps", "MuseEngine/apps", "DreamEngine/apps",
  "EgoGuard/apps",
  "protocols", "infra", "frontends/portal", "frontends/council", "common"
]

HINTS  =  [
    (re.compile(r"mind\.obs\.event"), "Reasoner-Conscious/apps"),
    (re.compile(r"cogbridge\.ask\.reason"), "Reasoner-Conscious/apps"),
    (re.compile(r"upper\.digest|upper\.plan"), "UpperCognition/apps"),
    (re.compile(r"cogbridge\.tell\.reasoned"), "CognitionBridge/apps"),
    (re.compile(r"memory\.write|conscious\.digest|upper\.digest"), "SubconSwitch/apps"),
    (re.compile(r"cog\.obs\.event"), "CorpusCallosum/apps"),
    (re.compile(r"interaction\.text"), "PerceptionEngine/apps"),
    (re.compile(r"cog\.how\.ask"), "CuriosityEngine/apps"),
    (re.compile(r"muse\.artifact"), "MuseEngine/apps"),
    (re.compile(r"cog\.sandbox\.intent|dream"), "DreamEngine/apps"),
    (re.compile(r"mind\.intent\.(approved|denied)|policy|deny"), "EgoGuard/apps"),
    (re.compile(r"compose\.ya?ml|nats.*\.conf|jetstream"), "infra"),
    (re.compile(r"seed|persona|affect|mind_map"), "protocols"),
    (re.compile(r"react|next|tsx|tailwind"), "frontends/portal"),
]

INFER_BY_EXT  =  {
    ".conf": "infra", ".ini": "infra", ".yaml": "protocols", ".yml": "protocols",
    ".md": "protocols", ".json": "protocols"
}

def score_by_rules(path: pathlib.Path, header: str) -> tuple[str, float, str]:
    p  =  str(path).replace("\\","/")
    ext  =  path.suffix.lower()
    if ext in INFER_BY_EXT:
        return INFER_BY_EXT[ext], 0.55, f"ext:{ext}"
    for rx, cat in HINTS:
        if rx.search(header) or rx.search(p):
            return cat, 0.8, f"hint:{rx.pattern}"
    if any(k in p for k in ["/common/", "utils", "shared"]):
        return "common", 0.6, "path:common"
    return ("protocols" if ext in (".md",".txt") else "common"), 0.4, "default"
