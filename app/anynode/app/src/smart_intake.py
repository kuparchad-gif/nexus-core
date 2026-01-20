
    #!/usr/bin/env python3
    import argparse, fnmatch, os, re, sys, json, pathlib, hashlib, shutil, datetime

    CODE_FENCE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+.\-]*)\s*(?P<body>.*?)```", re.DOTALL)

    DEFAULT_INCLUDE = ["*.py","*.ts","*.tsx","*.js","*.jsx","*.json","*.yml","*.yaml","*.md","*.txt","*.html","*.css","Dockerfile","Containerfile*","*.ps1","*.sh","*.conf"]
    DEFAULT_EXCLUDE = [".git",".venv","node_modules","dist","build",".qdrant",".pytest_cache",".cache","logs","coverage","tmp","pack","__pycache__",".DS_Store","Thumbs.db"]

    EXT_BY_LANG = {
        "python":"py","py":"py","powershell":"ps1","shell":"sh","bash":"sh",
        "typescript":"ts","ts":"ts","tsx":"tsx","javascript":"js","js":"js",
        "json":"json","yaml":"yaml","yml":"yml","html":"html","css":"css","nginx":"conf","dockerfile":"Dockerfile"
    }

    def read_rules(path):
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)

    def is_text_file(p: pathlib.Path) -> bool:
        try:
            with open(p,"rb") as f:
                chunk = f.read(4096)
            return b"\x00" not in chunk
        except Exception:
            return False

    def sniff_head(p: pathlib.Path, n=2000) -> str:
        try:
            with open(p,"r",encoding="utf-8",errors="ignore") as f:
                return f.read(n).lower()
        except Exception:
            return ""

    def detect_ext_from_lang(lang: str) -> str:
        mapping = {
            "python":"py","py":"py","powershell":"ps1","shell":"sh","bash":"sh",
            "typescript":"ts","ts":"ts","tsx":"tsx","javascript":"js","js":"js",
            "json":"json","yaml":"yaml","yml":"yml","html":"html","css":"css","nginx":"conf","dockerfile":"Dockerfile"
        }
        return mapping.get((lang or "").lower().strip(), "txt")

    def matches_any(s: str, pats):
        if not pats: return False
        s = s.lower()
        for pat in pats:
            if pat.lower() in s: return True
        return False

    def classify(path: pathlib.Path, head: str, rules: dict) -> str:
        rels = str(path).lower()
        for d in rules["destinations"]:
            pats = d.get("patterns", {})
            if matches_any(rels, pats.get("path_any")) or matches_any(head, pats.get("content_any")):
                exts = pats.get("ext_any")
                if exts:
                    if path.suffix.lower().lstrip(".") in [e.lower().lstrip(".") for e in exts] or \
                       (path.name.lower() in [e.lower() for e in exts]):
                        return d["dest"]
                else:
                    return d["dest"]
        return rules.get("fallback_dest")

    def extract_code_blocks(raw_text: str, base_name: str, out_dir: pathlib.Path):
        blocks = []
        for i, m in enumerate(CODE_FENCE.finditer(raw_text), start=1):
            lang = m.group("lang") or ""
            body = m.group("body")
            ext = detect_ext_from_lang(lang)
            fname = f"{base_name}.block{i}.{ext}"
            outp = out_dir/fname
            outp.write_text(body, encoding="utf-8")
            blocks.append((fname, ext))
        return blocks

    def main():
        ap = argparse.ArgumentParser(description="Smart Intake classifier")
        ap.add_argument("--src", required=True, help="resource directory with mixed code and babble")
        ap.add_argument("--dst", required=True, help="repo root to propose/apply destinations")
        ap.add_argument("--rules", default="rules/rules_nexus_v1.json")
        ap.add_argument("--include", action="append")
        ap.add_argument("--exclude", action="append")
        ap.add_argument("--apply", action="store_true", help="actually copy files into --dst")
        ap.add_argument("--mirror", action="store_true", help="delete in dst when not present in src (careful)")
        ap.add_argument("--out", default="smart_intake_out")
        args = ap.parse_args()

        src = pathlib.Path(args.src).resolve()
        dst = pathlib.Path(args.dst).resolve()
        rules_path = pathlib.Path(args.rules)
        if not rules_path.is_absolute():
            rules_path = pathlib.Path(__file__).resolve().parent.parent / rules_path
        rules = read_rules(str(rules_path))

        include = args.include or DEFAULT_INCLUDE
        exclude = args.exclude or DEFAULT_EXCLUDE

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = pathlib.Path(args.out).resolve()/ts
        extracted_dir = out_root/"extracted_blocks"
        out_root.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        intake_map = []
        total_files = 0

        for root, dirs, files in os.walk(src):
            rootp = pathlib.Path(root)
            dirs[:] = [d for d in dirs if d not in exclude and not any(frag in (rootp/d).parts for frag in exclude)]
            for f in files:
                sp = rootp/f
                if not is_text_file(sp):
                    continue
                if not any(fnmatch.fnmatch(sp.name, pat) or fnmatch.fnmatch(str(sp), pat) for pat in include):
                    continue

                try:
                    head = sniff_head(sp)
                except Exception:
                    head = ""

                dest = classify(sp, head, rules)
                rec = {"source": str(sp), "dest": str(dst/dest), "reason": f"classified by rules → {dest}"}
                intake_map.append(rec)
                total_files += 1

                if sp.suffix.lower() in [".md",".txt"]:
                    try:
                        raw = sp.read_text(encoding="utf-8", errors="ignore")
                        blocks = extract_code_blocks(raw, sp.stem, extracted_dir)
                        for fname, ext in blocks:
                            bp = extracted_dir/fname
                            bhead = sniff_head(bp)
                            bdest = classify(bp, bhead, rules)
                            intake_map.append({"source": str(bp), "dest": str(dst/bdest),
                                               "reason": f"extracted code block ({ext}) from {sp.name} → {bdest}"})
                    except Exception:
                        pass

        map_path = out_root/"intake_map.json"
        rep_path = out_root/"intake_report.md"
        json.dump(intake_map, open(map_path,"w",encoding="utf-8"), indent=2)

        with open(rep_path,"w",encoding="utf-8") as rep:
            rep.write(f"# Smart Intake Report ({ts})\\n\\n")
            rep.write(f"- Source: {src}\\n- Dest root: {dst}\\n- Files scanned: {total_files}\\n\\n")
            for item in intake_map:
                rep.write(f"- **{item['source']}** → `{item['dest']}`  \\n  _{item['reason']}_\\n")

        if args.apply:
            for item in intake_map:
                sp = pathlib.Path(item["source"])
                dp = pathlib.Path(item["dest"])
                dp.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if not dp.exists() or sp.stat().st_mtime >= dp.stat().st_mtime:
                        shutil.copy2(sp, dp)
                except Exception:
                    pass

            if args.mirror:
                keep = {str(pathlib.Path(i["dest"]).resolve()) for i in intake_map}
                for root, dirs, files in os.walk(dst):
                    for f in files:
                        p = str(pathlib.Path(root)/f)
                        if p.startswith(str(dst)) and p not in keep:
                            try:
                                os.remove(p)
                            except Exception:
                                pass

            commit_ps1 = out_root/"apply_commit.ps1"
            with open(commit_ps1,"w",encoding="utf-8") as f:
                f.write("""param([string]$Repo = ".")
Push-Location $Repo
git add -A
git commit -m "chore(intake): apply smart intake mapping"
Pop-Location
""")

        print(str(out_root))
    if __name__ == "__main__":
        main()

