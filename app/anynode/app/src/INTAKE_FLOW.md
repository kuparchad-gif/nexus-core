# Smart Intake — “What is this and where does it go?”

1) Put your mixed “gold” folder somewhere (e.g., D:\GoodCode).
2) Dry‑run classification:
   ```powershell
   cd C:\Projects\Stacks\nexus-metatron
   python scripts\smart_intake.py --src "D:\GoodCode" --dst "C:\Projects\Stacks\nexus-metatron" --out intake_out
   ```
3) Review `intake_out/<timestamp>/intake_report.md` and `intake_map.json`.
4) Apply:
   ```powershell
   python scripts\smart_intake.py --src "D:\GoodCode" --dst "C:\Projects\Stacks\nexus-metatron" --apply
   ```
5) Need stricter mapping? Edit `rules\rules_nexus_v1.json` (or use `rules_template.json`/`.yaml`) and re‑run with:
   ```powershell
   python scripts\smart_intake.py --src "D:\GoodCode" --dst "C:\Projects\Stacks\nexus-metatron" --rules "rules\my_rules.json"
   ```
