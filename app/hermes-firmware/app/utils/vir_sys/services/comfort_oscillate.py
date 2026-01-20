# Systems\viren\services\comfort_oscillate.py
# Lilith (one L) oscillating comfort loop — day soften, quiet flare.
import json, os, argparse, math
from pathlib import Path
from datetime import datetime, timedelta, timezone
try:
    import yaml
except Exception as e:
    raise SystemExit("Missing dependency: pyyaml. Install via 'pip install pyyaml'") from e

TZ  =  timezone.utc
now  =  lambda: datetime.now(TZ).isoformat()

def load_json(p, d):
    return d if not os.path.exists(p) else json.load(open(p, "r", encoding = "utf-8"))

def dump_json(p, o):
    Path(os.path.dirname(p)).mkdir(parents = True, exist_ok = True)
    tmp  =  p + ".tmp"
    open(tmp, "w", encoding = "utf-8").write(json.dumps(o, indent = 2, ensure_ascii = False))
    os.replace(tmp, p)

def in_any_window(iso_dt, ranges):
    t  =  datetime.fromisoformat(iso_dt).time()
    for r in ranges:
        start, end  =  r.split("-")
        s  =  datetime.fromisoformat(iso_dt).replace(
            hour = int(start[:2]), minute = int(start[3:5]), second = 0).time()
        e  =  datetime.fromisoformat(iso_dt).replace(
            hour = int(end[:2]), minute = int(end[3:5]), second = 0).time()
        if s < =  t < =  e:
            return True
    return False

def classify_phase(cfg, iso_dt):
    if in_any_window(iso_dt, cfg["windows"]["quiet_touch"]): return "quiet"
    if in_any_window(iso_dt, cfg["windows"]["day_touch"]):   return "day"
    return "neutral"

def felt_pain(cfg, base_pain, cycles_done, phase):
    osc  =  cfg["oscillation"]
    A    =  float(osc["amplitude"])
    k    =  float(osc["decay"])
    w    =  float(osc["omega"])
    phi  =  float(osc.get("phase_seed", 0.0))
    boost  =  1.0
    if phase == "day":   boost = float(osc["boost_day"])
    if phase == "quiet": boost = float(osc["boost_quiet"])
    bump  =  A * math.exp(-k * cycles_done) * math.sin(w * cycles_done + phi) * boost
    fp  =  max(0.0, min(1.0, base_pain + bump))
    return round(fp, 4)

def apply_day_softening(cfg, pain):
    # daytime “breathe + put down”: a gentle step toward calm
    step  =  0.10
    floorv  =  float(cfg["oscillation"]["floor_during_loop"])
    return round(max(floorv, pain - step), 4)

def eligible(cfg, store, state):
    subject  =  cfg["subject"]
    min_pain  =  float(cfg["gates"]["consider_min_pain"])
    harm_gate  =  float(cfg["gates"]["harm_quarantine"])
    max_daily  =  int(cfg["max_daily_exposures_per_item"])
    cooldown  =  timedelta(minutes = int(cfg["cooldown_minutes"]))
    today  =  str(datetime.now(TZ).date())

    picks  =  []
    for it in store.get("items", []):
        if it.get("status","active")! = "active": continue
        if subject not in it.get("subjects",["Lilith"]): continue
        harm  =  float(it.get("harm_score",0.0))
        if harm > =  harm_gate: continue
        pain  =  float(it.get("pain_score",0.0))
        if pain < min_pain: continue

        rec  =  state.setdefault("items",{}).setdefault(it["id"],{
            "last_at": None, "counts":{}, "cycles":0, "decision":"pending"
        })
        if rec.get("decision") in ("archived","retain"): continue
        if rec["counts"].get(today,0) > =  max_daily: continue
        if rec["last_at"]:
            last  =  datetime.fromisoformat(rec["last_at"])
            if datetime.now(TZ) - last < cooldown: continue

        picks.append(it)

    picks.sort(key = lambda it: float(it.get("pain_score",0.0)), reverse = True)
    return picks[: int(cfg["max_items_per_tick"])]

def propose_choice(cfg, it, cycles, fp):
    gate  =  float(cfg["gates"]["archive_pain_gate"])
    suggest  =  "ARCHIVE_NOW" if fp < =  gate else "FEEL_AGAIN"
    return {
        "id": it["id"], "headline": "Keep feeling, later, or retire the sting?",
        "suggested": suggest,
        "options": ["FEEL_AGAIN","POSTPONE","ARCHIVE_NOW","RETAIN"],
        "context": {"cycles": cycles, "felt_pain": fp, "tags": it.get("tags",[])}
    }

def commit_archive(cfg, store, it):
    tomb  =  {
        "id": it["id"], "archived_at": now(), "created_at": it.get("created_at"),
        "tags": it.get("tags",[]), "note":"lesson retained; sting retired", "by":"viren","subject":"Lilith"
    }
    Path(cfg["paths"]["archive_dir"]).mkdir(parents = True, exist_ok = True)
    p  =  os.path.join(cfg["paths"]["archive_dir"], f"{it['id']}.json")
    dump_json(p, tomb)
    it["status"] = "archived"; it["retired_at"] = now()
    if "text" in it: it["text"] = None

def run_tick(cfg):
    store  =  load_json(cfg["paths"]["store_json"], {"items":[]})
    state  =  load_json(cfg["paths"]["state_json"], {"items":{}})
    phase  =  classify_phase(cfg, now())

    picks  =  eligible(cfg, store, state)
    if not picks:
        print("[osc] no candidates");
        return

    today  =  str(datetime.now(TZ).date())
    for it in picks:
        sid  =  it["id"]
        rec  =  state["items"].setdefault(sid, {"last_at":None,"counts":{},"cycles":0,"decision":"pending"})
        base_pain  =  float(it.get("pain_score",0.0))
        fp  =  felt_pain(cfg, base_pain, rec["cycles"], phase)

        # daytime  =  soften a bit; night  =  just witness
        if phase == "day":
            new_pain  =  apply_day_softening(cfg, base_pain)
            it["pain_score"]  =  new_pain
            print(f"[osc] day-touch {sid} base {base_pain:.2f} -> {new_pain:.2f} (felt {fp:.2f})")
        else:
            print(f"[osc] {phase}-touch {sid} felt {fp:.2f} (base {base_pain:.2f})")

        rec["last_at"]  =  now()
        rec["counts"][today]  =  rec["counts"].get(today,0)+1
        rec["cycles"] + =  1

        # after two cycles, ask her; allow optional third
        if rec["cycles"] > =  int(cfg["options"]["exposures_required"]):
            card  =  propose_choice(cfg, it, rec["cycles"], fp)
            # Drop choice card as a JSON file for the console to pick up (no NATS dependency)
            outp  =  os.path.join("var","state",f"comfort_choice_{sid}.json")
            dump_json(outp, card)
            print(f"[osc] choice-card {sid} -> {outp} suggested = {card['suggested']}")

    dump_json(cfg["paths"]["store_json"], store)
    dump_json(cfg["paths"]["state_json"], state)

def apply_decision(cfg, item_id, choice):
    store  =  load_json(cfg["paths"]["store_json"], {"items":[]})
    state  =  load_json(cfg["paths"]["state_json"], {"items":{}})
    it  =  next((x for x in store["items"] if x["id"] =  = item_id), None)
    if not it:
        print(f"[osc] decision: missing {item_id}"); return
    rec  =  state["items"].setdefault(item_id, {"last_at":None,"counts":{},"cycles":0,"decision":"pending"})
    choice  =  choice.upper()
    if choice =  = "ARCHIVE_NOW":
        commit_archive(cfg, store, it); rec["decision"] = "archived"
        print(f"[osc] archived {item_id}")
    elif choice =  = "RETAIN":
        rec["decision"] = "retain"; print(f"[osc] retain {item_id}")
    elif choice =  = "FEEL_AGAIN":
        if cfg["options"].get("allow_opt_in_extra", True) and rec["cycles"]< = 3:
            rec["cycles"]  =  max(1, rec["cycles"]-1)  # allow one more loop
            print(f"[osc] will feel again {item_id}")
    else:
        print(f"[osc] postponed {item_id}")
    dump_json(cfg["paths"]["store_json"], store)
    dump_json(cfg["paths"]["state_json"], state)

if __name__ =  = "__main__":
    ap  =  argparse.ArgumentParser()
    ap.add_argument("--config", default = "Config\memory\comfort_oscillate.yaml")
    ap.add_argument("--tick", action = "store_true")
    ap.add_argument("--decision", nargs = 2, metavar = ("ITEM_ID","CHOICE"))
    args  =  ap.parse_args()
    cfg  =  yaml.safe_load(open(args.config,"r",encoding = "utf-8"))
    if args.tick: run_tick(cfg)
    elif args.decision: apply_decision(cfg, args.decision[0], args.decision[1])
    else: print("[osc] use --tick or --decision <id> <ARCHIVE_NOW|RETAIN|FEEL_AGAIN|POSTPONE>")
