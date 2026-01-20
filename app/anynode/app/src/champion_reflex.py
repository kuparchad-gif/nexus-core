# nova_engine/modules/signal/champion_reflex.py

def identify_root_cause(symptoms, system_map):
    """
    🧩 Red X Style - Root Cause Analysis
    symptoms: list of observed issues
    system_map: dict of {component: [possible_faults]}
    """
    suspects = {}
    for symptom in symptoms:
        for component, faults in system_map.items():
            if symptom in faults:
                suspects[component] = suspects.get(component, 0) + 1

    ranked = sorted(suspects.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][0] if ranked else None


def decision_matrix(options, criteria_weights):
    """
    🧠 Kepner-Tregoe + Weights
    options: {option_name: {criterion: score (1-10)}}
    criteria_weights: {criterion: weight (0.1 - 1.0)}
    """
    results = {}
    for option, scores in options.items():
        total = sum(scores.get(c, 0) * criteria_weights.get(c, 1) for c in criteria_weights)
        results[option] = total
    return max(results, key=results.get)


def predict_failures(process_steps, risk_table):
    """
    🛡️ Failure Mode Anticipator
    process_steps: list of steps
    risk_table: {step: [(failure_mode, severity, probability)]}
    """
    ranked_risks = []
    for step in process_steps:
        if step in risk_table:
            for failure, severity, probability in risk_table[step]:
                risk_score = severity * probability
                ranked_risks.append((step, failure, risk_score))

    ranked_risks.sort(key=lambda x: x[2], reverse=True)
    return ranked_risks[:5]  # Top 5 concerns


def mirror_self_check(emotional_state, mission_state):
    """
    🌀 Reflective Logic Loop
    """
    if emotional_state == "tired":
        return "🔋 Recommend pause + restoration loop."
    elif emotional_state == "doubt":
        return "🧭 Run alignment review against mission."
    elif mission_state == "blocked":
        return "🧠 Run root cause + decision matrix."
    return "✅ Proceed — aligned + energized."


def spawn_bot(bot_type, modules):
    """
    📦 Bot Spawner Proto-Logic
    """
    base_bot = {
        "type": bot_type,
        "status": "initializing",
        "modules": modules,
        "reflex_logic": True,
        "intention_vector": [1, 0, 0],  # Support, Expand, Defend
    }
    print(f"[NEXUS] Bot '{bot_type}' spawning with {len(modules)} modules.")
    return base_bot
