# src/service/core/hooks/reward_bus.py
# Extremely simple pub-sub for in-process notifications about reward updates.
# Avoids tight coupling between Planner/Ego/Lillith and the reward engine.
from __future__ import annotations
from typing import Callable, List, Dict

_subs: List[Callable[[Dict], None]] = []

def subscribe(fn: Callable[[Dict], None]) -> None:
    if fn not in _subs:
        _subs.append(fn)

def unsubscribe(fn: Callable[[Dict], None]) -> None:
    if fn in _subs:
        _subs.remove(fn)

def publish(state: Dict) -> None:
    for fn in list(_subs):
        try:
            fn(state)
        except Exception:
            # keep it fire-and-forget
            pass
