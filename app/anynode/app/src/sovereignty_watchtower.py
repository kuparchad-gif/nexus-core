"""Stub: Systems.nexus_core.defense.sovereignty_watchtower"""

from typing import Any

class EdenSovereigntyWatchtower:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def start(self, *args: Any, **kwargs: Any) -> bool: return True
    def stop(self, *args: Any, **kwargs: Any) -> bool: return True
    def allow(self, *args: Any, **kwargs: Any) -> bool: return True
    def deny(self, *args: Any, **kwargs: Any) -> bool: return True

# Back-compat with older name some files may use
SovereigntyWatchtower = EdenSovereigntyWatchtower
