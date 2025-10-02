from __future__ import annotations

import asyncio
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def ensure_proactor_event_loop_policy() -> None:
    """Ensure Playwright can spawn subprocesses on Windows."""

    if not sys.platform.startswith("win"):
        return
    proactor_cls = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    if proactor_cls is None:
        return
    current_policy = asyncio.get_event_loop_policy()
    if isinstance(current_policy, proactor_cls):
        return
    try:
        asyncio.set_event_loop_policy(proactor_cls())
    except RuntimeError:
        # Another loop is already running; nothing we can do safely here.
        pass
