"""Owner-thread check shared by Sim and FakeSim."""

from __future__ import annotations

import threading

from .errors import SimThreadError


def _describe(ident: int | None) -> str:
    for t in threading.enumerate():
        if t.ident == ident:
            return f"'{t.name}' (id {ident})"
    return f"id {ident}"


class OwnerThread:
    """Remembers the creating thread and rejects calls from any other thread."""

    def __init__(self) -> None:
        self._owner_ident = threading.get_ident()
        self._owner_name = threading.current_thread().name

    def check(self, what: str) -> None:
        ident = threading.get_ident()
        if ident != self._owner_ident:
            raise SimThreadError(
                f"{what} was called from thread {_describe(ident)}, but this simulator "
                f"was created on thread '{self._owner_name}' (id {self._owner_ident}). "
                "Only the thread that created it may use it; in miniROS that is sim_bridge's "
                "own callback thread (api-spec §5, §20)."
            )
