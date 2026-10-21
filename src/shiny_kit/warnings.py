"""Context manager for intercepting, filtering, and deduplicating user-facing warnings."""

import warnings
from collections.abc import Iterable
from types import TracebackType
from typing import Any, Self


class WarningCatcher:
    """Context manager that intercepts, filters, deduplicates, and optionally re-emits user-facing warnings.

    Attributes:
        reemit_on_exit (bool): Whether captured warnings should be re-emitted via `warnings.warn` on `__exit__`.
        messages (tuple[str, ...]): Deduplicated tuple of captured warning strings after context exit.

    Examples:
        ```python
        import warnings
        from shiny_kit.warnings import WarningCatcher

        with WarningCatcher() as catcher:
            warnings.warn("Unpadded strata detected", UserWarning)
        assert catcher.messages == ("Unpadded strata detected",)
        ```
    """

    def __init__(self, *, reemit: bool = False) -> None:
        """Initializes a `WarningCatcher` context manager.

        Args:
            reemit: Whether to re-emit captured user warnings when exiting the `with` block.
        """
        self.reemit_on_exit = reemit
        self.messages: tuple[str, ...] = ()
        self._catch_ctx: Any = None
        self._recorded: list[warnings.WarningMessage] | None = None

    @staticmethod
    def is_relevant(w: warnings.WarningMessage) -> bool:
        """Returns True if `w` is an explicit `UserWarning` or originates from non-library application code.

        Args:
            w: Recorded `warnings.WarningMessage` instance.

        Returns:
            `True` if the warning should be surfaced to the user, `False` otherwise.
        """
        if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning, ResourceWarning, ImportWarning)):
            return False
        if issubclass(w.category, UserWarning):
            return True
        norm_file = (w.filename or "").replace("\\", "/")
        return bool(norm_file and "/site-packages/" not in norm_file and "/lib/python" not in norm_file)

    @staticmethod
    def merge(*message_iters: Iterable[str]) -> tuple[str, ...]:
        """Deduplicates warning messages across iterables while preserving first-seen order.

        Args:
            *message_iters: Variable number of iterables yielding warning strings.

        Returns:
            Deduplicated tuple of stripped, non-empty warning messages.
        """
        seen: dict[str, None] = {}
        for it in message_iters:
            for msg in it:
                clean = str(msg).strip()
                if clean and clean not in seen:
                    seen[clean] = None
        return tuple(seen.keys())

    @staticmethod
    def emit_messages(messages: Iterable[str], *, stacklevel: int = 2) -> None:
        """Emits each message in `messages` as a standard `UserWarning`.

        Args:
            messages: Iterable of warning strings to emit.
            stacklevel: Stack level passed to `warnings.warn`. Defaults to `2`.
        """
        for msg in WarningCatcher.merge(messages):
            warnings.warn(msg, UserWarning, stacklevel=stacklevel)

    def __enter__(self) -> Self:
        """Enters the `warnings.catch_warnings(record=True)` context."""
        self._catch_ctx = warnings.catch_warnings(record=True)
        self._recorded = self._catch_ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exits the warning capture context, populates `self.messages`, and optionally re-emits."""
        if self._catch_ctx is not None:
            self._catch_ctx.__exit__(exc_type, exc_val, exc_tb)
        raw_msgs = [str(w.message).strip() for w in (self._recorded or []) if self.is_relevant(w)]
        self.messages = self.merge(raw_msgs)
        if self.reemit_on_exit and exc_type is None and self.messages:
            self.emit_messages(self.messages, stacklevel=3)
