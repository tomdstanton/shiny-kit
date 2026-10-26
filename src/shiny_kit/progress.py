"""Zero-dependency, context-propagated hierarchical progress reporting."""

import asyncio
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import shiny.ui as shiny_ui
from shiny import Session

type ProgressCallback = Callable[[float, str | None, str | None], None]

_SHARED_VAR_ATTR = "__active_progress_contextvar__"


def _get_shared_contextvar() -> ContextVar[Any]:
    var = getattr(sys, _SHARED_VAR_ATTR, None)
    if not isinstance(var, ContextVar):
        var = ContextVar(_SHARED_VAR_ATTR, default=None)
        setattr(sys, _SHARED_VAR_ATTR, var)
    return var


@dataclass(slots=True)
class _SharedProgressState:
    """Shared root state across a hierarchy of scoped `ProgressReporter` instances."""

    callback: ProgressCallback | None
    min_interval_sec: float = 0.0
    last_emit_time: float = 0.0
    max_fraction: float = 0.0


@dataclass(slots=True)
class ProgressReporter:
    """Hierarchical, rate-throttled progress reporter with sub-range composition.

    Maps local progress fractions in `[0.0, 1.0]` into `[start, end]` of the root task,
    ensuring nested estimators and cross-validation folds advance monotonically from 0% to 100%.

    Attributes:
        start (float): Lower bound of this reporter's progress interval within `[0.0, 1.0]`.
        end (float): Upper bound of this reporter's progress interval within `[0.0, 1.0]`.
        default_message (str | None): Default primary status message when `report()` omits `message`.
        detail_prefix (str | None): Optional hierarchical prefix prepended to `detail` strings.

    Examples:
        ```python
        from shiny_kit.progress import current_progress, progress_context

        events = []
        with progress_context(lambda frac, msg, det: events.append((frac, msg, det))):
            prog = current_progress()
            prog.report(0.2, message="Loading data")
            with prog.subrange(0.2, 1.0, detail_prefix="Fold 1"):
                current_progress().report(0.5, detail="Fitting")
        ```
    """

    _shared: _SharedProgressState = field(default_factory=lambda: _SharedProgressState(callback=None))
    start: float = 0.0
    end: float = 1.0
    default_message: str | None = None
    detail_prefix: str | None = None

    @property
    def is_active(self) -> bool:
        """Whether an active listener callback is bound to this reporter.

        Returns:
            `True` if a `ProgressCallback` is attached, `False` for the default no-op reporter.
        """
        return self._shared.callback is not None

    active = is_active

    def report(
        self,
        fraction: float,
        message: str | None = None,
        detail: str | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Emits a progress update mapped into `[self.start, self.end]`.

        Args:
            fraction: Local completion fraction in `[0.0, 1.0]`.
            message: Optional primary status heading.
            detail: Optional secondary detail string.
            force: If `True`, bypasses rate-limiting `min_interval_sec` throttling.
        """
        cb = self._shared.callback
        if cb is None:
            return

        if message is not None and self.detail_prefix is None:
            self.default_message = message

        clamped = min(1.0, max(0.0, float(fraction)))
        span = max(0.0, self.end - self.start)
        mapped = min(1.0, max(self._shared.max_fraction, self.start + clamped * span))

        now = time.monotonic()
        if (
            not force
            and self._shared.min_interval_sec > 0.0
            and clamped < 1.0
            and (now - self._shared.last_emit_time) < self._shared.min_interval_sec
        ):
            return

        self._shared.last_emit_time = now
        self._shared.max_fraction = mapped

        if self.detail_prefix is not None:
            eff_message = self.default_message if self.default_message is not None else message
        else:
            eff_message = message if message is not None else self.default_message
        if self.detail_prefix and detail:
            eff_detail: str | None = f"{self.detail_prefix} · {detail}"
        elif self.detail_prefix:
            eff_detail = self.detail_prefix
        else:
            eff_detail = detail

        cb(mapped, eff_message, eff_detail)

    @contextmanager
    def subrange(
        self,
        start: float,
        end: float,
        *,
        message: str | None = None,
        detail_prefix: str | None = None,
    ) -> Generator["ProgressReporter"]:
        """Scopes nested `current_progress()` calls into `[start, end]` of the current range.

        Args:
            start: Relative start fraction in `[0.0, 1.0]` within the parent reporter's range.
            end: Relative end fraction in `[start, 1.0]` within the parent reporter's range.
            message: Optional override for `default_message` inside the subrange.
            detail_prefix: Optional prefix prepended to nested `detail` strings.

        Yields:
            Scoped child `ProgressReporter` bound to `_ACTIVE_PROGRESS`.
        """
        if not self.is_active:
            yield self
            return

        s_clamped = min(1.0, max(0.0, float(start)))
        e_clamped = min(1.0, max(s_clamped, float(end)))
        span = max(0.0, self.end - self.start)
        child_start = self.start + s_clamped * span
        child_end = self.start + e_clamped * span

        if self.detail_prefix and detail_prefix:
            combined_prefix: str | None = f"{self.detail_prefix} · {detail_prefix}"
        else:
            combined_prefix = detail_prefix or self.detail_prefix

        child = ProgressReporter(
            _shared=self._shared,
            start=child_start,
            end=child_end,
            default_message=message if message is not None else self.default_message,
            detail_prefix=combined_prefix,
        )
        token = _ACTIVE_PROGRESS.set(child)
        try:
            yield child
        finally:
            _ACTIVE_PROGRESS.reset(token)


_NOOP_REPORTER = ProgressReporter()
_ACTIVE_PROGRESS: ContextVar[ProgressReporter | None] = _get_shared_contextvar()


def current_progress() -> ProgressReporter:
    """Returns the context-local `ProgressReporter` (or a zero-cost no-op reporter if unbound).

    Returns:
        Active `ProgressReporter` for the current execution context.
    """
    active = _ACTIVE_PROGRESS.get()
    return active if active is not None else _NOOP_REPORTER


@contextmanager
def progress_context(
    callback: ProgressCallback,
    *,
    message: str | None = None,
    min_interval_sec: float = 0.0,
) -> Generator[ProgressReporter]:
    """Binds a root `ProgressReporter` to the current execution context (propagates into `asyncio.to_thread`).

    Args:
        callback: Listener callable `(fraction, message, detail) -> None` invoked on progress updates.
        message: Optional initial default status message.
        min_interval_sec: Minimum elapsed seconds between consecutive non-final progress emissions.

    Yields:
        Root `ProgressReporter` bound to the current `contextvars` context.
    """
    shared = _SharedProgressState(callback=callback, min_interval_sec=min_interval_sec)
    reporter = ProgressReporter(_shared=shared, start=0.0, end=1.0, default_message=message)
    token = _ACTIVE_PROGRESS.set(reporter)
    try:
        yield reporter
    finally:
        _ACTIVE_PROGRESS.reset(token)


class LazyShinyProgress:
    """Thread-safe lazy bridge from `ProgressReporter` to `shiny.ui.Progress`.

    Suppresses opening the Shiny progress notification if the task finishes within
    `grace_sec` (default 150ms), so instant `BaseCache` hits or trivial operations
    never flash a progress bar. If the task is still running after `grace_sec`, opens
    and updates `shiny.ui.Progress` on the main asyncio event loop.
    """

    def __init__(
        self,
        session: Session | None,
        *,
        default_message: str = "Working...",
        grace_sec: float = 0.15,
        loop: asyncio.AbstractEventLoop | None = None,
        progress_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._session = session
        self._grace_sec = max(0.0, float(grace_sec))
        self._loop = loop
        self._progress_factory = progress_factory
        self._start_time = time.monotonic()
        self._latest: tuple[float, str | None, str | None] = (0.0, default_message, None)
        self._progress: Any | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._closed: bool = False

        if self._loop is not None and (self._session is not None or self._progress_factory is not None):
            if self._grace_sec <= 0.0:
                self._flush_on_loop()
            else:
                self._timer = self._loop.call_later(self._grace_sec, self._flush_on_loop)

    def update(self, fraction: float, message: str | None, detail: str | None) -> None:
        """Records the latest milestone and dispatches to the Shiny event loop if past grace period."""
        if self._closed:
            return
        prev_msg = self._latest[1]
        self._latest = (fraction, message if message is not None else prev_msg, detail)
        if self._loop is None or (self._session is None and self._progress_factory is None):
            return
        elapsed = time.monotonic() - self._start_time
        if self._progress is not None or elapsed >= self._grace_sec:
            try:
                self._loop.call_soon_threadsafe(self._flush_on_loop)
            except RuntimeError:
                pass

    def _flush_on_loop(self) -> None:
        if self._closed:
            return
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._progress is None:
            if self._progress_factory is not None:
                self._progress = self._progress_factory()
            elif self._session is not None:
                self._progress = shiny_ui.Progress(min=0, max=1, session=self._session)
            else:
                return
        fraction, message, detail = self._latest
        self._progress.set(
            value=fraction,
            message=message or None,
            detail=detail or None,
        )

    def close(self) -> None:
        """Cancels any pending grace timer and closes the Shiny progress bar if opened."""
        if self._closed:
            return
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._progress is not None:
            try:
                self._progress.close()
            finally:
                self._progress = None
