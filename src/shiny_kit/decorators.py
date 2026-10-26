"""Reactive descriptors and class-based server binding decorators (`BaseBinder`, `bind_*`)."""

import asyncio
import importlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import shiny.ui as shiny_ui
from shiny import Inputs, Outputs, Session, reactive, render, req
from shiny.session import get_current_session
from shiny.types import SilentCancelOutputException, SilentException

from shiny_kit.progress import LazyShinyProgress, progress_context
from shiny_kit.warnings import WarningCatcher

if TYPE_CHECKING:
    from shiny_kit.component import Component


@lru_cache(maxsize=256)
def _callable_param_count(fn: Callable[..., Any]) -> int:
    """Returns the number of parameters accepted by `fn` (cached across invocations)."""
    try:
        return len(inspect.signature(fn).parameters)
    except (ValueError, TypeError):
        return 1


def _wrap_sync_or_async(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    before: Callable[[], Any] | None = None,
    after: Callable[[Any], Any] | None = None,
) -> Callable[..., Any]:
    """Wraps a synchronous or asynchronous callable with optional `before` and `after` hooks."""
    resolved_name = name or getattr(fn, "__name__", "wrapped")
    if inspect.iscoroutinefunction(fn):

        async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
            if before is not None:
                before()
            res = await fn(*args, **kwargs)
            return after(res) if after is not None else res

        _async_wrapped.__name__ = resolved_name
        return _async_wrapped

    def _sync_wrapped(*args: Any, **kwargs: Any) -> Any:
        if before is not None:
            before()
        res = fn(*args, **kwargs)
        return after(res) if after is not None else res

    _sync_wrapped.__name__ = resolved_name
    return _sync_wrapped


class reactive_prop[T]:
    """Descriptor wrapping a PyShiny `reactive.value` with defaults and static type inference.

    Attributes:
        default (T | None): Default value stored in the per-instance `reactive.value`.
        name (str): Internal instance dictionary key storing the `reactive.Value`.

    Examples:
        ```python
        from shiny_kit.decorators import reactive_prop
        from shiny_kit.state import BaseSessionState

        class AppState(BaseSessionState):
            selected_year: reactive_prop[int] = reactive_prop(2024)
        ```
    """

    def __init__(self, default: T | None = None) -> None:
        self.default = default
        self.name: str = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.name = f"_{name}_reactive"

    def __get__(self, instance: Any, owner: type[Any]) -> T:
        if instance is None:
            return self  # ty: ignore[invalid-return-type]
        if self.name not in instance.__dict__:
            instance.__dict__[self.name] = reactive.value(self.default)
        return instance.__dict__[self.name].get()

    def __set__(self, instance: Any, value: T) -> None:
        if self.name not in instance.__dict__:
            instance.__dict__[self.name] = reactive.value(value)
        else:
            instance.__dict__[self.name].set(value)


class reactive_calc:
    """Descriptor wrapping a method into a per-instance, session-isolated `reactive.calc`.

    Supports synchronous and asynchronous functions as well as optional input event filtering.

    Attributes:
        fn (Callable[..., Any] | None): Underlying method wrapped by the descriptor.
        events (tuple[str, ...]): Optional input IDs to trigger reactive recalculation via `reactive.event`.
        name (str): Attribute name on the owning class.
    """

    def __init__(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        events: tuple[str, ...] = (),
    ) -> None:
        self.fn = fn
        self.events = events
        self.name: str = fn.__name__ if fn else ""  # ty: ignore[unresolved-attribute]

    def __call__(self, fn: Callable[..., Any]) -> Self:
        self.fn = fn
        self.name = fn.__name__  # ty: ignore[unresolved-attribute]
        return self

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: type[Any]) -> Any:
        if instance is None:
            return self

        cache_key = f"_calc_{self.name}"
        if cache_key not in instance.__dict__:
            if self.fn is None:
                raise RuntimeError(f"reactive_calc '{self.name}' has no underlying callable.")

            bound_method = self.fn.__get__(instance, owner)  # ty: ignore[unresolved-attribute]
            if self.events and getattr(instance, "_input", None) is not None:
                triggers = [instance.input[eid] for eid in self.events]
                wrapped = reactive.event(*triggers)(bound_method)
            else:
                wrapped = bound_method

            instance.__dict__[cache_key] = reactive.calc(wrapped)

        return instance.__dict__[cache_key]


class BaseBinder(ABC):
    """Abstract base descriptor and decorator for binding `Component` methods to a PyShiny server session.

    All `bind_*` decorators (`bind_output`, `bind_datagrid`, `bind_download`, `bind_effect`, `bind_task`,
    and custom domain binders) inherit from `BaseBinder` and implement `bind(...)`, allowing
    `Component.server()` to discover and bind them uniformly without hardcoded per-type loops.

    Attributes:
        fn (Callable[..., Any] | None): Underlying component method decorated by this binder.
        name (str): Attribute name on the owning `Component` class.
    """

    def __init__(self, fn: Callable[..., Any] | None = None) -> None:
        self.fn: Callable[..., Any] | None = fn
        self.name: str = getattr(fn, "__name__", "") if fn is not None else ""

    def __call__(self, fn: Callable[..., Any]) -> Self:
        self.fn = fn
        if not self.name:
            self.name = getattr(fn, "__name__", "")
        return self

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: type[Any]) -> Any:
        if instance is None:
            return self
        if self.fn is None:
            raise RuntimeError(f"{type(self).__name__} '{self.name}' has no underlying callable.")
        return self.fn.__get__(instance, owner)  # ty: ignore[unresolved-attribute]

    @abstractmethod
    def bind(
        self,
        component: "Component[Any]",
        input: Inputs,
        output: Outputs,
        session: Session,
    ) -> None:
        """Binds this descriptor's method onto the active PyShiny module server session."""
        raise NotImplementedError


class bind_output(BaseBinder):
    """Descriptor & decorator binding a `Component` method (sync or async) to a Shiny renderer.

    Subclasses (such as `bind_datagrid` or `bind_altair`) can set a default `renderer_factory`
    and override `transform(self, component, value)` to post-process the method's return value
    without writing manual sync/async wrapper boilerplate.

    Attributes:
        renderer_factory (Any): Shiny renderer decorator/factory (e.g. `render.ui`, `render.text`).
        id_ (str | None): Optional explicit output ID. Defaults to the decorated method's name.
    """

    def __init__(
        self,
        renderer_factory: Any,
        id_: str | None = None,
        *,
        id_attr: str | None = None,
    ) -> None:
        super().__init__()
        self.renderer_factory = renderer_factory
        self.id_ = id_
        self.id_attr = id_attr

    @property
    def target_id(self) -> str:
        """Resolved Shiny output ID."""
        return self.id_ or self.name

    def resolve_target_id(self, component: "Component[Any] | None" = None) -> str:
        """Resolves the Shiny output ID from `component.<id_attr>`, `self.id_`, or `self.name`."""
        if component is not None and self.id_attr:
            attr_val = getattr(component, self.id_attr, None)
            if isinstance(attr_val, str) and attr_val:
                return attr_val
        return self.id_ or self.name

    def client_dimensions(self, component: "Component[Any]") -> tuple[float | None, float | None]:
        """Reads `(width, height)` in pixels from `session.clientdata` for the resolved output ID."""
        w: float | None = None
        h: float | None = None
        sess = getattr(component, "_session", None)
        target_id = self.resolve_target_id(component)
        if sess is not None and hasattr(sess, "clientdata"):
            try:
                w = sess.clientdata.output_width(target_id)
            except Exception:
                w = None
            try:
                h = sess.clientdata.output_height(target_id)
            except Exception:
                h = None
        return w, h

    def transform(self, component: "Component[Any]", value: Any) -> Any:
        """Hook for subclasses to transform the method's return value before passing it to the renderer."""
        return value

    def __get__(self, instance: Any, owner: type[Any]) -> Any:
        if instance is None:
            return self
        if self.fn is None:
            raise RuntimeError(f"{type(self).__name__} '{self.name}' has no underlying callable.")

        bound_method = self.fn.__get__(instance, owner)  # ty: ignore[unresolved-attribute]
        if type(self).transform is bind_output.transform:
            return bound_method

        return _wrap_sync_or_async(
            bound_method,
            name=self.name,
            after=lambda val: self.transform(instance, val),
        )

    @staticmethod
    def _wrap_gated_renderer(
        component: "Component[Any]", method: Callable[..., Any], target_id: str
    ) -> Callable[..., Any]:
        """Wraps a renderer method with its target output ID and optional `requires` gate."""
        gate = (lambda: req(component.is_unlocked())) if component.requires else None
        return _wrap_sync_or_async(method, name=target_id, before=gate)

    def bind(
        self,
        component: "Component[Any]",
        input: Inputs,
        output: Outputs,
        session: Session,
    ) -> None:
        """Wraps the bound method with `requires` gating and registers it on `output`."""
        method = getattr(component, self.name)
        target_id = self.resolve_target_id(component)
        guarded = self._wrap_gated_renderer(component, method, target_id)
        output(id=target_id)(self.renderer_factory(guarded))


class bind_datagrid(bind_output):
    """Descriptor & decorator binding a method returning a tabular object to `shiny.render.DataGrid`.

    Attributes:
        selection_mode (Literal["none", "row", "rows"]): Row selection mode.
        filters (bool): Whether to enable per-column filter headers in the DataGrid.
        height (str | int | None): Optional CSS height string or pixel integer.
        height_attr (str | None): Optional attribute name on the component resolving the DataGrid height.
        width (str | int): CSS width string or pixel integer. Defaults to `"100%"`.
    """

    def __init__(
        self,
        id_: str | None = None,
        *,
        id_attr: str | None = None,
        selection_mode: Literal["none", "row", "rows"] = "none",
        filters: bool = True,
        height: str | int | None = None,
        height_attr: str | None = None,
        width: str | int = "100%",
    ) -> None:
        super().__init__(render.data_frame, id_=id_, id_attr=id_attr)
        self.selection_mode: Literal["none", "row", "rows"] = selection_mode
        self.filters = filters
        self.height = height
        self.height_attr = height_attr
        self.width = width

    def transform(self, component: "Component[Any]", value: Any) -> Any:
        """Wraps non-null tabular values into `shiny.render.DataGrid`."""
        if value is not None and not isinstance(value, (render.DataGrid, render.DataTable)):
            resolved_height = getattr(component, self.height_attr, self.height) if self.height_attr else self.height
            return render.DataGrid(
                value,
                selection_mode=self.selection_mode,
                filters=self.filters,
                height=resolved_height,
                width=self.width,
            )
        return value


class bind_widget(bind_output):
    """Descriptor & decorator binding a method returning an interactive widget/chart to `shinywidgets`.

    Lazily imports `shinywidgets` (defaulting to `shinywidgets.render_altair` or `render_widget`)
    and optionally adapts the returned widget to its client container dimensions via a pluggable
    `resizer` callback `(widget, width, height) -> widget` (or `component.resizer`).

    Attributes:
        autosize (bool): Whether to resize the returned widget using `client_dimensions(component)`.
        resizer (Callable[[Any, float | None, float | None], Any] | None): Optional resizer callback.
    """

    def __init__(
        self,
        id_: str | None = None,
        *,
        id_attr: str | None = None,
        renderer_factory: Any = None,
        autosize: bool = True,
        resizer: Callable[[Any, float | None, float | None], Any] | None = None,
    ) -> None:
        resolved_factory = renderer_factory
        if resolved_factory is None:
            try:
                resolved_factory = self._default_renderer_factory()
            except Exception:
                resolved_factory = None
        super().__init__(resolved_factory, id_=id_, id_attr=id_attr)
        self.autosize = autosize
        self.resizer = resizer

    @staticmethod
    def _default_renderer_factory() -> Any:
        shinywidgets = importlib.import_module("shinywidgets")
        return getattr(shinywidgets, "render_altair", shinywidgets.render_widget)

    def transform(self, component: "Component[Any]", value: Any) -> Any:
        """Resizes the returned widget to fit the active client container dimensions when `autosize=True`."""
        if not self.autosize or value is None:
            return value
        w, h = self.client_dimensions(component)
        resizer_fn = self.resizer
        if resizer_fn is None:
            comp_resizer = getattr(type(component), "resizer", None) or getattr(component, "resizer", None)
            if callable(comp_resizer):
                resizer_fn = comp_resizer
        if resizer_fn is not None:
            return resizer_fn(value, w, h)
        if hasattr(value, "fit_to_container") and callable(value.fit_to_container):
            return value.fit_to_container(width=w, height=h)
        return value

    def bind(
        self,
        component: "Component[Any]",
        input: Inputs,
        output: Outputs,
        session: Session,
    ) -> None:
        if self.renderer_factory is None:
            self.renderer_factory = self._default_renderer_factory()
        super().bind(component, input, output, session)


class bind_download(bind_output):
    """Descriptor & decorator binding a method yielding or returning bytes to `@render.download_button`.

    Attributes:
        id_ (str | None): Optional explicit download button output ID.
        filename (str | Callable[[Any], str]): Static filename string or callable `(self) -> str`.
    """

    def __init__(
        self,
        id_: str | None = None,
        *,
        filename: str | Callable[[Any], str] = "download.bin",
    ) -> None:
        super().__init__(render.download_button, id_=id_)
        self.filename = filename

    def bind(
        self,
        component: "Component[Any]",
        input: Inputs,
        output: Outputs,
        session: Session,
    ) -> None:
        """Wraps the download generator with `requires` gating and registers it on `output`."""
        method = getattr(component, self.name)
        dl_id = self.target_id
        raw_fn = self.filename
        if callable(raw_fn):

            def _resolve_filename() -> str:
                return str(raw_fn(component))

            fn_arg: str | Callable[[], str] | None = _resolve_filename
        else:
            fn_arg = raw_fn
        guarded = self._wrap_gated_renderer(component, method, dl_id)
        output(id=dl_id)(render.download_button(filename=fn_arg)(guarded))


class bind_effect(BaseBinder):
    """Descriptor & decorator binding a `Component` method (sync or async) to a guarded reactive effect.

    Attributes:
        event_ids (tuple[str, ...]): Input IDs triggering the effect via `reactive.event`.
        notify_error (bool): Whether uncaught non-silent exceptions are displayed as error toasts.
        notify_warning (bool): Whether captured user-facing warnings are displayed as warning toasts.
    """

    def __init__(
        self,
        *input_ids: str,
        notify_error: bool = True,
        notify_warning: bool = True,
    ) -> None:
        super().__init__()
        self.event_ids: tuple[str, ...] = input_ids
        self.notify_error = notify_error
        self.notify_warning = notify_warning

    @staticmethod
    def _is_silent_exception(exc: BaseException) -> bool:
        """Returns True if `exc` is a PyShiny reactive control-flow or cancellation exception."""
        return isinstance(exc, (SilentException, SilentCancelOutputException, asyncio.CancelledError))

    def _wrap_guarded_effect(self, component: "Component[Any]", method: Callable[..., Any]) -> Callable[..., Any]:
        """Wraps a `@bind_effect` method to capture user-facing warnings and catch non-silent exceptions as UI toasts."""
        notify_err = bool(getattr(method, "_notify_error", self.notify_error))
        notify_warn = bool(getattr(method, "_notify_warning", self.notify_warning))
        fn_name = getattr(method, "__name__", "effect")

        def _emit_warnings(messages: tuple[str, ...]) -> None:
            if notify_warn and messages:
                with reactive.isolate():
                    for w_msg in messages:
                        component.notify_warning(w_msg)

        def _handle_exc(exc: BaseException) -> None:
            if self._is_silent_exception(exc) or not notify_err:
                raise exc
            with reactive.isolate():
                component.notify_error(exc)

        if inspect.iscoroutinefunction(method):

            async def _async_effect() -> Any:
                try:
                    with WarningCatcher() as catcher:
                        res = await method()
                    _emit_warnings(catcher.messages)
                    return res
                except BaseException as exc:
                    _handle_exc(exc)
                    return None

            _async_effect.__name__ = fn_name
            return _async_effect

        def _sync_effect() -> Any:
            try:
                with WarningCatcher() as catcher:
                    res = method()
                _emit_warnings(catcher.messages)
                return res
            except BaseException as exc:
                _handle_exc(exc)
                return None

        _sync_effect.__name__ = fn_name
        return _sync_effect

    def bind(
        self,
        component: "Component[Any]",
        input: Inputs,
        output: Outputs,
        session: Session,
    ) -> None:
        """Wraps the effect method with `WarningCatcher` and error handling and registers `reactive.effect`."""
        method = getattr(component, self.name)
        guarded_effect = self._wrap_guarded_effect(component, method)
        if self.event_ids:
            triggers = [input[eid] for eid in self.event_ids]
            reactive.effect(priority=10)(reactive.event(*triggers)(guarded_effect))
        else:
            reactive.effect(priority=10)(guarded_effect)


class BoundTask[R]:
    """Per-instance handle to a PyShiny `ExtendedTask` bound via `@bind_task`.

    Attributes:
        task (reactive.ExtendedTask[Any, R]): Underlying PyShiny `ExtendedTask`.
        invocation_count (int): Monotonically increasing count of task invocations.
        warnings (tuple[str, ...]): User-facing warnings captured during the latest run.
        last_error (BaseException | None): Exception raised by the latest failed run, if any.
    """

    def __init__(self, task: reactive.ExtendedTask[Any, R]) -> None:
        self.task = task
        self.invocation_count: int = 0
        self.warnings: tuple[str, ...] = ()
        self.last_error: BaseException | None = None

    def invoke(self, *args: Any, **kwargs: Any) -> None:
        """Invokes the background task with the given arguments."""
        self.invocation_count += 1
        self.task.invoke(*args, **kwargs)

    __call__ = invoke

    def result(self) -> R:
        """Reactively retrieves the completed task result (raises silent cancel while running)."""
        return self.task.result()

    def status(self) -> Literal["initial", "running", "success", "error", "cancelled"]:
        """Reactively returns the current execution status of the task."""
        return self.task.status()

    def cancel(self) -> None:
        """Cancels an in-flight background execution."""
        self.task.cancel()


class bind_task(BaseBinder):
    """Descriptor & decorator wrapping a method into a non-blocking PyShiny `ExtendedTask`.

    Synchronous methods are automatically offloaded to a background worker thread via
    `asyncio.to_thread` so heavy workloads never block the main asyncio event loop.
    Supports declarative `register_to` artifact registration, `on_success` callbacks,
    automatic `LazyShinyProgress` reporting, and toast notifications upon completion.
    """

    def __init__(
        self,
        button_id: str | None = None,
        *,
        register_to: str | None = None,
        on_success: Callable[[Any, Any], str | None] | str | None = None,
        notify_success: str | Callable[[Any], str] | bool = True,
        notify_warning: bool = True,
        notify_error: bool = True,
        progress_message: str | None = None,
        **register_kwargs: Any,
    ) -> None:
        super().__init__()
        self.button_id = button_id
        self.register_to = register_to
        self.register_kwargs = register_kwargs
        self.on_success = on_success
        self.notify_success = notify_success
        self.notify_warning = notify_warning
        self.notify_error = notify_error
        self.progress_message = progress_message

    @property
    def display_label(self) -> str:
        """Human-readable label for progress bars and error notifications."""
        if self.progress_message:
            return self.progress_message
        cleaned = self.name.replace("task_", "").replace("run_", "").replace("_", " ").strip().capitalize()
        return cleaned or "Task"

    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: Any, owner: type[Any]) -> BoundTask[Any]: ...

    def __get__(self, instance: Any, owner: type[Any]) -> BoundTask[Any] | Self:
        if instance is None:
            return self

        cache_key = f"_task_{self.name}"
        if cache_key not in instance.__dict__:
            if self.fn is None:
                raise RuntimeError(f"bind_task '{self.name}' has no underlying callable.")

            bound_method = self.fn.__get__(instance, owner)  # ty: ignore[unresolved-attribute]
            is_coro = inspect.iscoroutinefunction(self.fn)
            default_msg = self.display_label if self.display_label != "Task" else "Working..."
            holder: list[BoundTask[Any]] = []

            def _sync_worker(*args: Any, **kwargs: Any) -> tuple[Any, tuple[str, ...]]:
                with WarningCatcher() as catcher, reactive.isolate():
                    res = bound_method(*args, **kwargs)
                return res, WarningCatcher.merge(catcher.messages, getattr(res, "warnings_", ()))

            async def _async_runner(*args: Any, **kwargs: Any) -> Any:
                if holder:
                    holder[0].warnings = ()
                    holder[0].last_error = None
                loop = asyncio.get_running_loop()
                sess = getattr(instance, "_session", None) or get_current_session()
                lazy_prog = LazyShinyProgress(sess, default_message=default_msg, loop=loop)
                try:
                    with progress_context(lazy_prog.update, message=default_msg):
                        if is_coro:
                            with WarningCatcher() as catcher, reactive.isolate():
                                res = await bound_method(*args, **kwargs)
                            merged = WarningCatcher.merge(catcher.messages, getattr(res, "warnings_", ()))
                        else:
                            res, merged = await asyncio.to_thread(_sync_worker, *args, **kwargs)
                    if holder:
                        holder[0].warnings = merged
                    return res
                except BaseException as exc:
                    if holder:
                        holder[0].last_error = exc
                    raise
                finally:
                    lazy_prog.close()

            ext_task = reactive.extended_task(_async_runner)
            bound_handle = BoundTask(ext_task)
            holder.append(bound_handle)
            instance.__dict__[cache_key] = bound_handle

        return instance.__dict__[cache_key]

    def _handle_completion(
        self,
        component: "Component[Any]",
        res: Any,
        *,
        warnings: Sequence[str] = (),
    ) -> None:
        """Processes declarative artifact registration, warning toasts, `on_success` callbacks, and completion toasts."""
        if res is None or component._state is None:
            return

        if self.notify_warning:
            for w_msg in WarningCatcher.merge(warnings, getattr(res, "warnings_", ())):
                component.notify_warning(w_msg)

        if self.register_to is not None:
            component._state.register_artifact(self.register_to, res, **self.register_kwargs)

        custom_msg: str | None = None
        if self.on_success is not None:
            cb: Callable[..., Any] = (
                getattr(component, self.on_success) if isinstance(self.on_success, str) else self.on_success
            )
            is_bound = getattr(cb, "__self__", None) is not None
            raw_fn = getattr(cb, "__func__", cb)
            param_cnt = _callable_param_count(raw_fn) - (1 if is_bound else 0)
            custom_msg = cb(res) if param_cnt == 1 else cb(component, res)

        if custom_msg:
            component.notify(custom_msg, type_="message")
        elif self.notify_success:
            if callable(self.notify_success):
                msg = str(self.notify_success(res) or "")
            elif isinstance(self.notify_success, str):
                msg = self.notify_success.format(
                    res=res,
                    name=getattr(res, "name", ""),
                    rows=len(res) if hasattr(res, "__len__") else 0,
                )
            else:
                msg = component._state.format_task_success(res, registry_name=self.register_to)
            if msg:
                component.notify(msg, type_="message")

    def bind(
        self,
        component: "Component[Any]",
        input: Inputs,
        output: Outputs,
        session: Session,
    ) -> None:
        """Binds action-button triggers, busy-state synchronization, and completion handlers for the task."""
        bound_task: BoundTask[Any] = getattr(component, self.name)
        if self.button_id:
            btn_id = self.button_id
            was_running = False

            @reactive.effect(priority=10)
            @reactive.event(input[btn_id])
            def _on_task_btn() -> None:
                bound_task.invoke()

            @reactive.effect(priority=10)
            def _sync_btn_busy_state() -> None:
                nonlocal was_running
                is_running = bound_task.status() == "running"
                if is_running or was_running:
                    was_running = is_running
                    shiny_ui.update_action_button(btn_id, disabled=is_running, session=session)

        if (
            self.register_to is not None
            or self.on_success is not None
            or bool(self.notify_success)
            or self.notify_warning
            or self.notify_error
        ):
            last_handled_invocation = -1

            @reactive.effect(priority=10)
            def _on_task_done() -> None:
                nonlocal last_handled_invocation
                status = bound_task.status()
                if status not in ("success", "error") or bound_task.invocation_count == last_handled_invocation:
                    return
                last_handled_invocation = bound_task.invocation_count
                if status == "error":
                    if self.notify_error:
                        try:
                            bound_task.result()
                        except Exception as exc:
                            with reactive.isolate():
                                component.notify_error(exc, context=self.display_label)
                    return
                res = bound_task.result()
                with reactive.isolate():
                    self._handle_completion(component, res, warnings=bound_task.warnings)
