"""Session state (`BaseSessionState`) and reactive artifact registries (`ArtifactRegistry`, `artifact_registry`, `Requirement`)."""

import inspect
from abc import ABC
from collections.abc import Callable, Sized
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Literal, Self, overload

import shiny.ui as shiny_ui
from htmltools import Tag, TagChild
from shiny import Inputs, Outputs, Session, reactive, render
from shiny.session import get_current_session

from shiny_kit.caching import BaseCache, default_cache
from shiny_kit.decorators import reactive_prop
from shiny_kit.ui import Icon


class ArtifactRegistry[T]:
    """Reactive, session-isolated store for named pipeline artifacts.

    Tracks a dictionary of named artifacts alongside an active selection key and a version counter,
    triggering PyShiny reactive invalidation automatically whenever artifacts are added, removed,
    or re-selected.
    """

    def __init__(
        self,
        label: str,
        *,
        icon: str | None = None,
        badge_class: str = "bg-primary",
        show_in_navbar: bool = True,
        namer: Callable[[T], str] | None = None,
        toast: Callable[[T], str] | str | None = None,
        on_register: Callable[..., None] | str | None = None,
        on_select: Callable[..., None] | str | None = None,
        on_remove: Callable[..., None] | str | None = None,
        owner_state: "BaseSessionState | None" = None,
    ) -> None:
        self.label = label
        self.icon = icon
        self.badge_class = badge_class
        self.show_in_navbar = show_in_navbar
        self._namer = namer
        self._toast = toast
        self._on_register = on_register
        self._on_select = on_select
        self._on_remove = on_remove
        self._owner_state = owner_state
        self._items: reactive.Value[dict[str, T]] = reactive.value({})
        self._active_key: reactive.Value[str | None] = reactive.value(None)
        self._version: reactive.Value[int] = reactive.value(0)

    @property
    def items(self) -> dict[str, T]:
        """Reactively returns the mapping of registered artifact names to instances."""
        return self._items.get()

    @property
    def names(self) -> list[str]:
        """Reactively returns the ordered list of registered artifact names."""
        return list(self._items.get().keys())

    @property
    def count(self) -> int:
        """Reactively returns the number of registered artifacts."""
        return len(self._items.get())

    @property
    def version(self) -> int:
        """Reactively returns the mutation counter for this registry."""
        return self._version.get()

    @property
    def active_key(self) -> str | None:
        """Reactively returns the name of the currently active artifact."""
        return self._active_key.get()

    @active_key.setter
    def active_key(self, key: str | None) -> None:
        self.select(key)

    @property
    def active(self) -> T | None:
        """Reactively returns the currently active artifact instance (or `None` if empty)."""
        key = self._active_key.get()
        current = self._items.get()
        if key is not None and key in current:
            return current[key]
        if current:
            return next(iter(current.values()))
        return None

    def get(self, name: str) -> T | None:
        """Reactively looks up a specific artifact by name."""
        return self._items.get().get(name)

    def resolve_name(self, item: T, explicit_name: str | None = None, *, sync_item_name: bool = False) -> str:
        """Resolves the registry key for `item` using `explicit_name`, `self._namer`, or common attributes."""
        if explicit_name and explicit_name.strip():
            clean = explicit_name.strip()
            if sync_item_name and hasattr(item, "name"):
                try:
                    item.name = clean  # ty: ignore[invalid-assignment]
                except Exception:
                    pass
            return clean
        if self._namer is not None:
            return str(self._namer(item)).strip()
        fallback = (
            getattr(item, "name", None)
            or getattr(item, "summary_label", None)
            or getattr(item, "display_name", None)
            or str(item)
        )
        return str(fallback).strip()

    def format_toast(self, item: T) -> str:
        """Formats a completion toast message for `item` using the registry's `toast` specification."""
        if self._toast is None:
            return ""
        if callable(self._toast):
            return str(self._toast(item) or "")
        return self._toast.format(
            item=item,
            res=item,
            name=getattr(item, "name", ""),
            rows=len(item) if isinstance(item, Sized) else 0,
        )

    @overload
    def register(self, name: str, item: T, *, set_active: bool = True, **kwargs: Any) -> T: ...

    @overload
    def register(self, item: T, *, name: str | None = None, set_active: bool = True, **kwargs: Any) -> T: ...

    def register(self, *args: Any, name: str | None = None, set_active: bool = True, **kwargs: Any) -> T:
        """Registers or updates a named artifact and triggers reactive dependents and `on_register` hooks."""
        if len(args) == 2:
            raw_name, item = str(args[0]), args[1]
            clean_name = self.resolve_name(item, explicit_name=raw_name, sync_item_name=False)
        elif len(args) == 1:
            item = args[0]
            clean_name = self.resolve_name(item, explicit_name=name, sync_item_name=True)
        else:
            raise TypeError(f"ArtifactRegistry.register() expected 1 or 2 positional arguments, got {len(args)}.")

        if not clean_name:
            raise ValueError(f"{self.label} name cannot be empty.")
        with reactive.isolate():
            updated = dict(self._items.get())
            updated[clean_name] = item
            active_now = self._active_key.get()
            ver_now = self._version.get()
        self._items.set(updated)
        if set_active or active_now is None:
            self._active_key.set(clean_name)
        self._version.set(ver_now + 1)
        self._invoke_state_hook(self._on_register, item, name=clean_name, **kwargs)
        return item

    def remove(self, name: str, **kwargs: Any) -> None:
        """Removes a named artifact from the registry and updates the active selection if needed."""
        with reactive.isolate():
            current = self._items.get()
            if name not in current:
                return
            removed_item = current[name]
            updated = {k: v for k, v in current.items() if k != name}
            active_now = self._active_key.get()
            ver_now = self._version.get()
        self._items.set(updated)
        if active_now == name:
            self._active_key.set(next(iter(updated.keys()), None))
        self._version.set(ver_now + 1)
        self._invoke_state_hook(self._on_remove, removed_item, name=name, **kwargs)

    def select(self, name: str | None, **kwargs: Any) -> None:
        """Sets the active artifact by name and invokes `on_select` hooks."""
        with reactive.isolate():
            current = self._items.get()
            if name is not None and name not in current:
                raise KeyError(f"'{name}' is not registered in {self.label}: {list(current.keys())}")
            selected_item = current.get(name) if name is not None else None
        self._active_key.set(name)
        self._invoke_state_hook(self._on_select, selected_item, name=name, **kwargs)

    def clear(self) -> None:
        """Removes all artifacts from the registry."""
        with reactive.isolate():
            ver_now = self._version.get()
        self._items.set({})
        self._active_key.set(None)
        self._version.set(ver_now + 1)

    @staticmethod
    @lru_cache(maxsize=256)
    def _inspect_hook_signature(
        fn: Callable[..., Any],
    ) -> tuple[frozenset[str], bool, int, bool]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        pos_count = sum(
            1 for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        return frozenset(sig.parameters.keys()), has_var_kw, pos_count, has_var_pos

    def _invoke_state_hook(self, hook: Callable[..., Any] | str | None, *args: Any, **kwargs: Any) -> Any:
        """Invokes a state hook (method name or callable), forwarding only accepted keyword arguments."""
        if hook is None or self._owner_state is None:
            return None
        fn: Callable[..., Any] = getattr(self._owner_state, hook) if isinstance(hook, str) else hook
        target = getattr(fn, "__func__", fn)
        param_names, has_var_kw, pos_count, has_var_pos = self._inspect_hook_signature(target)
        filtered_kwargs = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in param_names}

        if isinstance(hook, str) or getattr(fn, "__self__", None) is not None:
            adj_pos = max(pos_count - (1 if getattr(fn, "__func__", None) is not None else 0), 0)
            eff_args = args if has_var_pos else args[:adj_pos]
            return fn(*eff_args, **filtered_kwargs)
        return fn(self._owner_state, *args, **filtered_kwargs)


@dataclass(frozen=True, slots=True)
class Requirement:
    """Declarative prerequisite rule gating a `Tab` or `Component` on `BaseSessionState` resources.

    Args:
        registry: Name of the `ArtifactRegistry` attribute on `BaseSessionState`.
        min_count: Minimum number of registered items required to unlock the view.
        label: Human-readable description of the required resource.
        target_tab: Optional tab `id_` to jump to when the user clicks the unlock action button.
    """

    registry: str
    min_count: int = 1
    label: str = ""
    target_tab: str | None = None

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        return self.registry.title()

    def get_registry(self, state: "BaseSessionState") -> ArtifactRegistry[Any]:
        return state.get_registry(self.registry)

    def is_met(self, state: "BaseSessionState") -> bool:
        """Reactively checks whether this requirement is satisfied in `state`."""
        return self.get_registry(state).count >= self.min_count

    def status_text(self, state: "BaseSessionState") -> str:
        """Returns a formatted progress string (e.g. 'Formulations: 1 / 2 available')."""
        reg = self.get_registry(state)
        return f"{self.display_label}: {reg.count} / {self.min_count} available"


class artifact_registry[T]:
    """Class-level descriptor declaring an `ArtifactRegistry[T]` on a `BaseSessionState` subclass."""

    def __init__(
        self,
        label: str = "",
        *,
        icon: str | None = None,
        badge_class: str = "bg-primary",
        show_in_navbar: bool = True,
        namer: Callable[[T], str] | None = None,
        toast: Callable[[T], str] | str | None = None,
        on_register: Callable[..., None] | str | None = None,
        on_select: Callable[..., None] | str | None = None,
        on_remove: Callable[..., None] | str | None = None,
    ) -> None:
        self.label = label
        self.attr_name: str = ""
        self._config: dict[str, Any] = {
            "icon": icon,
            "badge_class": badge_class,
            "show_in_navbar": show_in_navbar,
            "namer": namer,
            "toast": toast,
            "on_register": on_register,
            "on_select": on_select,
            "on_remove": on_remove,
        }

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.attr_name = name
        if not self.label:
            self.label = name.replace("_", " ").title()

    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: "BaseSessionState", owner: type[Any]) -> ArtifactRegistry[T]: ...

    def __get__(self, instance: "BaseSessionState | None", owner: type[Any]) -> ArtifactRegistry[T] | Self:
        if instance is None:
            return self
        cache_key = f"_reg_{self.attr_name}"
        if cache_key not in instance.__dict__:
            instance.__dict__[cache_key] = ArtifactRegistry(
                self.label or self.attr_name.title(),
                owner_state=instance,
                **self._config,
            )
        return instance.__dict__[cache_key]


class BaseSessionState(ABC):  # noqa: B024
    """Abstract session state base class holding reactive `artifact_registry` stores, theme state, and cache.

    Subclasses declare domain registries as class-level `artifact_registry[T](...)` attributes.
    Ties state instances to `session.id` and auto-cleans via PyShiny's lifecycle hooks.

    Attributes:
        color_mode (Literal["light", "dark"]): Reactive UI color theme mode.
        user_id (str | None): Optional authenticated user identifier.
        preferences (dict[str, str]): Arbitrary per-session user preferences dictionary.
        root_session (Session | None): Root Shiny `Session` owning this state instance.
        navset_id (str): Top-level navbar DOM ID used by `navigate_to()`.
        cache (BaseCache): Two-tier `BaseCache` shared across the application.

    Examples:
        ```python
        from shiny_kit.state import BaseSessionState, artifact_registry

        class MySessionState(BaseSessionState):
            tables = artifact_registry[object]("Tables", icon="table")

        state = MySessionState()
        assert state.tables.count == 0
        ```
    """

    _registry: ClassVar[dict[str, "BaseSessionState"]] = {}
    _registry_specs: ClassVar[dict[str, artifact_registry[Any]]] = {}

    color_mode: reactive_prop[Literal["light", "dark"]] = reactive_prop("light")

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls is BaseSessionState:
            raise TypeError(
                "Can't instantiate abstract class BaseSessionState directly; "
                "subclass BaseSessionState to define application state."
            )
        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        specs: dict[str, artifact_registry[Any]] = {}
        for base in reversed(cls.__mro__):
            for attr_name, attr_val in base.__dict__.items():
                if isinstance(attr_val, artifact_registry):
                    specs[attr_name] = attr_val
        cls._registry_specs = specs

    @classmethod
    def from_session(cls, session: Session | None = None) -> Self:
        """Retrieves or initializes typed state for the given session."""
        sess = session or get_current_session()
        if sess is None:
            raise RuntimeError("No active Shiny session available.")

        root_sess = getattr(sess, "_root_session", sess)
        existing = cls._registry.get(root_sess.id)
        if isinstance(existing, cls):
            return existing

        instance = cls()
        instance.root_session = root_sess
        cls._registry[root_sess.id] = instance

        if existing is None:

            def _cleanup() -> None:
                cls._registry.pop(root_sess.id, None)

            root_sess.on_ended(_cleanup)

        return instance

    def __init__(self, *, cache: BaseCache | None = None, navset_id: str = "main_nav") -> None:
        self.user_id: str | None = None
        self.preferences: dict[str, str] = {}
        self.root_session: Session | None = None
        self.navset_id: str = navset_id
        self.cache: BaseCache = cache or default_cache
        for reg_name in self._registry_specs:
            _ = getattr(self, reg_name)

    @property
    def registries(self) -> dict[str, ArtifactRegistry[Any]]:
        """Returns the ordered mapping of declared registry names to bound `ArtifactRegistry` instances."""
        return {name: getattr(self, name) for name in self._registry_specs}

    def get_registry(self, name: str) -> ArtifactRegistry[Any]:
        """Retrieves a named `ArtifactRegistry` attribute on this state instance."""
        reg = getattr(self, name, None)
        if not isinstance(reg, ArtifactRegistry):
            raise AttributeError(f"{type(self).__name__} has no ArtifactRegistry named '{name}'.")
        return reg

    def register_artifact(
        self,
        registry_name: str,
        item: Any,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Registers `item` into the named `ArtifactRegistry`, triggering its `namer` and `on_register` hooks."""
        reg = self.get_registry(registry_name)
        return reg.register(item, name=name, set_active=True, **kwargs)

    def format_task_success(self, res: Any, *, registry_name: str | None = None) -> str:
        """Formats a default completion toast message for a task result `res`."""
        if registry_name and registry_name in self._registry_specs:
            msg = self.get_registry(registry_name).format_toast(res)
            if msg:
                return msg
        for reg in self.registries.values():
            if reg._toast is not None and any(res is v for v in reg.items.values()):
                return reg.format_toast(res)
        return ""

    def summary_counts(self) -> dict[str, int]:
        """Reactively returns `{registry_name: count}` across all declared registries."""
        return {name: reg.count for name, reg in self.registries.items()}

    def reset_all(self) -> None:
        """Clears all declared `ArtifactRegistry` stores and calls `on_reset()`."""
        for reg in self.registries.values():
            reg.clear()
        self.on_reset()

    def on_reset(self) -> None:  # noqa: B027
        """Optional hook called by `reset_all()` when clearing session state."""
        pass

    def render_navbar_badges(self) -> Tag:
        """Renders reactive status badges for all declared registries with `show_in_navbar=True`."""
        visible = [(name, reg) for name, reg in self.registries.items() if reg.show_in_navbar]
        badges: list[Any] = []
        for idx, (_, reg) in enumerate(visible):
            cnt = reg.count
            active_cls = reg.badge_class if cnt > 0 else "bg-secondary"
            margin_cls = "me-2 " if idx < len(visible) - 1 else ""
            badges.append(
                shiny_ui.span(
                    Icon.lucide(reg.icon, size=13) if reg.icon else None,
                    f"{reg.label}: {cnt}",
                    class_=f"badge {active_cls} {margin_cls}p-2",
                )
            )
        return shiny_ui.div(*badges, class_="d-flex align-items-center pe-2")

    def navigate_to(self, tab_id: str, *, navset_id: str | None = None) -> None:
        """Switches the active navigation panel in the root session."""
        target_navset = navset_id or self.navset_id
        sess = self.root_session or get_current_session()
        if sess is not None:
            shiny_ui.update_navset(target_navset, selected=tab_id, session=sess)

    @classmethod
    def navbar_controls(
        cls,
        *,
        color_mode_id: str = "color_mode",
        badges_id: str = "navbar_registry_badges",
        default_mode: Literal["light", "dark"] = "light",
    ) -> tuple[Any, ...]:
        """Returns `(nav_spacer(), nav_control(output_ui(badges_id)), nav_control(input_dark_mode(...)))` items."""
        return (
            shiny_ui.nav_spacer(),
            shiny_ui.nav_control(
                shiny_ui.output_ui(badges_id),
            ),
            shiny_ui.nav_control(
                shiny_ui.div(
                    shiny_ui.input_dark_mode(id=color_mode_id, mode=default_mode),
                    class_="d-flex align-items-center ps-1 pe-2",
                ),
            ),
        )

    def bind_navbar(
        self,
        input: Inputs,
        output: Outputs,
        *,
        color_mode_id: str = "color_mode",
        badges_id: str = "navbar_registry_badges",
    ) -> None:
        """Binds top-level navbar dark-mode synchronization and reactive registry status badges."""

        @reactive.effect
        def _sync_color_mode() -> None:
            try:
                mode_val = input[color_mode_id]()
            except Exception:
                mode_val = "light"
            self.color_mode = "dark" if mode_val == "dark" else "light"

        @render.ui
        def _render_badges() -> TagChild:
            return self.render_navbar_badges()

        output(id=badges_id)(_render_badges)
