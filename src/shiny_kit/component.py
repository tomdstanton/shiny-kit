"""Core OOP PyShiny lifecycle (`Component`, `SubComponentProxy`, `subcomponent`, `Tab`, `ParentTab`)."""

from typing import Any, ClassVar, Literal, Self, get_args, overload

import shiny.ui as shiny_ui
from htmltools import HTML, Tag
from shiny import Inputs, Outputs, Session, module, reactive, render

from shiny_kit.decorators import BaseBinder
from shiny_kit.state import BaseSessionState, Requirement
from shiny_kit.ui import Icon


class _session_attr[T]:
    """Descriptor guarding session-bound attributes (`input`, `output`, `session`, `state`) until `server()` is bound."""

    def __init__(self) -> None:
        self.name: str = ""
        self.private_name: str = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.name = name
        self.private_name = f"_{name}"

    @overload
    def __get__(self, instance: None, owner: type[Any]) -> Self: ...

    @overload
    def __get__(self, instance: Any, owner: type[Any]) -> T: ...

    def __get__(self, instance: Any, owner: type[Any]) -> T | Self:
        if instance is None:
            return self
        val = getattr(instance, self.private_name, None)
        if val is None:
            raise RuntimeError(f"{instance.__class__.__name__}.{self.name} accessed before server() was bound.")
        return val

    def __set__(self, instance: Any, value: T | None) -> None:
        setattr(instance, self.private_name, value)


class Component[S: BaseSessionState]:
    """Base class for OOP-driven Shiny modules with integrated UI, Server, Subcomponents, and Resource Gating.

    Subclasses can bind a typed `BaseSessionState` subclass via PEP 695 type parameter `S` and `state_cls`,
    compose child `Component`s via `subcomponent(...)`, and declare pipeline prerequisites via `requires`.

    Attributes:
        state_cls (type[S]): Concrete `BaseSessionState` associated with this component.
        requires (ClassVar[tuple[Requirement, ...]]): Declarative prerequisite rules gating the component.
        module_id (str): Shiny module namespace ID assigned when `server(id_)` is bound.

    Examples:
        ```python
        from shiny import render, ui
        from shiny_kit.component import Component
        from shiny_kit.decorators import bind_output
        from shiny_kit.state import BaseSessionState

        class AppState(BaseSessionState):
            pass

        class CounterCard(Component[AppState]):
            @classmethod
            def _render_ui(cls):
                return ui.card(ui.output_text("label"))

            @bind_output(render.text)
            def label(self) -> str:
                return f"Theme: {self.state.color_mode}"
        ```
    """

    state_cls: type[S] = BaseSessionState  # ty: ignore[invalid-assignment]
    requires: ClassVar[tuple[Requirement, ...]] = ()

    _registered_binders: ClassVar[list[str]] = []
    _registered_subcomponents: ClassVar[list[str]] = []

    input: _session_attr[Inputs] = _session_attr()
    output: _session_attr[Outputs] = _session_attr()
    session: _session_attr[Session] = _session_attr()
    state: _session_attr[S] = _session_attr()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "state_cls" not in cls.__dict__:
            for base in getattr(cls, "__orig_bases__", ()):
                for arg in get_args(base):
                    if isinstance(arg, type) and issubclass(arg, BaseSessionState):
                        cls.state_cls = arg
                        break

        cls._registered_binders = list(cls._registered_binders)
        cls._registered_subcomponents = list(cls._registered_subcomponents)

        for name, attr in cls.__dict__.items():
            if isinstance(attr, BaseBinder) and name not in cls._registered_binders:
                cls._registered_binders.append(name)
            elif isinstance(attr, subcomponent) and name not in cls._registered_subcomponents:
                cls._registered_subcomponents.append(name)

    def __init__(self) -> None:
        self._input: Inputs | None = None
        self._output: Outputs | None = None
        self._session: Session | None = None
        self._state: S | None = None
        self.module_id: str = ""

    @property
    def has_session(self) -> bool:
        """Whether this component is currently bound to an active Shiny session."""
        return self._input is not None

    def val(
        self,
        id_: str,
        default: Any = None,
        *,
        isolate: bool = False,
        strip: bool = True,
    ) -> Any:
        """Safely reads `self.input[id_]()` with fallback `default`, optional reactive isolation, and string stripping."""
        if self._input is None:
            return default

        def _read() -> Any:
            if self._input is None or id_ not in self._input:
                return default
            raw = self._input[id_]()
            if raw is None:
                return default
            if isinstance(raw, str):
                cleaned = raw.strip() if strip else raw
                if not cleaned and default is not None:
                    return default
                return cleaned
            return raw

        if isolate:
            with reactive.isolate():
                return _read()
        return _read()

    def notify(
        self,
        message: str,
        *,
        type_: Literal["message", "warning", "error", "default"] = "message",
        duration: int | None = 5,
    ) -> None:
        """Displays a toast notification if running inside an active Shiny session."""
        shiny_ui.notification_show(message, type=type_, duration=duration)

    def notify_warning(self, message: str, *, duration: int | None = 8) -> None:
        """Displays a warning toast notification in the active Shiny session."""
        clean = str(message).strip()
        if clean:
            self.notify(clean, type_="warning", duration=duration)

    def notify_error(
        self,
        error: BaseException | str,
        *,
        context: str | None = None,
        duration: int | None = 8,
    ) -> None:
        """Formats and displays an error toast notification in the active Shiny session."""
        detail = str(error).strip() or error.__class__.__name__
        msg = f"{context}: {detail}" if context else detail
        self.notify(msg, type_="error", duration=duration)

    def is_unlocked(self) -> bool:
        """Reactively checks whether all declarative `requires` rules are satisfied."""
        if not self.requires or self._state is None:
            return True
        return all(r.is_met(self._state) for r in self.requires)

    def unmet_requirements(self) -> list[Requirement]:
        """Reactively returns the list of currently unmet `Requirement` rules."""
        if not self.requires or self._state is None:
            return []
        return [r for r in self.requires if not r.is_met(self._state)]

    @classmethod
    def ui(cls, id_: str, *args: Any, **kwargs: Any) -> Any:
        """Constructs and namespaces the UI tree for this component, adding a resource-gate container if `requires` is set."""

        @module.ui
        def _ui() -> Any:
            content = cls._render_ui(*args, **kwargs)
            if not cls.requires:
                return content
            return shiny_ui.TagList(
                shiny_ui.output_ui("_gate_callout"),
                shiny_ui.div(content, id=module.resolve_id("_tab_content_wrapper")),
            )

        return _ui(id_)

    @classmethod
    def _render_ui(cls, *args: Any, **kwargs: Any) -> Any:
        """Override to implement the component layout."""
        raise NotImplementedError

    def server(self, id_: str) -> Any:
        """Binds server logic, state lifecycle, declarative `BaseBinder`s, and subcomponents to a session."""
        self.module_id = id_

        @module.server
        def _server(input: Inputs, output: Outputs, session: Session) -> None:
            self._input = input
            self._output = output
            self._session = session
            self._state = self.state_cls.from_session(session)

            for name in self._registered_binders:
                binder: BaseBinder = getattr(self.__class__, name)
                binder.bind(self, input, output, session)

            for sub_name in self._registered_subcomponents:
                sub_desc: subcomponent[Any] = getattr(self.__class__, sub_name)
                child_inst: Component[Any] = getattr(self, sub_name)
                child_inst.server(sub_desc.child_id)

            if self.requires:
                self._bind_resource_gate(input, output, session)

            self._register_server()

        return _server(id_)

    def _bind_resource_gate(self, input: Inputs, output: Outputs, session: Session) -> None:
        """Binds the empty-state prerequisite callout card and navigation jump buttons."""

        @render.ui
        def _gate_callout() -> Tag | None:
            if self.is_unlocked() or self._state is None:
                return None

            items_ui: list[Any] = []
            for idx, r in enumerate(self.requires):
                met = r.is_met(self._state)
                badge_cls = "bg-success" if met else "bg-secondary"
                icon_name = "check" if met else "lock"
                row_children: list[Any] = [
                    shiny_ui.span(
                        Icon.lucide(icon_name, size=14),
                        r.status_text(self._state),
                        class_=f"badge {badge_cls} me-2 p-2",
                    )
                ]
                if not met and r.target_tab:
                    row_children.append(
                        shiny_ui.input_action_button(
                            f"_jump_req_{idx}",
                            f"Go to {r.display_label}",
                            icon=Icon.lucide("arrow-right", size=14),
                            class_="btn-sm btn-outline-primary",
                        )
                    )
                items_ui.append(shiny_ui.div(*row_children, class_="d-flex align-items-center mb-2"))

            wrapper_id = session.ns("_tab_content_wrapper")
            hide_css = shiny_ui.tags.style(f"#{wrapper_id} {{ display: none !important; }}")

            return shiny_ui.div(
                hide_css,
                shiny_ui.card(
                    shiny_ui.card_header(
                        Icon.lucide("lock", size=18),
                        " Prerequisite Resources Required",
                        class_="fw-bold text-secondary",
                    ),
                    shiny_ui.card_body(
                        shiny_ui.p(
                            "Complete the following pipeline steps to unlock this section:",
                            class_="text-muted mb-3",
                        ),
                        *items_ui,
                    ),
                    class_="border-secondary-subtle shadow-sm my-3",
                ),
            )

        output(id="_gate_callout")(_gate_callout)

        for idx, r in enumerate(self.requires):
            if r.target_tab:
                btn_id = f"_jump_req_{idx}"
                target = r.target_tab

                def _bind_jump(bid: str, dest: str) -> None:
                    @reactive.effect
                    @reactive.event(input[bid])
                    def _on_jump() -> None:
                        if self._state is not None:
                            self._state.navigate_to(dest)

                _bind_jump(btn_id, target)

    def _register_server(self) -> None:
        """Override this hook to append manual reactive logic or custom observers."""
        pass


class SubComponentProxy[C: "Component[Any]"]:
    """Class-level proxy returned when accessing a `subcomponent` descriptor on a Component class."""

    def __init__(self, comp_cls: type[C], child_id: str) -> None:
        self.comp_cls = comp_cls
        self.child_id = child_id

    def ui(self, *args: Any, **kwargs: Any) -> Any:
        """Renders the child component's UI namespaced under `self.child_id`."""
        return self.comp_cls.ui(self.child_id, *args, **kwargs)


class subcomponent[C: "Component[Any]"]:
    """Descriptor declaring a child `Component` auto-mounted and auto-bound by its parent `Component`."""

    def __init__(self, comp_cls: type[C], *, id_: str | None = None) -> None:
        self.comp_cls = comp_cls
        self.custom_id = id_
        self.attr_name: str = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self.attr_name = name

    @property
    def child_id(self) -> str:
        return self.custom_id or self.attr_name

    @overload
    def __get__(self, instance: None, owner: type[Any]) -> SubComponentProxy[C]: ...

    @overload
    def __get__(self, instance: Any, owner: type[Any]) -> C: ...

    def __get__(self, instance: Any, owner: type[Any]) -> C | SubComponentProxy[C]:
        if instance is None:
            return SubComponentProxy(self.comp_cls, self.child_id)
        cache_key = f"_subcomp_{self.attr_name}"
        if cache_key not in instance.__dict__:
            instance.__dict__[cache_key] = self.comp_cls()
        return instance.__dict__[cache_key]


class Tab[S: BaseSessionState](Component[S]):
    """Specialized `Component` representing a navigable tab in a tabbed layout.

    Extracts its display title from the first line of the subclass's docstring via `title()`
    and wraps `super().ui(id_)` inside a `shiny.ui.nav_panel`.

    Attributes:
        icon (HTML | None): Optional Lucide SVG icon rendered in the tab header.
    """

    icon: HTML | None = Icon.lucide(None)

    @classmethod
    def title(cls) -> str:
        """Derives title from the class docstring without inheriting parent docstrings."""
        raw_doc = cls.__dict__.get("__doc__")
        if raw_doc:
            summary = raw_doc.strip().splitlines()[0].strip().rstrip(".")
            if summary:
                return summary
        return cls.__name__.removesuffix("Tab")

    @classmethod
    def ui(cls, id_: str, *args: Any, **kwargs: Any) -> shiny_ui._navs.NavPanel:
        """Wraps the component UI in a Shiny `NavPanel`."""
        return shiny_ui.nav_panel(cls.title(), super().ui(id_, *args, **kwargs), icon=cls.icon, value=id_)


class ParentTab[S: BaseSessionState](Tab[S]):
    """Container tab hosting child `Tab` modules inside a card navigation panel.

    Attributes:
        children (ClassVar[list[type[Tab[Any]]]]): Ordered list of child `Tab` classes to mount.
    """

    children: ClassVar[list[type[Tab[Any]]]] = []

    def __init__(self) -> None:
        super().__init__()
        self._child_instances: list[Tab[Any]] = []

    @classmethod
    def _render_ui(cls) -> Any:
        navs = [child.ui(child.__name__) for child in cls.children]
        return shiny_ui.navset_card_underline(*navs, id="_sub_nav")

    def _register_server(self) -> None:
        self._child_instances = [child_cls() for child_cls in self.children]
        for child_instance in self._child_instances:
            child_instance.server(child_instance.__class__.__name__)
