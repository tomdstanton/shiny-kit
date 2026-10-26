"""Core OOP PyShiny lifecycle (`Component`, `SubComponentProxy`, `subcomponent`, `Tab`, `ParentTab`, `BaseHeroTab`)."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self, get_args, overload

import shiny.ui as shiny_ui
from htmltools import HTML, Tag, TagChild
from shiny import Inputs, Outputs, Session, module, reactive, render

from shiny_kit.decorators import BaseBinder
from shiny_kit.state import BaseSessionState, Requirement
from shiny_kit.ui import Icon
from shiny_kit.widgets import DocstringParser


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
        self._version_tracker: dict[str, Any] = {}

    def has_changed(self, key: str, version: Any, *, initial: Any = 0) -> bool:
        """Tracks `version` under `key` and returns `True` when `version` differs from the previously recorded value."""
        prev = self._version_tracker.get(key, initial)
        if version != prev:
            self._version_tracker[key] = version
            return True
        return False

    def sync_version(self, key: str, version: Any) -> None:
        """Updates the recorded version for `key` without reporting a change."""
        self._version_tracker[key] = version

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
        return DocstringParser.display_label(cls, strip_suffix="Tab", inherit_doc=False)

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


@dataclass(frozen=True, slots=True)
class HeroStatSpec:
    """Static specification for a metric tile in the `.sv-stat-strip` hero strip.

    Attributes:
        value (str): Highlighted headline metric value (rendered with gradient text).
        label (str): Descriptive caption displayed beneath `value`.
    """

    value: str
    label: str


@dataclass(frozen=True, slots=True)
class WorkflowStepSpec:
    """Static specification for a pillar/workflow card below the Hero section.

    Attributes:
        step_label (str): Card heading for the pipeline step.
        description (str): Body copy describing the step's capabilities.
        icon (str): Lucide icon name rendered in the card header. Defaults to `"sparkles"`.
        badge (str): Optional uppercase pill badge text rendered at the top-right of the card.
        metrics (tuple[str, ...]): Optional metric pills displayed inside the card footer.
        target_tab (str | None): Optional target navbar tab ID associated with the step.
    """

    step_label: str
    description: str
    icon: str = "sparkles"
    badge: str = ""
    metrics: tuple[str, ...] = ()
    target_tab: str | None = None


@dataclass(frozen=True, slots=True)
class HeroConfig:
    """Configuration contract for a `BaseHeroTab` landing view.

    Attributes:
        badge_text (str): Eyebrow pill text displayed above the headline.
        headline (str): Main gradient headline text.
        subtitle (str): Lead paragraph text beneath the headline.
        primary_cta_label (str): Label for the primary action button.
        secondary_cta_label (str): Label for the secondary demo-loading button.
        preview_title (str): Header title of the glassmorphic live preview card.
        workflow_steps (tuple[WorkflowStepSpec, ...]): Sequence of workflow cards rendered below the fold.
        stats (tuple[HeroStatSpec, ...]): Sequence of metrics rendered in the `.sv-stat-strip` bar.
    """

    badge_text: str
    headline: str
    subtitle: str
    primary_cta_label: str
    secondary_cta_label: str
    preview_title: str
    workflow_steps: tuple[WorkflowStepSpec, ...]
    stats: tuple[HeroStatSpec, ...] = ()
    pipeline_heading: str = "Workflow Pipeline"
    pipeline_subtitle: str = "Explore each stage of the workflow below."


@dataclass(frozen=True, slots=True)
class HomeTabEvents:
    """Read-only reactive event streams emitted by `BaseHeroTab.server()`.

    Attributes:
        launch_clicked (Callable[[], int]): Reactive getter returning the click count of the primary launch button.
        demo_clicked (Callable[[], int]): Reactive getter returning the click count of the demo cohort button.
    """

    launch_clicked: Callable[[], int]
    demo_clicked: Callable[[], int]


HERO_CSS = """
body {
    overflow-x: clip;
}

:root,
[data-bs-theme="light"] {
    --sv-gradient: linear-gradient(135deg, #7c3aed 0%, #6366f1 52%, #db2777 100%);
    --sv-glow: rgba(139, 92, 246, 0.16);
    --sv-border: rgba(139, 92, 246, 0.30);
    --sv-surface-tint: linear-gradient(145deg, rgba(124, 58, 237, 0.09), rgba(219, 39, 119, 0.05));
    --sv-accent: #7c3aed;
    --sv-accent-soft: rgba(139, 92, 246, 0.12);
    --sv-card-bg: rgba(255, 255, 255, 0.86);
    --sv-stat-bg: rgba(15, 23, 42, 0.035);
    --bs-primary: #7c3aed;
    --bs-primary-rgb: 124, 58, 237;
    --bs-link-color: #7c3aed;
    --bs-link-hover-color: #6d28d9;
}

[data-bs-theme="dark"] {
    --sv-gradient: linear-gradient(135deg, #a78bfa 0%, #818cf8 52%, #f472b6 100%);
    --sv-glow: rgba(167, 139, 250, 0.20);
    --sv-border: rgba(167, 139, 250, 0.32);
    --sv-surface-tint: linear-gradient(145deg, rgba(124, 58, 237, 0.16), rgba(219, 39, 119, 0.10));
    --sv-accent: #c084fc;
    --sv-accent-soft: rgba(168, 85, 247, 0.16);
    --sv-card-bg: rgba(15, 23, 42, 0.78);
    --sv-stat-bg: rgba(15, 23, 42, 0.58);
    --bs-body-bg: #0b1120;
    --bs-tertiary-bg: #111827;
    --bs-primary: #8b5cf6;
    --bs-primary-rgb: 139, 92, 246;
}

[data-bs-theme="dark"] .bg-light {
    background-color: var(--bs-tertiary-bg) !important;
    color: var(--bs-body-color) !important;
}

[data-bs-theme="dark"] .text-dark {
    color: var(--bs-body-color) !important;
}

.navbar-brand .sv-brand-icon svg {
    color: var(--sv-accent);
}

.sv-hero {
    position: relative;
    padding: 2.4rem 2.2rem;
    margin-top: 0.75rem;
    margin-bottom: 1.5rem;
    border-radius: 1rem;
    background:
        radial-gradient(circle at 85% 15%, var(--sv-glow), transparent 55%),
        var(--sv-surface-tint);
    border: 1px solid var(--sv-border);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    container-type: inline-size;
}

.sv-hero-eyebrow,
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.32rem 0.78rem;
    margin-bottom: 0.9rem;
    border-radius: 999px;
    background: var(--sv-accent-soft);
    color: var(--sv-accent);
    border: 1px solid var(--sv-border);
}

.hero-headline {
    background: var(--sv-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.15;
}

.hero-cta-group {
    display: inline-grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
}

.hero-cta-group > .btn {
    width: 100%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

@container (min-width: 1120px) {
    .hero-cta-group {
        grid-template-columns: repeat(2, 1fr);
    }
}

.sv-btn-primary.btn {
    color: #ffffff !important;
    background: var(--sv-gradient) !important;
    border: none !important;
    border-radius: 0.6rem;
    font-weight: 600;
    box-shadow: 0 6px 18px -5px var(--sv-glow);
    transition: transform 0.16s ease, box-shadow 0.16s ease, opacity 0.16s ease;
}

.sv-btn-primary.btn:hover {
    transform: translateY(-1px);
    opacity: 0.95;
}

.sv-btn-secondary.btn {
    color: var(--bs-body-color) !important;
    background: rgba(148, 163, 184, 0.10) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 0.6rem;
    font-weight: 600;
    transition: transform 0.16s ease, border-color 0.16s ease, background-color 0.16s ease;
}

.sv-btn-secondary.btn:hover {
    transform: translateY(-1px);
    border-color: var(--sv-accent) !important;
    background: var(--sv-accent-soft) !important;
}

.hero-glass-card {
    background: var(--sv-card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--sv-border);
    border-radius: 0.9rem;
    box-shadow: 0 14px 34px -12px var(--sv-glow);
}

.sv-stat-strip {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.75rem;
    margin-top: 1.75rem;
    padding-top: 1.35rem;
    border-top: 1px solid rgba(148, 163, 184, 0.20);
}

@media (min-width: 640px) {
    .sv-stat-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (min-width: 992px) {
    .sv-stat-strip {
        grid-template-columns: repeat(4, minmax(0, 1fr));
    }
}

.sv-stat-item {
    padding: 0.7rem 0.9rem;
    border-radius: 0.65rem;
    background: var(--sv-stat-bg);
    border: 1px solid rgba(148, 163, 184, 0.18);
}

.sv-stat-value {
    display: block;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: var(--sv-gradient);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: transparent;
}

.sv-stat-label {
    display: block;
    font-size: 0.76rem;
    opacity: 0.84;
    margin-top: 0.15rem;
    line-height: 1.4;
}

.sv-pillar-card,
.hero-workflow-card {
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 1.45rem 1.4rem;
    border-radius: 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: var(--sv-card-bg);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    overflow: hidden;
}

.sv-pillar-card::before,
.hero-workflow-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--sv-gradient);
}

.sv-pillar-card:hover,
.hero-workflow-card:hover {
    transform: translateY(-3px);
    border-color: var(--sv-border);
    box-shadow: 0 14px 32px -12px var(--sv-glow);
}

.sv-pillar-badge {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.22rem 0.58rem;
    border-radius: 999px;
    background: var(--sv-accent-soft);
    color: var(--sv-accent);
    border: 1px solid var(--sv-border);
}

.sv-pillar-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 1rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(148, 163, 184, 0.16);
}

.sv-metric-pill {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.18rem 0.54rem;
    border-radius: 0.4rem;
    background: rgba(148, 163, 184, 0.10);
    border: 1px solid rgba(148, 163, 184, 0.18);
}
"""


class BaseHeroTab[S: BaseSessionState](Tab[S]):
    """Reusable glassmorphic Hero & Workflow landing `Tab` component.

    Renders a two-column `.sv-hero` banner with gradient headline, primary/secondary CTA buttons,
    an interactive live preview widget card, `.sv-stat-strip` metric tiles, and `.sv-pillar-card`
    pipeline cards below the fold.
    """

    icon: HTML | None = Icon.lucide("house")
    config: ClassVar[HeroConfig]
    preview_icon: ClassVar[str] = "chart-network"
    preview_output_id: ClassVar[str] = "hero_preview"
    preview_height: ClassVar[str] = "320px"
    pipeline_heading: ClassVar[str | None] = None
    pipeline_subtitle: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        config: HeroConfig | None = None,
        color_mode: Callable[[], str] | None = None,
    ) -> None:
        super().__init__()
        self.instance_config: HeroConfig = config if config is not None else self.config
        self._color_mode_fn = color_mode

    @classmethod
    def render_preview_controls_ui(cls) -> TagChild:
        """Optional header controls rendered on the right side of the live preview card."""
        return None

    @classmethod
    def render_preview_widget_ui(cls) -> TagChild:
        """Renders the output widget container inside the glassmorphic preview card."""
        shinywidgets = importlib.import_module("shinywidgets")
        return shinywidgets.output_widget(
            cls.preview_output_id,
            width="100%",
            height=cls.preview_height,
            fill=True,
            fillable=True,
        )

    @classmethod
    def _render_ui(cls) -> TagChild:
        cfg = cls.config
        return shiny_ui.div(
            shiny_ui.head_content(shiny_ui.tags.style(HERO_CSS)),
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.layout_columns(
                        shiny_ui.div(
                            shiny_ui.h1(
                                cfg.headline,
                                class_="display-5 fw-bold hero-headline mb-3",
                            ),
                            shiny_ui.p(
                                cfg.subtitle,
                                class_="lead text-body-secondary mb-4",
                            ),
                            shiny_ui.div(
                                shiny_ui.input_action_button(
                                    "btn_launch",
                                    cfg.primary_cta_label,
                                    icon=Icon.lucide("rocket", size=16),
                                    class_="sv-btn-primary btn-lg px-4",
                                ),
                                shiny_ui.input_action_button(
                                    "btn_load_demo",
                                    cfg.secondary_cta_label,
                                    icon=Icon.lucide("play", size=16),
                                    class_="sv-btn-secondary btn-lg px-4",
                                ),
                                class_="hero-cta-group",
                            ),
                        ),
                        shiny_ui.div(
                            shiny_ui.div(
                                shiny_ui.div(
                                    Icon.lucide(cls.preview_icon, size=16, class_="text-primary me-2"),
                                    shiny_ui.span(cfg.preview_title, class_="fw-semibold small"),
                                    class_="d-flex align-items-center",
                                ),
                                shiny_ui.div(
                                    cls.render_preview_controls_ui(),
                                    class_="d-flex align-items-center gap-3",
                                ),
                                class_="d-flex flex-wrap justify-content-between align-items-center pb-2 mb-2 border-bottom",
                            ),
                            cls.render_preview_widget_ui(),
                            class_="hero-glass-card p-3 p-md-4",
                        ),
                        col_widths={"sm": (12, 12), "lg": (6, 6)},
                        class_="align-items-center g-4",
                        fill=False,
                    ),
                    shiny_ui.div(
                        *[
                            shiny_ui.div(
                                shiny_ui.span(stat.value, class_="sv-stat-value"),
                                shiny_ui.span(stat.label, class_="sv-stat-label"),
                                class_="sv-stat-item",
                            )
                            for stat in cfg.stats
                        ],
                        class_="sv-stat-strip",
                    )
                    if cfg.stats
                    else None,
                    class_="sv-hero",
                ),
                shiny_ui.div(
                    shiny_ui.div(
                        shiny_ui.h4(cls.pipeline_heading or cfg.pipeline_heading, class_="fw-bold mb-1"),
                        shiny_ui.p(
                            cls.pipeline_subtitle or cfg.pipeline_subtitle,
                            class_="text-body-secondary mb-4",
                        ),
                    ),
                    shiny_ui.layout_column_wrap(
                        *[
                            shiny_ui.div(
                                shiny_ui.div(
                                    shiny_ui.div(
                                        shiny_ui.span(
                                            Icon.lucide(step.icon, size=18),
                                            style="color: var(--sv-accent); background: var(--sv-accent-soft);",
                                            class_="p-2 rounded-3 d-inline-flex align-items-center justify-content-center",
                                        ),
                                        shiny_ui.span(
                                            step.badge,
                                            class_="sv-pillar-badge",
                                        )
                                        if step.badge
                                        else None,
                                        class_="d-flex justify-content-between align-items-center mb-3",
                                    ),
                                    shiny_ui.h6(step.step_label, class_="fw-bold mb-2"),
                                    shiny_ui.p(step.description, class_="text-body-secondary small mb-0"),
                                ),
                                shiny_ui.div(
                                    *[shiny_ui.span(m, class_="sv-metric-pill") for m in step.metrics],
                                    class_="sv-pillar-metrics",
                                )
                                if step.metrics
                                else None,
                                class_="sv-pillar-card hero-workflow-card h-100",
                            )
                            for step in cfg.workflow_steps
                        ],
                        width=1 / 3,
                        class_="g-4",
                        fill=False,
                    ),
                    class_="py-3",
                ),
                class_="container py-3",
            ),
        )

    def _resolve_active_theme(self) -> Literal["light", "dark"]:
        if self._color_mode_fn is not None:
            try:
                mode_val = self._color_mode_fn()
                return "dark" if mode_val == "dark" else "light"
            except Exception:
                pass
        return "dark" if self.state.color_mode == "dark" else "light"

    def server(
        self,
        id_: str,
        *,
        color_mode: Callable[[], str] | None = None,
        **_: Any,
    ) -> HomeTabEvents:
        """Binds server logic for the Hero tab and returns `HomeTabEvents` streams."""
        if color_mode is not None:
            self._color_mode_fn = color_mode
        super().server(id_)

        def _launch_clicked() -> int:
            return int(self.val("btn_launch", default=0) or 0)

        def _demo_clicked() -> int:
            return int(self.val("btn_load_demo", default=0) or 0)

        return HomeTabEvents(
            launch_clicked=_launch_clicked,
            demo_clicked=_demo_clicked,
        )
