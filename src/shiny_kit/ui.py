"""Stateless UI layout primitives (`Icon`, `EmptyState`, `KPICard`, `KPIRow`) and reactive `StagedList`."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import lucide
from htmltools import HTML, Tag, TagChild
from shiny import reactive
from shiny import ui as shiny_ui


class Icon:
    """Renders Lucide SVG icons wrapped in `htmltools.HTML` for PyShiny UI elements.

    Examples:
        ```python
        from shiny_kit.ui import Icon

        svg = Icon.lucide("database", size=16)
        ```
    """

    @classmethod
    def lucide(cls, name: str | None, size: int = 18, class_: str = "me-1", **kwargs: Any) -> HTML | None:
        """Render a Lucide SVG icon wrapped in `htmltools.HTML` for Shiny UI integration.

        Args:
            name: Lucide icon identifier (e.g. `"database"`, `"shield-check"`), or `None`.
            size: Width and height of the rendered SVG in pixels. Defaults to `18`.
            class_: CSS classes appended to `"lucide-icon"`. Defaults to `"me-1"`.
            **kwargs: Additional SVG attributes forwarded to `lucide._render_icon`.

        Returns:
            `htmltools.HTML` containing the `<svg>` markup, or `None` if `name` is `None`.

        Raises:
            ValueError: If `name` does not match a valid Lucide icon.
        """
        if name is None:
            return None

        icon_name = str(name).strip().lower().replace("_", "-")
        combined_class = f"lucide-icon {class_}".strip()
        icon_kwargs = {"class": combined_class, **kwargs}
        try:
            svg_str = lucide._render_icon(icon_name, size, **icon_kwargs)
        except (lucide.IconDoesNotExist, FileNotFoundError, AttributeError, ValueError) as err:
            raise ValueError(f"Icon '{name}' (normalized as '{icon_name}') does not exist in Lucide icon set.") from err

        if "xmlns=" not in svg_str.lower():
            svg_str = svg_str.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)

        return HTML(svg_str)


@dataclass(frozen=True, slots=True)
class EmptyState:
    """Declarative empty-state placeholder card with a Lucide icon, title, and muted description.

    Attributes:
        icon (str): Lucide icon name.
        title (str): Primary heading text.
        message (str): Secondary explanatory paragraph text.
        dashed (bool): Whether to render a dashed border instead of a solid card background.
        compact (bool): Whether to use tighter vertical padding.
        icon_size (int): Pixel size of the Lucide icon. Defaults to `36`.

    Examples:
        ```python
        from shiny_kit.ui import EmptyState

        placeholder = EmptyState(
            icon="database",
            title="No Dataset Loaded",
            message="Upload or select a benchmark cohort in the sidebar.",
        )
        tag = placeholder.ui()
        ```
    """

    icon: str
    title: str
    message: str
    dashed: bool = False
    compact: bool = False
    icon_size: int = 36

    def ui(self) -> Tag:
        """Renders the empty-state Bootstrap container.

        Returns:
            Bootstrap `div` tag containing the icon, title, and message.
        """
        pad_cls = "py-3 my-2" if self.compact else "py-5 my-3"
        border_cls = "border-dashed" if self.dashed else "bg-light rounded border"
        return shiny_ui.div(
            Icon.lucide(self.icon, size=self.icon_size, class_="text-muted mb-2"),
            shiny_ui.h6(self.title, class_="text-secondary mt-1 mb-1"),
            shiny_ui.p(
                self.message,
                class_="text-muted small mb-0",
                style="max-width: 560px; margin: 0 auto;",
            ),
            class_=f"text-center {pad_cls} px-3 {border_cls}",
        )

    __call__ = ui


@dataclass(frozen=True, slots=True)
class KPICard:
    """Declarative summary metric card inside a `KPIRow`.

    Attributes:
        label (str): Muted top label describing the metric.
        value (str): Bold primary value text.
        subtitle (str): Secondary caption beneath the primary value.
        col_width (int): Bootstrap 12-column grid span (`col-md-{col_width}`). Defaults to `3`.
        value_class (str): Additional CSS classes applied to the primary value element.
        subtitle_class (str): Additional CSS classes applied to the subtitle element.
        truncate_subtitle (bool): Whether to apply `text-truncate` and a hover `title` attribute to the subtitle.
        tooltip (str | None): Optional hover tooltip text for the subtitle.
    """

    label: str
    value: str
    subtitle: str = ""
    col_width: int = 3
    value_class: str = "fs-5"
    subtitle_class: str = "text-muted"
    truncate_subtitle: bool = False
    tooltip: str | None = None

    def ui(self) -> Tag:
        """Renders the responsive Bootstrap column and bordered metric card.

        Returns:
            Bootstrap `div` column containing the formatted KPI card.
        """
        sub_classes = f"small {self.subtitle_class}" + (" text-truncate" if self.truncate_subtitle else "")
        sub_kwargs: dict[str, Any] = {"class_": sub_classes}
        if self.tooltip or self.truncate_subtitle:
            sub_kwargs["title"] = self.tooltip or self.subtitle

        return shiny_ui.div(
            shiny_ui.div(
                shiny_ui.div(self.label, class_="text-muted small"),
                shiny_ui.div(self.value, class_=f"fw-bold {self.value_class}"),
                shiny_ui.div(self.subtitle, **sub_kwargs) if self.subtitle else None,
                class_="p-2 bg-light rounded border h-100",
            ),
            class_=f"col-md-{self.col_width}",
        )


@dataclass(frozen=True, slots=True)
class KPIRow:
    """Responsive Bootstrap row of `KPICard` summary metrics.

    Attributes:
        cards (tuple[KPICard, ...]): Tuple of `KPICard` specifications rendered across the row.
    """

    cards: tuple[KPICard, ...]

    def __init__(self, *cards: KPICard) -> None:
        """Initializes `KPIRow` with one or more `KPICard` items.

        Args:
            *cards: Variable number of `KPICard` instances.
        """
        object.__setattr__(self, "cards", tuple(cards))

    def ui(self) -> Tag:
        """Renders the `row g-2 mb-3` grid of KPI cards.

        Returns:
            Bootstrap `div.row` containing all rendered `KPICard` columns.
        """
        return shiny_ui.div(*(card.ui() for card in self.cards), class_="row g-2 mb-3")


class StagedList[T]:
    """Per-instance reactive list of staged items with UI badge and button helpers."""

    def __init__(self, initial: Iterable[T] = ()) -> None:
        self._val: reactive.Value[tuple[T, ...]] = reactive.value(tuple(initial))

    @property
    def items(self) -> list[T]:
        """Reactively returns the staged items as a list."""
        return list(self._val.get())

    @property
    def is_empty(self) -> bool:
        """Reactively returns True when no items are staged."""
        return len(self._val.get()) == 0

    def __len__(self) -> int:
        return len(self._val.get())

    def set(self, items: Iterable[T]) -> None:
        """Replaces the staged items."""
        self._val.set(tuple(items))

    def add(self, item: T, *, deduplicate: bool = False) -> None:
        """Appends `item` to the staged collection."""
        with reactive.isolate():
            cur = list(self._val.get())
        if deduplicate and item in cur:
            cur = [x for x in cur if x != item]
        cur.append(item)
        self._val.set(tuple(cur))

    def pop(self) -> T | None:
        """Removes and returns the last staged item, if any."""
        with reactive.isolate():
            cur = list(self._val.get())
        if not cur:
            return None
        removed = cur.pop()
        self._val.set(tuple(cur))
        return removed

    def remove_at(self, index: int) -> None:
        """Removes the item at `index` if in range."""
        with reactive.isolate():
            cur = list(self._val.get())
        if 0 <= index < len(cur):
            cur.pop(index)
            self._val.set(tuple(cur))

    def clear(self) -> None:
        """Clears all staged items."""
        self._val.set(())

    def render_badges(
        self,
        formatter: Callable[[int, T], TagChild],
        *,
        empty_text: str = "No items staged yet.",
        wrapper_class: str = "",
    ) -> TagChild:
        """Renders the staged items using `formatter(index, item)` or an empty caption."""
        cur = self.items
        if not cur:
            return shiny_ui.div(
                empty_text,
                class_="text-muted small fst-italic p-2 border rounded bg-body-tertiary",
            )
        return shiny_ui.div(
            *(formatter(idx, item) for idx, item in enumerate(cur, start=1)),
            class_=wrapper_class or None,
        )

    @staticmethod
    def action_buttons(
        add_id: str,
        *,
        add_label: str = "Add",
        pop_id: str | None = None,
        pop_label: str = "Remove Last",
        clear_id: str | None = None,
        clear_label: str = "Clear",
    ) -> Tag:
        """Renders a compact horizontal toolbar of Add / Remove Last / Clear action buttons."""
        btns: list[TagChild] = [
            shiny_ui.input_action_button(
                add_id,
                add_label,
                icon=Icon.lucide("plus", size=14),
                class_="btn-sm btn-outline-primary flex-fill",
            )
        ]
        if pop_id:
            btns.append(
                shiny_ui.input_action_button(
                    pop_id,
                    pop_label,
                    class_="btn-sm btn-outline-secondary",
                )
            )
        if clear_id:
            btns.append(
                shiny_ui.input_action_button(
                    clear_id,
                    clear_label,
                    icon=Icon.lucide("x", size=14),
                    class_="btn-sm btn-outline-danger",
                )
            )
        return shiny_ui.div(*btns, class_="d-flex gap-1 mt-1 mb-2")
