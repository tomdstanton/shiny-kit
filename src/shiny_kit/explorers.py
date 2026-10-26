"""High-level domain explorer components (`ArtifactSwitcher`, `TableExplorer`, `BasePlotterExplorer`, `WidgetPlotterExplorer`)."""

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from htmltools import TagChild
from shiny import module, render, req
from shiny import ui as shiny_ui

from shiny_kit.component import Component
from shiny_kit.decorators import (
    bind_datagrid,
    bind_download,
    bind_effect,
    bind_output,
    bind_task,
    bind_widget,
    reactive_calc,
)
from shiny_kit.state import ArtifactRegistry
from shiny_kit.ui import EmptyState, Icon
from shiny_kit.widgets import PlotterWidgetSelector, WidgetSelector


class ArtifactSwitcher[T](Component[Any]):
    """Reusable selector, summary badge bar, deletion control, and file I/O component for an `ArtifactRegistry`.

    Attributes:
        registry_kind (ClassVar[str]): Name of the `ArtifactRegistry` attribute on `self.state`.
        label (ClassVar[str]): Display label for the artifact selector dropdown.
        show_count_in_label (ClassVar[bool]): Whether to append `(count)` to the selector label.
        enable_delete (ClassVar[bool]): Whether to render a delete button for the active artifact.
        enable_clear (ClassVar[bool]): Whether to render a `"Clear All"` button.
        enable_download (ClassVar[bool]): Whether to render an artifact download button.
        enable_upload (ClassVar[bool]): Whether to render an artifact file upload control.
        file_extension (ClassVar[str]): Default file extension appended to downloaded artifacts.
        upload_accept (ClassVar[Sequence[str]]): Accepted file extensions for the upload control.
        download_button_label (ClassVar[str]): Label for the download button.
        upload_label (ClassVar[str]): Heading label above the file upload input.
        upload_placeholder (ClassVar[str]): Placeholder text inside the file upload input.
    """

    registry_kind: ClassVar[str] = "artifacts"
    label: ClassVar[str] = "Active Artifact"
    show_count_in_label: ClassVar[bool] = True
    show_select_label: ClassVar[bool] = True
    select_width: ClassVar[str | None] = None
    inline_layout: ClassVar[bool] = False
    enable_delete: ClassVar[bool] = True
    delete_button_label: ClassVar[str] = ""
    enable_clear: ClassVar[bool] = False
    enable_download: ClassVar[bool] = False
    enable_upload: ClassVar[bool] = False
    file_extension: ClassVar[str] = ".pkl"
    upload_accept: ClassVar[Sequence[str]] = (".pkl",)
    download_button_label: ClassVar[str] = "Save"
    upload_label: ClassVar[str] = "Load Saved"
    upload_placeholder: ClassVar[str] = "Select file..."
    card_wrapped: ClassVar[bool] = False
    show_bottom_hr: ClassVar[bool] = True

    @property
    def registry(self) -> ArtifactRegistry[T]:
        """Returns the target `ArtifactRegistry` on `self.state`."""
        return self.state.get_registry(self.registry_kind)

    def active_for_download(self) -> T | None:
        """Returns the artifact instance to serialize when downloading (defaults to `self.registry.active`)."""
        return self.registry.active

    @classmethod
    def _render_ui(cls) -> TagChild:
        if cls.enable_upload:
            return shiny_ui.div(
                shiny_ui.output_ui("switcher_body_ui"),
                shiny_ui.div(
                    shiny_ui.p(
                        Icon.lucide("folder-open", size=14, class_="me-1"),
                        shiny_ui.strong(cls.upload_label),
                        class_="small mb-1 d-flex align-items-center",
                    ),
                    shiny_ui.input_file(
                        "artifact_file_upload",
                        label=None,
                        accept=list(cls.upload_accept),
                        multiple=False,
                        placeholder=cls.upload_placeholder,
                    ),
                    class_="mt-2",
                ),
            )
        return shiny_ui.output_ui("switcher_body_ui")

    def empty_ui(self) -> TagChild:
        """Rendered when the registry has 0 items."""
        return shiny_ui.div()

    def choice_label(self, name: str, item: T) -> str:
        """Formats the dropdown label for `item`."""
        return name

    def summary_ui(self, item: T) -> TagChild:
        """Optional status/badge block rendered below the dropdown."""
        return None

    def extra_buttons_ui(self, item: T) -> Sequence[TagChild]:
        """Optional extra action buttons placed alongside the Save/Delete controls."""
        return ()

    def load_artifact_from_path(self, path: Path, filename: str) -> T:
        """Override when `enable_upload = True` to deserialize an uploaded artifact file."""
        raise NotImplementedError

    def serialize_artifact(self, item: T) -> bytes | None:
        """Serializes `item` into bytes for download. Defaults to calling `item.save()` if defined."""
        if hasattr(item, "save") and callable(item.save):
            try:
                raw = item.save(path=None)
            except TypeError:
                raw = item.save()
            if isinstance(raw, bytes):
                return raw
        return None

    def download_filename(self, item: T | None) -> str:
        """Returns a sanitized filename with `self.file_extension` for the active artifact."""
        ext = self.file_extension if self.file_extension.startswith(".") else f".{self.file_extension}"
        if item is None:
            return f"{self.registry_kind.rstrip('s')}{ext}"
        raw_name = (
            getattr(item, "name", None) or getattr(item, "display_name", None) or self.registry.active_key or "artifact"
        )
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(raw_name)).strip("_")
        return f"{safe or 'artifact'}{ext}"

    @bind_output(render.ui)
    def switcher_body_ui(self) -> TagChild:
        reg = self.registry
        names = reg.names
        if not names:
            return self.empty_ui()

        active_key = reg.active_key or names[0]
        active_item = reg.get(active_key)
        select_label = (
            (f"{self.label} ({len(names)})" if self.show_count_in_label else self.label)
            if self.show_select_label
            else None
        )
        choices = {n: self.choice_label(n, reg.items[n]) for n in names}

        action_btns: list[TagChild] = []
        if active_item is not None:
            action_btns.extend(self.extra_buttons_ui(active_item))

        if self.enable_download:
            action_btns.append(
                shiny_ui.download_button(
                    module.resolve_id("download_artifact"),
                    self.download_button_label,
                    icon=Icon.lucide("download", size=14),
                    class_="btn-sm btn-outline-primary flex-grow-1",
                )
            )
        if self.enable_delete:
            action_btns.append(
                shiny_ui.input_action_button(
                    "btn_delete_artifact",
                    self.delete_button_label,
                    icon=Icon.lucide("trash-2", size=14),
                    class_="btn-sm btn-outline-danger",
                )
            )
        if self.enable_clear:
            action_btns.append(
                shiny_ui.input_action_button(
                    "btn_clear_artifacts",
                    "Clear All",
                    class_="btn-sm btn-outline-secondary",
                )
            )

        select_node = shiny_ui.input_select(
            "active_artifact_select",
            select_label,
            choices=choices,
            selected=active_key,
            width=self.select_width,
        )

        body_children: list[TagChild] = []
        if self.inline_layout:
            body_children.append(
                shiny_ui.div(
                    select_node,
                    *action_btns,
                    class_="d-flex flex-wrap align-items-center gap-2 mb-2",
                )
            )
            if active_item is not None:
                summary = self.summary_ui(active_item)
                if summary is not None:
                    body_children.append(summary)
        else:
            body_children.append(select_node)
            if active_item is not None:
                summary = self.summary_ui(active_item)
                if summary is not None:
                    body_children.append(summary)
            if action_btns:
                body_children.append(shiny_ui.div(*action_btns, class_="d-flex flex-wrap gap-2 mb-2"))

        if self.show_bottom_hr and not self.card_wrapped:
            body_children.append(shiny_ui.hr(class_="my-2"))

        if self.card_wrapped:
            return shiny_ui.card(
                shiny_ui.card_body(*body_children, class_="p-2"),
                class_="mb-2 border-primary-subtle shadow-sm",
            )
        return shiny_ui.div(*body_children)

    @bind_effect("active_artifact_select")
    def _on_select_artifact(self) -> None:
        selected = self.val("active_artifact_select")
        reg = self.registry
        if selected and selected in reg.items and selected != reg.active_key:
            reg.select(str(selected))

    @bind_effect("btn_delete_artifact")
    def _on_delete_artifact(self) -> None:
        reg = self.registry
        active_key = reg.active_key
        if active_key:
            reg.remove(active_key)
            singular = self.registry_kind.rstrip("s")
            self.notify(f"Removed {singular} '{active_key}'.", type_="message")

    @bind_effect("btn_clear_artifacts")
    def _on_clear_artifacts(self) -> None:
        self.registry.clear()
        self.notify(f"Cleared all {self.registry_kind}.", type_="message")

    def _artifact_download_filename(self) -> str:
        return self.download_filename(self.active_for_download())

    @bind_download("download_artifact", filename=_artifact_download_filename)
    def download_artifact(self) -> Any:
        item = self.active_for_download()
        if item is not None:
            raw = self.serialize_artifact(item)
            if raw is not None:
                yield raw

    @bind_effect("artifact_file_upload")
    def _on_upload_artifact_file(self) -> None:
        if not self.enable_upload:
            return
        file_info = self.val("artifact_file_upload", strip=False)
        if not file_info or not isinstance(file_info, list):
            return
        entry = file_info[0]
        datapath = str(entry.get("datapath", ""))
        filename = str(entry.get("name", f"uploaded{self.file_extension}"))
        if datapath:
            self.load_artifact_task.invoke(datapath, filename)

    @bind_task(on_success="_on_artifact_loaded")
    def load_artifact_task(self, datapath: str, filename: str) -> T:
        return self.load_artifact_from_path(Path(datapath), filename)

    def _on_artifact_loaded(self, item: T) -> str:
        self.state.register_artifact(self.registry_kind, item)
        label = getattr(item, "name", None) or getattr(item, "display_name", None) or self.registry.active_key
        return f"Loaded '{label}'."


class TableExplorer[M](Component[Any]):
    """Generic multi-view DataFrame and CSV export explorer.

    Subclasses implement `active_subject()`, `table_views(subject)`, and `resolve_table(subject, mode)`
    to provide a reactive radio-button view switcher, optional status badges, a filterable
    `render.DataGrid`, and one-click CSV export.

    Attributes:
        empty_icon (ClassVar[str]): Lucide icon name shown when `active_subject()` is `None`.
        empty_title (ClassVar[str]): Empty-state heading shown when `active_subject()` is `None`.
        empty_message (ClassVar[str]): Empty-state description shown when `active_subject()` is `None`.
        csv_prefix (ClassVar[str]): Filename prefix for exported `.csv` files.
        grid_height (ClassVar[str]): CSS height of the rendered DataGrid.
    """

    empty_icon: ClassVar[str] = "table"
    empty_title: ClassVar[str] = "No Data Available"
    empty_message: ClassVar[str] = "Fit or select a model in the sidebar to inspect tables."
    csv_prefix: ClassVar[str] = "export"
    grid_height: ClassVar[str] = "520px"
    grid_output_id: ClassVar[str] = "table_datagrid"

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.div(
            shiny_ui.output_ui("explorer_toolbar_ui"),
            shiny_ui.output_ui("explorer_body_ui"),
        )

    def active_subject(self) -> M | None:
        """Returns the active domain object backing the table views (or None if unavailable)."""
        raise NotImplementedError

    def table_views(self, subject: M) -> dict[str, str]:
        """Returns `{view_key: label}` choices available for `subject`."""
        raise NotImplementedError

    def resolve_table(self, subject: M, mode: str) -> Any | None:
        """Returns the tabular object for `subject` and `mode`."""
        raise NotImplementedError

    def toolbar_badges_ui(self, subject: M) -> Sequence[TagChild]:
        """Optional status badges displayed on the left of the table toolbar."""
        return ()

    def resolve_selected_mode(self, subject: M, choices: dict[str, str], current_mode: str | None) -> str:
        """Selects the active radio button mode given `choices` and the previous `current_mode`."""
        if current_mode and current_mode in choices:
            return current_mode
        return next(iter(choices.keys()), "")

    def serialize_csv(self, df: Any) -> bytes:
        """Serializes `df` to UTF-8 CSV bytes. Supports Polars (`write_csv`) and Pandas (`to_csv`) out of the box."""
        if hasattr(df, "write_csv") and callable(df.write_csv):
            return str(df.write_csv()).encode("utf-8")
        if hasattr(df, "to_csv") and callable(df.to_csv):
            return str(df.to_csv(index=False)).encode("utf-8")
        raise NotImplementedError(f"Override serialize_csv() to export {type(df).__name__} to CSV.")

    @bind_output(render.ui)
    def explorer_toolbar_ui(self) -> TagChild:
        subject = self.active_subject()
        if subject is None:
            return shiny_ui.div()

        view_choices = self.table_views(subject)
        if not view_choices:
            return shiny_ui.div()

        current_mode = self.val("table_view_mode", isolate=True)
        selected_mode = self.resolve_selected_mode(subject, view_choices, current_mode)
        badges = list(self.toolbar_badges_ui(subject))

        return shiny_ui.div(
            shiny_ui.div(*badges, class_="d-flex flex-wrap align-items-center gap-1") if badges else shiny_ui.div(),
            shiny_ui.div(
                shiny_ui.input_radio_buttons(
                    "table_view_mode",
                    label=None,
                    choices=view_choices,
                    selected=selected_mode,
                    inline=True,
                ),
                shiny_ui.download_button(
                    module.resolve_id("download_table_csv"),
                    "Export CSV",
                    icon=Icon.lucide("download", size=14),
                    class_="btn-sm btn-outline-secondary ms-2",
                ),
                class_="d-flex align-items-center",
            ),
            class_="d-flex flex-wrap justify-content-between align-items-center mb-3 p-2 bg-light rounded border",
        )

    @reactive_calc
    def active_table_df(self) -> Any | None:
        subject = self.active_subject()
        if subject is None:
            return None
        views = self.table_views(subject)
        default_mode = next(iter(views.keys()), "")
        mode = str(self.val("table_view_mode", default=default_mode) or default_mode)
        if mode not in views and default_mode:
            mode = default_mode
        return self.resolve_table(subject, mode)

    @bind_output(render.ui)
    def explorer_body_ui(self) -> TagChild:
        df = self.active_table_df()
        if df is None:
            return EmptyState(
                icon=self.empty_icon,
                title=self.empty_title,
                message=self.empty_message,
            ).ui()
        return shiny_ui.output_data_frame(self.grid_output_id)

    @bind_datagrid(id_attr="grid_output_id", height_attr="grid_height", filters=True)
    def render_table_datagrid(self) -> Any:
        df = self.active_table_df()
        req(df is not None)
        assert df is not None
        return df

    def _csv_filename(self) -> str:
        mode = str(self.val("table_view_mode", default="table") or "table")
        return f"{self.csv_prefix}_{mode}.csv"

    @bind_download("download_table_csv", filename=_csv_filename)
    def download_table_csv(self) -> Any:
        df = self.active_table_df()
        if df is not None:
            yield self.serialize_csv(df)


class BasePlotterExplorer[M](Component[Any]):
    """Generic interactive plotter explorer with `WidgetSelector`-introspected controls.

    Subclasses bind a `plotter_registry` mapping plot keys to `(label, plotter_cls)` pairs,
    configure `widget_selector_cls`, implement `active_subject()`, and bind a chart output
    renderer (such as `WidgetPlotterExplorer`).

    Attributes:
        plotter_registry (ClassVar[Mapping[str, tuple[str, type[Any]]]]): Mapping of `{plot_key: (label, plotter_cls)}`.
        widget_selector_cls (ClassVar[type[WidgetSelector]]): `WidgetSelector` subclass used to introspect plotters.
        default_plot_key (ClassVar[str]): Default selected plot key when available.
        param_prefix (ClassVar[str]): Input ID prefix for introspected plotter widgets.
        sidebar_width (ClassVar[int]): Width of the plot controls sidebar in pixels.
        chart_height (ClassVar[str]): CSS height of the chart output widget.
        chart_output_id (ClassVar[str]): Output ID of the main chart widget. Defaults to `"main_chart"`.
    """

    plotter_registry: ClassVar[Mapping[str, tuple[str, type[Any]]]] = {}
    widget_selector_cls: ClassVar[type[WidgetSelector]] = PlotterWidgetSelector
    default_plot_key: ClassVar[str] = ""
    param_prefix: ClassVar[str] = "plot"
    download_prefix: ClassVar[str] = "plot"
    sidebar_width: ClassVar[int] = 310
    chart_height: ClassVar[str] = "72vh"
    chart_output_id: ClassVar[str] = "main_chart"
    empty_icon: ClassVar[str] = "chart-bar"
    empty_title: ClassVar[str] = "No Active Model to Plot"
    empty_message: ClassVar[str] = "Fit or select a model in the sidebar to explore interactive visualizations."

    def __init__(self) -> None:
        super().__init__()
        self._plotter_selector = self.widget_selector_cls.default()

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.layout_sidebar(
            shiny_ui.sidebar(
                shiny_ui.output_ui("plot_selector_ui"),
                shiny_ui.output_ui("extra_controls_ui"),
                shiny_ui.output_ui("plot_parameters_ui"),
                width=cls.sidebar_width,
            ),
            shiny_ui.output_ui("plot_canvas_ui", fill=True, fillable=True),
            fillable=True,
            fill=True,
        )

    def active_subject(self) -> M | None:
        """Returns the active domain model to plot, or None if unavailable."""
        raise NotImplementedError

    def available_choices(self, subject: M) -> dict[str, str]:
        """Returns `{plot_key: label}` choices available for `subject`."""
        return {key: label for key, (label, _) in self.plotter_registry.items()}

    def selector_header_ui(self, subject: M) -> TagChild:
        """Optional status or unlock hint badge rendered above the Plot Type dropdown."""
        return None

    def render_extra_controls(self, subject: M, plot_key: str) -> TagChild:
        """Optional plot-specific controls rendered between Plot Type and Plot Parameters."""
        return shiny_ui.div()

    def build_chart(self, plotter: Any, subject: M, plot_key: str) -> Any | None:
        """Invokes `plotter.plot(subject)` (override for plotters requiring special arguments)."""
        return plotter.plot(subject)

    def render_chart_widget(self) -> TagChild:
        """Override in subclasses to render the output container widget for `self.chart_output_id`."""
        raise NotImplementedError

    def resolve_selected_plot(
        self,
        subject: M,
        choices: dict[str, str],
        current_plot: str | None,
    ) -> str | None:
        """Selects the active plot key given `choices` and the previous `current_plot`."""
        if not choices:
            return None
        fallback = self.default_plot_key if self.default_plot_key in choices else next(iter(choices.keys()), None)
        return str(current_plot) if current_plot in choices else fallback

    def resolved_plot_key(self, subject: M) -> str | None:
        """Resolves the currently selected valid plot key for `subject`."""
        choices = self.available_choices(subject)
        if not choices:
            return None
        fallback = self.default_plot_key if self.default_plot_key in choices else next(iter(choices.keys()))
        cur = self.val("plot_type", default=fallback)
        return str(cur) if cur in choices else fallback

    @bind_output(render.ui)
    def plot_selector_ui(self) -> TagChild:
        subject = self.active_subject()
        if subject is None:
            return shiny_ui.p(self.empty_message, class_="text-muted small mb-0")

        choices = self.available_choices(subject)
        current_plot = self.val("plot_type", isolate=True)
        selected_plot = self.resolve_selected_plot(subject, choices, current_plot)

        header = self.selector_header_ui(subject)
        return shiny_ui.div(
            header,
            shiny_ui.input_select(
                "plot_type",
                "Plot Type",
                choices=choices,
                selected=selected_plot,
            ),
        )

    @bind_output(render.ui)
    def extra_controls_ui(self) -> TagChild:
        subject = self.active_subject()
        if subject is None:
            return shiny_ui.div()
        plot_key = self.resolved_plot_key(subject)
        if not plot_key:
            return shiny_ui.div()
        return self.render_extra_controls(subject, plot_key)

    @bind_output(render.ui)
    def plot_parameters_ui(self) -> TagChild:
        subject = self.active_subject()
        if subject is None:
            return shiny_ui.div()
        plot_key = self.resolved_plot_key(subject)
        if not plot_key or plot_key not in self.plotter_registry:
            return shiny_ui.div()

        plotter_cls = self.plotter_registry[plot_key][1]
        widgets = self._plotter_selector.build_form(plotter_cls, prefix=f"{self.param_prefix}_{plot_key}_")
        return shiny_ui.div(*widgets)

    @reactive_calc
    def current_chart(self) -> Any | None:
        subject = self.active_subject()
        if subject is None:
            return None
        plot_key = self.resolved_plot_key(subject)
        if not plot_key or plot_key not in self.plotter_registry:
            return None

        plotter_cls = self.plotter_registry[plot_key][1]
        kwargs = self._plotter_selector.extract_kwargs(
            plotter_cls,
            self.input,
            prefix=f"{self.param_prefix}_{plot_key}_",
        )
        inst = plotter_cls(**kwargs)
        plotter = inst.with_theme(self.state.color_mode) if hasattr(inst, "with_theme") else inst
        return self.build_chart(plotter, subject, plot_key)

    @bind_output(render.ui)
    def plot_canvas_ui(self) -> TagChild:
        if self.active_subject() is None:
            return EmptyState(
                icon=self.empty_icon,
                title=self.empty_title,
                message=self.empty_message,
            ).ui()
        return self.render_chart_widget()


class WidgetPlotterExplorer[M](BasePlotterExplorer[M]):
    """Interactive widget plotter explorer with `PlotterWidgetSelector` controls and container auto-sizing.

    Lazily uses `shinywidgets.output_widget` and `bind_widget` so `shiny-kit` remains free of
    charting library dependencies while providing a ready-to-use base class for Altair/Plotly/widget
    plotter explorers.
    """

    resizer: ClassVar[Callable[[Any, float | None, float | None], Any] | None] = None

    def render_chart_widget(self) -> TagChild:
        """Renders the `shinywidgets.output_widget` container for `self.chart_output_id`."""
        shinywidgets = importlib.import_module("shinywidgets")
        return shinywidgets.output_widget(
            self.chart_output_id,
            width="100%",
            height=self.chart_height,
            fill=True,
            fillable=True,
        )

    @bind_widget(id_attr="chart_output_id")
    def render_main_chart(self) -> Any:
        """Reactively renders `self.current_chart()` fitted to its container."""
        chart = self.current_chart()
        req(chart is not None)
        assert chart is not None
        return chart
