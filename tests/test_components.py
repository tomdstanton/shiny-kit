"""Unit tests for shiny_kit.decorators, shiny_kit.component, and shiny_kit.explorers."""

import asyncio
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from htmltools import TagChild
from shiny import reactive, render
from shiny import ui as shiny_ui
from shiny.types import SilentException

from shiny_kit.component import Component, ParentTab, SubComponentProxy, Tab, subcomponent
from shiny_kit.decorators import (
    BoundTask,
    _wrap_sync_or_async,
    bind_datagrid,
    bind_download,
    bind_effect,
    bind_output,
    bind_task,
    reactive_calc,
    reactive_prop,
)
from shiny_kit.explorers import ArtifactSwitcher, BasePlotterExplorer, TableExplorer
from shiny_kit.state import BaseSessionState, Requirement, artifact_registry

# ---------------------------------------------------------------------------
# Shared test fixtures & domain classes
# ---------------------------------------------------------------------------


@dataclass
class _ModelArtifact:
    name: str
    data: list[int]
    warnings_: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.data)

    def save(self) -> bytes:
        return f"saved:{self.name}".encode()


class _AppState(BaseSessionState):
    artifacts = artifact_registry[_ModelArtifact]("Models", toast="Registered {name}")


class _ChildCard(Component[_AppState]):
    count: reactive_prop[int] = reactive_prop(10)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.div("child")

    @reactive_calc
    def doubled(self) -> int:
        return self.count * 2

    @reactive_calc(events=("trigger_btn",))
    def event_calc(self) -> int:
        return self.count + 1


class _ParentView(Tab[_AppState]):
    """Primary Analysis Tab."""

    requires = (Requirement("artifacts", min_count=1, label="Models", target_tab="setup_tab"),)
    child = subcomponent(_ChildCard)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.div(cls.child.ui())

    @bind_output(render.text)
    def summary_txt(self) -> str:
        return f"Active: {self.state.artifacts.active_key}"

    @bind_datagrid("grid_out", height="300px")
    def grid_out(self) -> Any:
        return {"col": [1, 2]}

    @bind_download("dl_static", filename="report.txt")
    def dl_static(self) -> Any:
        yield b"hello"

    @bind_download("dl_dynamic", filename=lambda self: f"{self.module_id}.bin")
    def dl_dynamic(self) -> Any:
        yield b"bytes"

    @bind_effect("run_btn")
    def _on_run(self) -> str:
        warnings.warn("Effect warning", UserWarning, stacklevel=1)
        if self.val("fail_mode") == "boom":
            raise ValueError("Invalid parameter")
        if self.val("fail_mode") == "silent":
            raise SilentException()
        return "ok"

    @bind_effect()
    async def _on_async_effect(self) -> str:
        warnings.warn("Async warning", UserWarning, stacklevel=1)
        if self.val("fail_mode") == "boom":
            raise RuntimeError("Async failure")
        return "async_ok"

    @bind_task("task_btn", register_to="artifacts", notify_success="Finished {name} ({rows})")
    def fit_model(self, label: str) -> _ModelArtifact:
        warnings.warn("Convergence note", UserWarning, stacklevel=1)
        return _ModelArtifact(name=label, data=[1, 2, 3], warnings_=("Extra warning",))

    @bind_task(on_success="_custom_done", progress_message="Running Async")
    async def async_task(self, label: str) -> _ModelArtifact:
        return _ModelArtifact(name=label, data=[1])

    @bind_task(notify_success=lambda res: f"Lambda: {res.name}")
    def lambda_task(self, label: str) -> _ModelArtifact:
        return _ModelArtifact(name=label, data=[1])

    @bind_task(notify_success=True, register_to="artifacts")
    def default_toast_task(self, label: str) -> _ModelArtifact:
        return _ModelArtifact(name=label, data=[1])

    def _custom_done(self, res: _ModelArtifact) -> str:
        return f"Custom done: {res.name}"


class _ContainerTab(ParentTab[_AppState]):
    children: ClassVar[list[type[Tab[Any]]]] = [_ParentView]


class _FakeInputs(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key not in self:
            return lambda: None
        val = super().__getitem__(key)
        return val if callable(val) else (lambda: val)


def _bind_component_mock(comp: Component[Any], state: _AppState, inputs_map: dict[str, Any] | None = None) -> None:
    comp.input = _FakeInputs(inputs_map or {})  # ty: ignore[invalid-assignment]
    comp.output = MagicMock()
    mock_sess = MagicMock()
    mock_sess.id = "sess-test"
    mock_sess.ns = lambda x: f"{comp.module_id}-{x}"
    comp.session = mock_sess
    comp.state = state
    comp.module_id = "test_mod"


# ---------------------------------------------------------------------------
# 1. Decorators & Component Lifecycle Tests
# ---------------------------------------------------------------------------


def test_wrap_sync_or_async_and_descriptors() -> None:
    log: list[str] = []
    wrapped_sync = _wrap_sync_or_async(
        lambda x: x + 1,
        name="my_sync",
        before=lambda: log.append("before"),
        after=lambda r: r * 10,
    )
    assert wrapped_sync.__name__ == "my_sync"
    assert wrapped_sync(2) == 30
    assert log == ["before"]

    async def _coro(x: int) -> int:
        return x + 2

    wrapped_async = _wrap_sync_or_async(
        _coro,
        name="my_async",
        before=lambda: log.append("before_async"),
        after=lambda r: r * 10,
    )
    assert asyncio.run(wrapped_async(3)) == 50
    assert log == ["before", "before_async"]


def test_component_lifecycle_and_binders() -> None:
    card = _ChildCard()
    assert not card.has_session
    with pytest.raises(RuntimeError, match="input accessed before server"):
        _ = card.input
    with pytest.raises(RuntimeError, match="output accessed before server"):
        _ = card.output
    with pytest.raises(RuntimeError, match="session accessed before server"):
        _ = card.session
    with pytest.raises(RuntimeError, match="state accessed before server"):
        _ = card.state

    # Class-level descriptor access
    assert isinstance(_ChildCard.count, reactive_prop)
    assert isinstance(_ChildCard.doubled, reactive_calc)
    assert isinstance(_ParentView.child, SubComponentProxy)
    assert isinstance(_ParentView.summary_txt, bind_output)
    assert isinstance(_ParentView.fit_model, bind_task)
    assert _ParentView.fit_model.display_label == "Fit model"
    assert _ParentView.async_task.display_label == "Running Async"

    with reactive.isolate():
        state = _AppState()
        _bind_component_mock(card, state, {"txt": "  hello  ", "empty": "   ", "num": 42, "none_val": None})
        assert card.has_session
        assert card.val("txt") == "hello"
        assert card.val("txt", strip=False) == "  hello  "
        assert card.val("empty", default="fallback") == "fallback"
        assert card.val("missing", default="def") == "def"
        assert card.val("none_val", default="def") == "def"
        assert card.val("num", isolate=True) == 42

        # Reactive calc access
        calc_fn = card.doubled
        assert callable(calc_fn)

        # Reactive prop set before get on a fresh instance
        card2 = _ChildCard()
        card2.count = 25
        assert card2.count == 25
        assert card2.val("any", default="no_sess") == "no_sess"

        # ParentView UI, Tab title, and resource gating
        assert _ParentView.title() == "Primary Analysis Tab"
        assert _ContainerTab.title() == "_Container"
        comp_ui_html = str(Component.ui.__func__(_ParentView, "pv"))  # ty: ignore[unresolved-attribute]
        assert "_gate_callout" in comp_ui_html
        assert "child" in str(_ChildCard.ui("c1"))
        assert isinstance(_ContainerTab._render_ui(), shiny_ui._navs.NavSetCard)
        assert isinstance(_ParentView.ui("pv"), shiny_ui._navs.NavPanel)

        pv = _ParentView()
        _bind_component_mock(pv, state, {"fail_mode": None})
        assert isinstance(pv.child, _ChildCard)
        assert not pv.is_unlocked()
        assert len(pv.unmet_requirements()) == 1

        # Transform on bind_datagrid & gated renderer
        grid_res = pv.grid_out()
        assert isinstance(grid_res, render.DataGrid)
        assert _ParentView.grid_out.transform(pv, grid_res) is grid_res
        assert _ParentView.grid_out.transform(pv, None) is None

        # Guarded sync and async effects
        with patch("shiny.ui.notification_show") as mock_notify:
            guarded_sync = _ParentView._on_run._wrap_guarded_effect(pv, pv._on_run)
            assert guarded_sync() == "ok"
            assert mock_notify.called

            pv.input["fail_mode"] = "boom"  # ty: ignore[invalid-assignment]
            assert guarded_sync() is None

            pv.input["fail_mode"] = "silent"  # ty: ignore[invalid-assignment]
            with pytest.raises(SilentException):
                guarded_sync()

            guarded_async = _ParentView._on_async_effect._wrap_guarded_effect(pv, pv._on_async_effect)
            pv.input["fail_mode"] = "boom"  # ty: ignore[invalid-assignment]
            assert asyncio.run(guarded_async()) is None
            pv.input["fail_mode"] = None  # ty: ignore[invalid-assignment]
            assert asyncio.run(guarded_async()) == "async_ok"

        # Task completion handling across all notify_success variants
        artifact = _ModelArtifact("m1", [10, 20])
        with patch("shiny.ui.notification_show") as mock_notify:
            _ParentView.fit_model._handle_completion(pv, None)
            _ParentView.fit_model._handle_completion(pv, artifact, warnings=("w1",))
            assert state.artifacts.active == artifact
            assert pv.is_unlocked()
            assert pv.unmet_requirements() == []
            assert mock_notify.called

            _ParentView.async_task._handle_completion(pv, artifact)
            _ParentView.lambda_task._handle_completion(pv, artifact)
            _ParentView.default_toast_task._handle_completion(pv, artifact)

        # BoundTask wrapper methods & descriptor instantiation
        bt_sync = pv.fit_model
        assert isinstance(bt_sync, BoundTask)
        bt_async = pv.async_task
        assert isinstance(bt_async, BoundTask)

        mock_ext = MagicMock()
        mock_ext.status.return_value = "success"
        mock_ext.result.return_value = artifact
        bt = BoundTask(mock_ext)
        bt("arg1")
        assert bt.invocation_count == 1
        assert bt.status() == "success"
        assert bt.result() == artifact
        bt.cancel()
        assert mock_ext.cancel.called

        # Test bind() methods directly on binders and resource gate
        with patch("shiny.reactive.effect", side_effect=lambda *a, **kw: lambda fn: fn):
            _ParentView.summary_txt.bind(pv, pv.input, pv.output, pv.session)
            _ParentView.dl_static.bind(pv, pv.input, pv.output, pv.session)
            _ParentView.dl_dynamic.bind(pv, pv.input, pv.output, pv.session)
            _ParentView._on_run.bind(pv, pv.input, pv.output, pv.session)
            _ParentView._on_async_effect.bind(pv, pv.input, pv.output, pv.session)
            _ParentView.fit_model.bind(pv, pv.input, pv.output, pv.session)
            pv._bind_resource_gate(pv.input, pv.output, pv.session)

        # Test full server() binding via mocked module.server
        def _fake_mod_server(fn: Any) -> Any:
            def _runner(id_: str) -> None:
                fn(pv.input, pv.output, pv.session)

            return _runner

        with (
            patch("shiny.module.server", side_effect=_fake_mod_server),
            patch.object(_AppState, "from_session", return_value=state),
            patch("shiny.reactive.effect", side_effect=lambda *a, **kw: lambda fn: fn),
        ):
            pv.server("main_pv")
            ct = _ContainerTab()
            _bind_component_mock(ct, state)
            ct._register_server()


# ---------------------------------------------------------------------------
# 2. Explorers (ArtifactSwitcher, TableExplorer, BasePlotterExplorer)
# ---------------------------------------------------------------------------


class _DemoSwitcher(ArtifactSwitcher[_ModelArtifact]):
    registry_kind = "artifacts"
    enable_download = True
    enable_clear = True
    enable_upload = True
    card_wrapped = True

    def summary_ui(self, item: _ModelArtifact) -> TagChild:
        return shiny_ui.span(f"Items: {len(item)}")

    def load_artifact_from_path(self, path: Path, filename: str) -> _ModelArtifact:
        return _ModelArtifact(name=filename.removesuffix(".pkl"), data=[1, 2])


class _PlainSwitcher(ArtifactSwitcher[_ModelArtifact]):
    registry_kind = "artifacts"
    enable_upload = False
    card_wrapped = False
    show_bottom_hr = True


class _PolarsLikeDF:
    def write_csv(self) -> str:
        return "a,b\n1,2\n"


class _PandasLikeDF:
    def to_csv(self, *, index: bool = False) -> str:
        return "x,y\n3,4\n"


class _DemoTableExplorer(TableExplorer[_ModelArtifact]):
    def active_subject(self) -> _ModelArtifact | None:
        return self.state.artifacts.active

    def table_views(self, subject: _ModelArtifact) -> dict[str, str]:
        return {"polars": "Polars View", "pandas": "Pandas View"}

    def resolve_table(self, subject: _ModelArtifact, mode: str) -> Any | None:
        return _PolarsLikeDF() if mode == "polars" else _PandasLikeDF()

    def toolbar_badges_ui(self, subject: _ModelArtifact) -> list[TagChild]:
        return [shiny_ui.span(subject.name)]


class _DummyPlotter:
    def __init__(self, title: str = "Plot", show_legend: bool = True) -> None:
        self.title = title
        self.show_legend = show_legend
        self.theme = "light"

    def with_theme(self, mode: str) -> "_DummyPlotter":
        self.theme = mode
        return self

    def plot(self, subject: _ModelArtifact) -> dict[str, Any]:
        return {"subject": subject.name, "title": self.title, "theme": self.theme}


class _DemoPlotterExplorer(BasePlotterExplorer[_ModelArtifact]):
    plotter_registry: ClassVar[dict[str, tuple[str, type[Any]]]] = {
        "bar": ("Bar Chart", _DummyPlotter),
    }
    default_plot_key = "bar"

    def active_subject(self) -> _ModelArtifact | None:
        return self.state.artifacts.active

    def render_chart_widget(self) -> TagChild:
        return shiny_ui.div(id=self.chart_output_id)


def test_artifact_switcher_table_and_plotter_explorers(tmp_path: Path) -> None:
    with reactive.isolate():
        state = _AppState()

        # 1. ArtifactSwitcher
        switcher_ui = str(_DemoSwitcher._render_ui())
        assert "artifact_file_upload" in switcher_ui
        assert "switcher_body_ui" in str(_PlainSwitcher._render_ui())

        sw = _DemoSwitcher()
        _bind_component_mock(sw, state)
        plain_sw = _PlainSwitcher()
        _bind_component_mock(plain_sw, state)

        # Empty state
        assert str(_DemoSwitcher.switcher_body_ui.fn(sw)) == "<div></div>"  # ty: ignore[call-non-callable]
        assert sw.download_filename(None) == "artifact.pkl"

        art1 = _ModelArtifact("alpha model!", [1, 2, 3])
        art2 = _ModelArtifact("beta", [4, 5])
        state.artifacts.register(art1)
        state.artifacts.register(art2, set_active=False)

        assert sw.download_filename(art1) == "alpha_model.pkl"
        assert sw._artifact_download_filename() == "alpha_model.pkl"
        assert sw.serialize_artifact(art1) == b"saved:alpha model!"
        assert sw.serialize_artifact(object()) is None  # ty: ignore[invalid-argument-type]
        assert list(sw.download_artifact()) == [b"saved:alpha model!"]

        body_html = str(_DemoSwitcher.switcher_body_ui.fn(sw))  # ty: ignore[call-non-callable]
        assert "Active Artifact (2)" in body_html
        assert "Clear All" in body_html
        plain_html = str(_PlainSwitcher.switcher_body_ui.fn(plain_sw))  # ty: ignore[call-non-callable]
        assert "<hr" in plain_html

        sw.input["active_artifact_select"] = "beta"  # ty: ignore[invalid-assignment]
        sw._on_select_artifact()
        assert state.artifacts.active_key == "beta"

        # Upload handler
        sw.input["artifact_file_upload"] = [{"datapath": str(tmp_path / "f.pkl"), "name": "up.pkl"}]  # ty: ignore[invalid-assignment]
        with patch.object(BoundTask, "invoke") as mock_invoke:
            sw._on_upload_artifact_file()
            mock_invoke.assert_called_once()
        plain_sw._on_upload_artifact_file()  # enable_upload=False early return

        with patch("shiny.ui.notification_show"):
            sw._on_delete_artifact()
            assert state.artifacts.count == 1
            sw._on_clear_artifacts()
            assert state.artifacts.count == 0

        loaded = _DemoSwitcher.load_artifact_task.fn(sw, str(tmp_path / "test.pkl"), "loaded_art.pkl")  # ty: ignore[call-non-callable]
        msg = sw._on_artifact_loaded(loaded)
        assert msg == "Loaded 'loaded_art'."
        assert state.artifacts.active_key == "loaded_art"

        # 2. TableExplorer
        texp = _DemoTableExplorer()
        _bind_component_mock(texp, state, {"table_view_mode": "polars"})
        assert "explorer_toolbar_ui" in str(_DemoTableExplorer._render_ui())
        toolbar_html = str(_DemoTableExplorer.explorer_toolbar_ui.fn(texp))  # ty: ignore[call-non-callable]
        assert "Export CSV" in toolbar_html

        df_polars = _DemoTableExplorer.active_table_df.fn(texp)  # ty: ignore[call-non-callable]
        assert texp.serialize_csv(df_polars) == b"a,b\n1,2\n"
        assert texp.serialize_csv(_PandasLikeDF()) == b"x,y\n3,4\n"
        with pytest.raises(NotImplementedError):
            texp.serialize_csv(object())

        body_populated = str(_DemoTableExplorer.explorer_body_ui.fn(texp))  # ty: ignore[call-non-callable]
        assert "estimates_datagrid" in body_populated
        assert texp._csv_filename() == "export_polars.csv"
        with patch.object(_DemoTableExplorer, "active_table_df", return_value=df_polars):
            assert list(texp.download_table_csv()) == [b"a,b\n1,2\n"]
            assert _DemoTableExplorer.estimates_datagrid.fn(texp) is df_polars  # ty: ignore[call-non-callable]

        # 3. BasePlotterExplorer
        pexp = _DemoPlotterExplorer()
        _bind_component_mock(pexp, state, {"plot_type": "bar", "plot_bar_title": "Custom Title"})
        assert _DemoPlotterExplorer._render_ui() is not None
        assert "Plot Type" in str(_DemoPlotterExplorer.plot_selector_ui.fn(pexp))  # ty: ignore[call-non-callable]
        assert "plot_bar_title" in str(_DemoPlotterExplorer.plot_parameters_ui.fn(pexp))  # ty: ignore[call-non-callable]
        assert str(_DemoPlotterExplorer.extra_controls_ui.fn(pexp)) == "<div></div>"  # ty: ignore[call-non-callable]
        assert "main_altair_chart" in str(_DemoPlotterExplorer.plot_canvas_ui.fn(pexp))  # ty: ignore[call-non-callable]

        state.color_mode = "dark"
        chart = _DemoPlotterExplorer.current_chart.fn(pexp)  # ty: ignore[call-non-callable]
        assert chart == {"subject": "loaded_art", "title": "Custom Title", "theme": "dark"}

        # Empty state when no active subject
        state.artifacts.clear()
        assert str(_DemoTableExplorer.explorer_toolbar_ui.fn(texp)) == "<div></div>"  # ty: ignore[call-non-callable]
        assert _DemoTableExplorer.active_table_df.fn(texp) is None  # ty: ignore[call-non-callable]
        assert "No Data Available" in str(_DemoTableExplorer.explorer_body_ui.fn(texp))  # ty: ignore[call-non-callable]
        assert "Fit or select" in str(_DemoPlotterExplorer.plot_selector_ui.fn(pexp))  # ty: ignore[call-non-callable]
        assert str(_DemoPlotterExplorer.extra_controls_ui.fn(pexp)) == "<div></div>"  # ty: ignore[call-non-callable]
        assert str(_DemoPlotterExplorer.plot_parameters_ui.fn(pexp)) == "<div></div>"  # ty: ignore[call-non-callable]
        assert _DemoPlotterExplorer.current_chart.fn(pexp) is None  # ty: ignore[call-non-callable]
        assert "No Active Model to Plot" in str(_DemoPlotterExplorer.plot_canvas_ui.fn(pexp))  # ty: ignore[call-non-callable]
