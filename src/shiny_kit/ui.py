"""Stateless UI layout primitives (`Icon`, `EmptyState`, `KPICard`, `KPIRow`), reactive `StagedList`, and app footer/CLI helpers."""

import argparse
import importlib
import importlib.metadata
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from email.utils import parseaddr
from functools import lru_cache
from pathlib import Path
from typing import Any

import lucide
from htmltools import HTML, Tag, TagChild
from shiny import reactive
from shiny import ui as shiny_ui
from shiny._main import run_app


class Icon:
    """Renders Lucide SVG icons wrapped in `htmltools.HTML` for PyShiny UI elements.

    Examples:
        ```python
        from shiny_kit.ui import Icon

        svg = Icon.lucide("database", size=16)
        ```
    """

    @staticmethod
    @lru_cache(maxsize=256)
    def _render_cached_svg(icon_name: str, size: int, combined_class: str) -> str:
        svg_str = lucide._render_icon(icon_name, size, **{"class": combined_class})
        if "xmlns=" not in svg_str.lower():
            svg_str = svg_str.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        return svg_str

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
        try:
            if not kwargs:
                svg_str = cls._render_cached_svg(icon_name, size, combined_class)
            else:
                svg_str = lucide._render_icon(icon_name, size, **{"class": combined_class, **kwargs})
                if "xmlns=" not in svg_str.lower():
                    svg_str = svg_str.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        except (lucide.IconDoesNotExist, FileNotFoundError, AttributeError, ValueError) as err:
            raise ValueError(f"Icon '{name}' (normalized as '{icon_name}') does not exist in Lucide icon set.") from err

        return HTML(svg_str)

    @classmethod
    def github(cls, size: int = 14) -> HTML:
        """Returns the official GitHub Octocat mark as an inline SVG inheriting `currentColor`."""
        return HTML(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            'viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
            '<path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 '
            "0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 "
            "1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 "
            "0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 "
            "1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 "
            "0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57 "
            'C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>'
        )

    @classmethod
    def pypi(cls, size: int = 14) -> HTML:
        """Returns the official PyPI isometric cube mark as an inline SVG inheriting `currentColor`."""
        return HTML(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M12 2L3 7l9 5 9-5-9-5z"/>'
            '<path d="M3 17l9 5 9-5"/>'
            '<path d="M3 12l9 5 9-5"/>'
            '<path d="M12 12v10"/>'
            '<path d="M3 7v10"/>'
            '<path d="M21 7v10"/>'
            "</svg>"
        )


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
        border_cls = "border-dashed" if self.dashed else "bg-body-tertiary rounded border"
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
                class_="p-2 bg-body-tertiary rounded border h-100",
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


@dataclass(frozen=True, slots=True)
class PackageMetadata:
    """Resolved Python distribution metadata extracted from `importlib.metadata` or `pyproject.toml`.

    Attributes:
        name (str): Distribution package name.
        version (str): Full PEP 440 version string (including any `.dev` or `+g...` suffix).
        summary (str): One-line package summary description.
        author (str): Primary author name.
        author_email (str): Primary author contact email address.
        author_url (str): Author website URL.
        homepage (str): Canonical documentation / homepage URL.
        documentation (str): Documentation site URL.
        repository (str): Source code GitHub repository URL.
        bug_tracker (str): Issue tracker URL.
        changelog (str): Changelog URL.
        pypi_url (str): PyPI package project URL.
        license_spdx (str): Short license identifier (`"GPL-3.0"`).
    """

    name: str
    version: str
    summary: str
    author: str
    author_email: str
    author_url: str
    homepage: str
    documentation: str
    repository: str
    bug_tracker: str
    changelog: str
    pypi_url: str
    license_spdx: str = "GPL-3.0"

    @property
    def short_version(self) -> str:
        """Returns the version string with any local `+g...` git build suffix removed for compact display."""
        return self.version.split("+", 1)[0]

    @classmethod
    @lru_cache(maxsize=8)
    def load(
        cls,
        package_name: str = "shiny-kit",
        *,
        pyproject_path: Path | None = None,
    ) -> "PackageMetadata":
        """Loads distribution metadata via `importlib.metadata` with a `pyproject.toml` fallback.

        Args:
            package_name: Installed Python distribution name to inspect.
            pyproject_path: Optional explicit path to `pyproject.toml` when uninstalled.

        Returns:
            Populated `PackageMetadata` instance.
        """
        name = package_name
        version = "0.1.0"
        summary = f"Interactive PyShiny application for {package_name}"
        author = ""
        author_email = ""
        author_url = ""
        urls: dict[str, str] = {}

        mod_name = package_name.replace("-", "_")
        try:
            ver_mod = importlib.import_module(f"{mod_name}._version")
            vcs_version = getattr(ver_mod, "__version__", None)
            if isinstance(vcs_version, str) and vcs_version:
                version = vcs_version
        except Exception:
            pass

        try:
            meta = importlib.metadata.metadata(package_name)
            name = meta.get("Name") or name
            version = meta.get("Version") or version
            summary = meta.get("Summary") or summary

            raw_author = meta.get("Author")
            raw_author_email = meta.get("Author-email")
            if raw_author_email:
                parsed_name, parsed_email = parseaddr(raw_author_email)
                if parsed_name:
                    author = parsed_name
                if parsed_email:
                    author_email = parsed_email
            if raw_author:
                author = raw_author

            project_urls = meta.get_all("Project-URL") or []
            for entry in project_urls:
                if "," in entry:
                    label, url_val = entry.split(",", 1)
                    urls[label.strip().lower()] = url_val.strip()
        except importlib.metadata.PackageNotFoundError:
            candidate = pyproject_path or (Path.cwd() / "pyproject.toml")
            if candidate.is_file():
                try:
                    data = tomllib.loads(candidate.read_text(encoding="utf-8"))
                    proj = data.get("project", {})
                    name = str(proj.get("name", name))
                    summary = str(proj.get("description", summary))
                    authors = proj.get("authors", [])
                    if authors and isinstance(authors[0], dict):
                        author = str(authors[0].get("name", author))
                        author_email = str(authors[0].get("email", author_email))
                        author_url = str(authors[0].get("url", author_url))
                    for label, url_val in proj.get("urls", {}).items():
                        urls[str(label).strip().lower()] = str(url_val).strip()
                except Exception:
                    pass

        repo_url = urls.get("repository", "")
        homepage_url = urls.get("homepage", repo_url or "#")
        docs_url = urls.get("documentation", homepage_url)
        bugs_url = urls.get("bug tracker", f"{repo_url}/issues" if repo_url else "#")
        changelog_url = urls.get("changelog", f"{repo_url}/blob/master/CHANGELOG.md" if repo_url else "#")
        resolved_author_url = (
            urls.get("author")
            or urls.get("author url")
            or author_url
            or (f"mailto:{author_email}" if author_email else "#")
        )

        return cls(
            name=name,
            version=version,
            summary=summary,
            author=author,
            author_email=author_email,
            author_url=resolved_author_url,
            homepage=homepage_url,
            documentation=docs_url,
            repository=repo_url or "#",
            bug_tracker=bugs_url,
            changelog=changelog_url,
            pypi_url=f"https://pypi.org/project/{name}/",
        )


@dataclass(frozen=True, slots=True)
class EcosystemPillarSpec:
    """Specification for an ecosystem pillar pill in `package_footer()`.

    Attributes:
        name (str): Display name of the pillar.
        slug (str): CSS modifier slug (e.g. `"seroverse"`, `"seroref"`, `"kbank"`, `"serogen"`, `"seroepi"`).
        icon (str): Lucide icon identifier.
        url (str): Documentation or landing page URL.
        description (str): Tooltip summary of the pillar's role.
        is_active (bool): Whether this pillar corresponds to the running application.
        color (str): Light-mode accent color for the pillar icon/active badge.
        dark_color (str | None): Optional dark-mode accent color override.
    """

    name: str
    slug: str
    icon: str
    url: str
    description: str
    is_active: bool = False
    color: str = "#7c3aed"
    dark_color: str | None = None


FOOTER_CSS = """
.sv-app-footer {
    margin-top: 2.25rem;
    padding: 0.9rem 1.25rem;
    border-top: 1px solid var(--sv-border, rgba(139, 92, 246, 0.28));
    background:
        radial-gradient(circle at 12% 50%, var(--sv-glow, rgba(139, 92, 246, 0.12)), transparent 50%),
        var(--sv-surface-tint, linear-gradient(145deg, rgba(124, 58, 237, 0.06), rgba(219, 39, 119, 0.03)));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    font-size: 0.8rem;
}

.sv-app-footer-inner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.85rem 1.25rem;
    max-width: 1600px;
    margin: 0 auto;
}

.sv-footer-left,
.sv-footer-right {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.55rem 0.75rem;
}

.sv-footer-pkg-brand {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    font-weight: 800;
    font-size: 0.88rem;
    color: var(--bs-body-color) !important;
    text-decoration: none !important;
}

.sv-footer-pkg-brand svg {
    color: var(--sv-accent, #7c3aed);
}

.sv-footer-pkg-brand:hover {
    color: var(--sv-accent, #7c3aed) !important;
}

.sv-footer-version-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    font-family: var(--bs-font-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.16rem 0.52rem;
    border-radius: 999px;
    background: var(--sv-accent-soft, rgba(139, 92, 246, 0.14));
    color: var(--sv-accent, #7c3aed);
    border: 1px solid var(--sv-border, rgba(139, 92, 246, 0.3));
    text-decoration: none !important;
}

.sv-footer-meta-sep {
    opacity: 0.35;
    user-select: none;
}

.sv-footer-author {
    color: var(--bs-secondary-color);
    font-size: 0.78rem;
}

.sv-footer-author a {
    color: var(--bs-body-color);
    font-weight: 600;
    text-decoration: none;
}

.sv-footer-author a:hover {
    color: var(--sv-accent, #7c3aed);
    text-decoration: underline;
}

.sv-footer-icon-links {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.sv-footer-icon-link {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.55rem;
    border-radius: 0.45rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--bs-secondary-color) !important;
    background: rgba(148, 163, 184, 0.09);
    border: 1px solid rgba(148, 163, 184, 0.2);
    text-decoration: none !important;
    transition: color 0.15s ease, border-color 0.15s ease, background-color 0.15s ease, transform 0.15s ease;
}

.sv-footer-icon-link:hover {
    color: var(--sv-accent, #7c3aed) !important;
    border-color: var(--sv-border, rgba(139, 92, 246, 0.35));
    background: var(--sv-accent-soft, rgba(139, 92, 246, 0.12));
    transform: translateY(-1px);
}

.sv-footer-eco-pills {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
}

.sv-footer-eco-pill {
    --active-pill-color: var(--pill-color, #7c3aed);
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    padding: 0.22rem 0.62rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-decoration: none !important;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(148, 163, 184, 0.08);
    color: var(--bs-body-color) !important;
    transition: transform 0.15s ease, border-color 0.15s ease, background-color 0.15s ease;
}

[data-bs-theme="dark"] .sv-footer-eco-pill {
    --active-pill-color: var(--pill-dark-color, var(--pill-color, #c084fc));
}

.sv-footer-eco-pill svg {
    color: var(--active-pill-color);
}

.sv-footer-eco-pill:hover {
    transform: translateY(-1px);
    border-color: color-mix(in srgb, var(--active-pill-color) 45%, transparent);
    background: color-mix(in srgb, var(--active-pill-color) 12%, transparent);
}

.sv-footer-eco-pill.is-active {
    color: var(--active-pill-color) !important;
    background: color-mix(in srgb, var(--active-pill-color) 14%, transparent);
    border-color: color-mix(in srgb, var(--active-pill-color) 40%, transparent);
}

.sv-footer-eco-pill.is-active svg {
    color: currentColor;
}
"""


def package_footer(
    package_name: str = "shiny-kit",
    *,
    brand_title: str | None = None,
    brand_icon: str = "chart-network",
    author_url: str | None = None,
    meta: PackageMetadata | None = None,
    pillars: Sequence[EcosystemPillarSpec] = (),
) -> Tag:
    """Renders a 2-zone glassmorphic application footer populated from `PackageMetadata` and optional ecosystem pills.

    Args:
        package_name: Distribution name used when `meta` is omitted.
        brand_title: Display title for the brand link. Defaults to `meta.name`.
        brand_icon: Lucide icon name for the brand mark. Defaults to `"chart-network"`.
        author_url: Optional explicit author profile URL overriding `meta.author_url`.
        meta: Optional pre-loaded `PackageMetadata` instance.
        pillars: Optional sequence of `EcosystemPillarSpec` items rendered in the right zone.

    Returns:
        `shiny.ui.Tag` `<footer>` element suitable for `shiny.ui.page_navbar(footer=...)`.
    """
    pkg = meta or PackageMetadata.load(package_name)
    display_brand = brand_title or pkg.name
    resolved_author_url = author_url or pkg.author_url

    left_zone = shiny_ui.div(
        shiny_ui.a(
            Icon.lucide(brand_icon, size=16),
            shiny_ui.span(display_brand),
            href=pkg.homepage,
            target="_blank",
            rel="noopener",
            title=pkg.summary,
            class_="sv-footer-pkg-brand",
        ),
        shiny_ui.a(
            f"v{pkg.short_version}",
            href=pkg.changelog,
            target="_blank",
            rel="noopener",
            title=f"{pkg.name} v{pkg.version} ({pkg.license_spdx} License) — View Changelog",
            class_="sv-footer-version-badge",
        ),
        shiny_ui.span("·", class_="sv-footer-meta-sep"),
        shiny_ui.span(
            "by ",
            shiny_ui.a(
                pkg.author,
                href=resolved_author_url,
                target="_blank",
                rel="noopener",
                title=f"{pkg.author} ({pkg.author_email})",
            ),
            class_="sv-footer-author",
        ),
        shiny_ui.span("·", class_="sv-footer-meta-sep"),
        shiny_ui.div(
            shiny_ui.a(
                Icon.lucide("book-open", size=13),
                shiny_ui.span("Docs"),
                href=pkg.documentation,
                target="_blank",
                rel="noopener",
                title=f"Documentation ({pkg.documentation})",
                class_="sv-footer-icon-link",
            ),
            shiny_ui.a(
                Icon.github(size=13),
                shiny_ui.span("GitHub"),
                href=pkg.repository,
                target="_blank",
                rel="noopener",
                title=f"GitHub Repository ({pkg.repository})",
                class_="sv-footer-icon-link",
            ),
            shiny_ui.a(
                Icon.pypi(size=13),
                shiny_ui.span("PyPI"),
                href=pkg.pypi_url,
                target="_blank",
                rel="noopener",
                title=f"PyPI Package ({pkg.pypi_url})",
                class_="sv-footer-icon-link",
            ),
            shiny_ui.a(
                Icon.lucide("bug", size=13),
                shiny_ui.span("Issues"),
                href=pkg.bug_tracker,
                target="_blank",
                rel="noopener",
                title=f"Bug Tracker ({pkg.bug_tracker})",
                class_="sv-footer-icon-link",
            ),
            class_="sv-footer-icon-links",
        ),
        class_="sv-footer-left",
    )

    pillar_nodes = [
        shiny_ui.a(
            Icon.lucide(pillar.icon, size=13),
            shiny_ui.span(pillar.name),
            href=pillar.url,
            target="_blank",
            rel="noopener",
            title=pillar.description,
            style=f"--pill-color: {pillar.color}; --pill-dark-color: {pillar.dark_color or pillar.color};",
            class_=(
                f"sv-footer-eco-pill sv-footer-eco-pill--{pillar.slug}" + (" is-active" if pillar.is_active else "")
            ),
        )
        for pillar in pillars
    ]

    right_zone = (
        shiny_ui.div(
            shiny_ui.div(*pillar_nodes, class_="sv-footer-eco-pills"),
            class_="sv-footer-right",
        )
        if pillar_nodes
        else None
    )

    return shiny_ui.tags.footer(
        shiny_ui.head_content(shiny_ui.tags.style(FOOTER_CSS)),
        shiny_ui.div(
            left_zone,
            right_zone,
            class_="sv-app-footer-inner",
        ),
        class_="sv-app-footer",
    )


def build_cli_parser(
    prog: str,
    *,
    package_name: str | None = None,
    description: str = "Launch the interactive PyShiny web application.",
    version: str | None = None,
    default_host: str = "127.0.0.1",
    default_port: int = 8000,
) -> argparse.ArgumentParser:
    """Constructs a standard CLI argument parser for a `shiny-kit` web application.

    Args:
        prog: CLI program name (e.g. `"seroepi-app"`).
        package_name: Optional distribution name used to resolve `--version` via `PackageMetadata.load()`.
        description: CLI help description string.
        version: Optional explicit version string overriding `PackageMetadata.load(package_name).version`.
        default_host: Default bind host address. Defaults to `"127.0.0.1"`.
        default_port: Default bind port number. Defaults to `8000`.

    Returns:
        Configured `argparse.ArgumentParser`.
    """
    resolved_ver = version
    if resolved_ver is None and package_name:
        resolved_ver = PackageMetadata.load(package_name).version
    resolved_ver = resolved_ver or "0.1.0"

    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {resolved_ver}",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        default=default_host,
        help=f"Host interface to bind the server to (default: {default_host}).",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=default_port,
        help=f"Port number to bind the server to (default: {default_port}; use 0 for a random available port).",
    )
    parser.add_argument(
        "-r",
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable auto-reload on source file changes (default: disabled).",
    )
    parser.add_argument(
        "-b",
        "--launch-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically open the application in the default web browser (default: enabled).",
    )
    return parser


def run_cli(
    app_target: str,
    *,
    parser: argparse.ArgumentParser | None = None,
    prog: str = "shiny-kit-app",
    package_name: str | None = None,
    description: str = "Launch the interactive PyShiny web application.",
    reload_dirs: Sequence[str | Path] | None = None,
    app_dir: str | Path | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    """Parses CLI arguments and launches the PyShiny application via `shiny._main.run_app`.

    Args:
        app_target: Module/attribute target string passed to `run_app` (e.g. `"seroepi.app.app:app"`).
        parser: Optional pre-built `argparse.ArgumentParser`.
        prog: CLI program name when `parser` is not supplied.
        package_name: Optional distribution name when `parser` is not supplied.
        description: CLI description when `parser` is not supplied.
        reload_dirs: Optional directories watched when `--reload` is active.
        app_dir: Optional root directory added to `sys.path` by `run_app`.
        argv: Optional sequence of command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        Exit status code `0`.
    """
    eff_parser = parser or build_cli_parser(prog, package_name=package_name, description=description)
    args = eff_parser.parse_args(list(argv) if argv is not None else None)
    run_app(
        app_target,
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        reload_dirs=[str(d) for d in reload_dirs] if reload_dirs else None,
        app_dir=str(app_dir) if app_dir is not None else None,
        launch_browser=bool(args.launch_browser),
    )
    return 0
