---
title: shiny-kit
description: Object-oriented PyShiny component architecture, reactive descriptors, introspective widgets, and two-tier model caching.
---

# shiny-kit

`shiny-kit` provides an object-oriented component architecture for [PyShiny](https://shiny.posit.co/py/), combining typed session state, reactive descriptors, introspective parameter widgets, hierarchical progress reporting, and two-tier content-addressable caching.

## Core Modules

- **`shiny_kit.component`**: `Component`, `Tab`, `ParentTab`, and `subcomponent` composition with declarative `Requirement` gating.
- **`shiny_kit.decorators`**: `reactive_prop`, `reactive_calc`, `BaseBinder`, `bind_output`, `bind_datagrid`, `bind_download`, `bind_effect`, and `bind_task`.
- **`shiny_kit.state`**: `BaseSessionState`, `ArtifactRegistry`, `artifact_registry`, and `Requirement`.
- **`shiny_kit.explorers`**: High-level reusable views (`ArtifactSwitcher`, `TableExplorer`, `BasePlotterExplorer`).
- **`shiny_kit.widgets`**: `WidgetSelector`, `ParameterSpec`, and `DocstringParser` for automatic Shiny input generation from Python class signatures.
- **`shiny_kit.ui`**: `Icon`, `EmptyState`, `KPICard`, `KPIRow`, and `StagedList`.
- **`shiny_kit.caching`**: `BaseCache` thread-safe LRU + disk content-addressable cache.
- **`shiny_kit.progress`**: `ProgressReporter`, `progress_context`, `current_progress`, and `LazyShinyProgress`.
- **`shiny_kit.warnings`**: `WarningCatcher` context manager for capturing and deduplicating user warnings.
