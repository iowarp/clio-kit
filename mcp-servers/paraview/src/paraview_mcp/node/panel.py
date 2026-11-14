"""The view's layout: a top bar, a left tool rail and its panel, and the 3D view.

Compact and icon-led, as ParaView's own toolbars are. Every icon button has a
short tooltip and the same text as its accessible name; a pressed state is
shown by ``aria-pressed`` and a pressed style, never by colour alone. At a
narrow width the panel opens as a sheet over the lower part of the view.
"""

from __future__ import annotations

from typing import Any

# panel id, icon, short name
PANELS = (
    ("pipeline", "mdi-file-tree-outline", "Pipeline"),
    ("filters", "mdi-filter-plus-outline", "Add filter"),
    ("display", "mdi-palette-outline", "Display"),
    ("camera", "mdi-video-outline", "Camera"),
    ("info", "mdi-information-outline", "Information"),
)
DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")
MODE_LABELS = {"local": "Draw in browser", "remote": "Render on ParaView machine"}
MODE_ICONS = {"local": "mdi-monitor", "remote": "mdi-server"}

STYLE = """
html, body, #app { margin: 0; height: 100%; overflow: hidden; }
.v-application { font-size: 13px; }
.clio-root { display: flex; flex-direction: column; height: 100vh; }
.clio-bar { display: flex; flex-wrap: nowrap; gap: 4px 10px; align-items: center;
  min-height: 36px; padding: 2px 8px; border-bottom: 1px solid rgba(var(--v-border-color), .2); }
.clio-bar strong { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 40vw; }
.clio-time { display: inline-flex; align-items: center; gap: 2px; }
.clio-time input[type=range] { width: 140px; }
.clio-time .clio-step { font-variant-numeric: tabular-nums; min-width: 44px; padding: 0 6px; }
.clio-step-menu { display: flex; align-items: center; gap: 4px; padding: 4px 8px; }
.clio-step-menu input[type=range] { width: 180px; }
.clio-bar strong { flex: 0 1 auto; min-width: 0; }
.clio-time, .clio-mode { flex: none; }
.clio-mode { margin-left: auto; }
.clio-body { flex: 1; display: flex; min-height: 0; position: relative; }
.clio-rail { display: flex; flex-direction: column; gap: 2px; padding: 4px 2px; width: 40px;
  flex: none; border-right: 1px solid rgba(var(--v-border-color), .2); }
.clio-panel { width: 300px; flex: none; display: flex; flex-direction: column; min-height: 0;
  border-right: 1px solid rgba(var(--v-border-color), .2); background: rgb(var(--v-theme-surface)); }
.clio-panel header { display: flex; align-items: center; gap: 4px; padding: 2px 4px 2px 10px;
  font-weight: 600; min-height: 34px; }
.clio-panel header span { flex: 1; }
.clio-panel .clio-scroll { overflow: auto; padding: 0 10px 10px; flex: 1; min-height: 0; }
.clio-view { flex: 1; position: relative; min-width: 0; min-height: 120px; }
.v-btn { text-transform: none; letter-spacing: 0; }
.v-btn.clio-pressed { background: rgba(var(--v-theme-primary), .16); color: rgb(var(--v-theme-primary));
  border-radius: 6px; }
.clio-rail .v-btn.clio-pressed { box-shadow: inset 3px 0 0 rgb(var(--v-theme-primary)); }
.clio-row { display: flex; align-items: center; gap: 2px; min-height: 30px; border-radius: 4px; }
.clio-row .clio-name { flex: 1; text-align: left; justify-content: flex-start; text-transform: none;
  letter-spacing: 0; font-weight: 400; min-width: 0; }
.clio-row .clio-name .v-btn__content { overflow: hidden; text-overflow: ellipsis; }
.clio-row.clio-active { background: rgba(var(--v-theme-on-surface), .12); }
.clio-row.clio-active .clio-name { font-weight: 600; }
.clio-grid { display: grid; grid-template-columns: repeat(4, 40px); gap: 4px; margin: 4px 0 8px; }
.clio-form .clio-vec { display: flex; gap: 4px; align-items: center; }
.clio-form label.clio-label { display: block; font-size: 12px; opacity: .8; margin: 6px 0 2px; }
.clio-axes { display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; margin: 6px 0; }
.clio-axes .v-btn { min-width: 0; text-transform: none; letter-spacing: 0; }
.clio-info { width: 100%; border-collapse: collapse; font-size: 12px; }
.clio-info th { text-align: left; font-weight: 600; padding: 2px 6px 2px 0; }
.clio-info td { padding: 2px 6px 2px 0; font-variant-numeric: tabular-nums; }
.clio-tools { display: flex; gap: 4px; align-items: center; margin: 4px 0; }
.clio-panel .v-field { font-size: 13px; }
.clio-panel .v-field__input { min-height: 34px; padding-top: 6px; padding-bottom: 6px; }
.clio-panel .v-label { font-size: 12px; }
@media (max-width: 599px) {
  .clio-panel { position: absolute; left: 40px; right: 0; bottom: 0; width: auto;
    max-height: 62%; z-index: 5; border-right: 0; border-top: 1px solid rgba(var(--v-border-color), .3);
    border-radius: 10px 10px 0 0; box-shadow: 0 -4px 16px rgba(0,0,0,.25); }
  .clio-bar { gap: 2px 6px; }
  .clio-bar strong { max-width: none; }
  .clio-time input[type=range], .clio-time .clio-stepbtn { display: none; }
}
"""


def _titles_js() -> str:
    return "({" + ", ".join(f"{k}: '{label}'" for k, _, label in PANELS) + "})"


def _aria(label: str, extra: list[str] | None = None) -> list[str]:
    return [f'aria-label="{label}"', *(extra or [])]


def icon_button(
    v3: Any, icon: str, label: str, click: str, pressed: str | None = None, **kw: Any
) -> Any:
    """A compact icon button with a tooltip; ``pressed`` is a JS expression."""
    extra = [f":aria-pressed=\"({pressed}) ? 'true' : 'false'\""] if pressed else []
    if pressed:
        kw["classes"] = (f"{{'clio-pressed': {pressed}}}",)
    with v3.VBtn(
        icon=True,
        variant="text",
        density="comfortable",
        size="small",
        click=click,
        raw_attrs=_aria(label, extra),
        **kw,
    ) as button:
        v3.VIcon(icon, size="small")
        v3.VTooltip(activator="parent", text=label, location="bottom")
    return button


def build(server: Any, view_factory: Any) -> Any:
    """The whole page; ``view_factory()`` makes the 3D view inside it."""
    from trame.ui.vuetify3 import VAppLayout
    from trame.widgets import client, html
    from trame.widgets import vuetify3 as v3

    with VAppLayout(server, theme=("clio_theme", "light")) as layout:
        client.Style(STYLE)
        with html.Div(classes="clio-root"):
            _top_bar(v3, html)
            with html.Div(classes="clio-body"):
                _rail(v3, html)
                with html.Section(
                    classes="clio-panel",
                    v_if="clio_panel",
                    raw_attrs=[
                        'role="region"',
                        f':aria-label="{_titles_js()}[clio_panel]"',
                    ],
                ):
                    _panel(v3, html)
                with html.Div(classes="clio-view"):
                    view = view_factory()
    return layout, view


def _top_bar(v3: Any, html: Any) -> None:
    with html.Div(classes="clio-bar"):
        html.Strong("{{ clio_label }}", title=("clio_path",))
        with html.Span(
            classes="clio-time",
            v_if="clio_steps > 1",
            raw_attrs=['role="group"', 'aria-label="Time"'],
        ):
            icon_button(
                v3,
                "mdi-skip-previous",
                "Previous step",
                "trigger('clio_step', [-1])",
                disabled=("clio_index <= 0",),
                classes="clio-stepbtn",
            )
            icon_button(
                v3,
                "mdi-play",
                "Play",
                "trigger('clio_play', [true])",
                v_if="!clio_playing",
            )
            icon_button(
                v3,
                "mdi-pause",
                "Pause",
                "trigger('clio_play', [false])",
                v_if="clio_playing",
            )
            icon_button(
                v3,
                "mdi-skip-next",
                "Next step",
                "trigger('clio_step', [1])",
                disabled=("clio_index >= clio_steps - 1",),
                classes="clio-stepbtn",
            )
            html.Input(
                type="range",
                min=0,
                max=("clio_steps - 1",),
                value=("clio_index",),
                raw_attrs=['aria-label="Time step"'],
                change="trigger('clio_set_time', [Number($event.target.value)])",
            )
            _step_menu(v3, html)
            _follow(v3)
        with html.Span(
            classes="clio-mode", raw_attrs=['role="group"', 'aria-label="Drawing"']
        ):
            for mode, label in MODE_LABELS.items():
                tip = (
                    f"'{label}' + (viewMode === '{mode}' && clio_mode_note"
                    " ? '. ' + clio_mode_note : '')"
                )
                with v3.VBtn(
                    icon=True,
                    variant="text",
                    density="comfortable",
                    size="small",
                    click=f"trigger('clio_set_mode', ['{mode}'])",
                    classes=(f"{{'clio-pressed': viewMode === '{mode}'}}",),
                    raw_attrs=[
                        f'aria-label="{label}"',
                        f":aria-pressed=\"viewMode === '{mode}' ? 'true' : 'false'\"",
                    ],
                ):
                    v3.VIcon(MODE_ICONS[mode], size="small")
                    v3.VTooltip(activator="parent", text=(tip,), location="bottom")


def _step_menu(v3: Any, html: Any) -> None:
    """The step as ``n/N``; pressing it opens previous, a slider and next.

    At phone width this is the whole time control beside Play, so the bar keeps
    one row.
    """
    with v3.VBtn(
        "{{ clio_index + 1 }}/{{ clio_steps }}",
        variant="text",
        size="small",
        classes="clio-step",
        raw_attrs=[
            ":aria-label=\"'Time step ' + (clio_index + 1) + ' of ' + clio_steps\""
        ],
    ):
        with v3.VMenu(
            activator="parent", location="bottom", close_on_content_click=False
        ):
            with v3.VCard(classes="clio-step-menu", raw_attrs=['role="group"']):
                icon_button(
                    v3,
                    "mdi-skip-previous",
                    "Previous step",
                    "trigger('clio_step', [-1])",
                    disabled=("clio_index <= 0",),
                )
                html.Input(
                    type="range",
                    min=0,
                    max=("clio_steps - 1",),
                    value=("clio_index",),
                    raw_attrs=['aria-label="Time step"'],
                    change="trigger('clio_set_time', [Number($event.target.value)])",
                )
                icon_button(
                    v3,
                    "mdi-skip-next",
                    "Next step",
                    "trigger('clio_step', [1])",
                    disabled=("clio_index >= clio_steps - 1",),
                )


def _follow(v3: Any) -> Any:
    # Following the latest step is a pressed toggle; newer steps say so in the tooltip.
    label = "Follow latest"
    tip = "clio_newer > 0 && !clio_follow ? 'Follow latest (' + clio_newer + ' newer)' : 'Follow latest'"
    with v3.VBtn(
        icon=True,
        variant="text",
        density="comfortable",
        size="small",
        click="trigger('clio_latest')",
        classes=("{'clio-pressed': clio_follow}",),
        raw_attrs=[
            f'aria-label="{label}"',
            ":aria-pressed=\"clio_follow ? 'true' : 'false'\"",
        ],
    ) as button:
        v3.VIcon("mdi-page-last", size="small")
        v3.VBadge(
            v_if="clio_newer > 0 && !clio_follow",
            content=("clio_newer",),
            floating=True,
            raw_attrs=['aria-hidden="true"'],
        )
        v3.VTooltip(activator="parent", text=(tip,), location="bottom")
    return button


def _rail(v3: Any, html: Any) -> None:
    with html.Nav(
        classes="clio-rail",
        raw_attrs=[
            'role="toolbar"',
            'aria-orientation="vertical"',
            'aria-label="ParaView"',
        ],
    ):
        for key, icon, label in PANELS:
            with v3.VBtn(
                icon=True,
                variant="text",
                density="comfortable",
                size="small",
                click=f"clio_panel = clio_panel === '{key}' ? '' : '{key}'",
                classes=(f"{{'clio-pressed': clio_panel === '{key}'}}",),
                raw_attrs=[
                    f'aria-label="{label}"',
                    f":aria-pressed=\"clio_panel === '{key}' ? 'true' : 'false'\"",
                ],
            ):
                v3.VIcon(icon, size="small")
                v3.VTooltip(activator="parent", text=label, location="end")


def _panel(v3: Any, html: Any) -> None:
    titles = {key: label for key, _, label in PANELS}
    with html.Header():
        for key, label in titles.items():
            html.Span(label, v_if=f"clio_panel === '{key}'")
        icon_button(v3, "mdi-close", "Close panel", "clio_panel = ''")
    v3.VAlert(
        v_if="clio_error",
        text=("clio_error",),
        type="error",
        variant="tonal",
        density="compact",
        closable=True,
        icon="mdi-alert-circle-outline",
        classes="mx-2 mb-2",
        click_close="clio_error = ''",
        raw_attrs=['role="alert"'],
    )
    with html.Div(classes="clio-scroll"):
        with html.Div(v_if="clio_panel === 'pipeline'"):
            _pipeline(v3, html)
        with html.Div(v_if="clio_panel === 'filters'"):
            _filters(v3, html)
        with html.Div(v_if="clio_panel === 'display'"):
            _display(v3, html)
        with html.Div(v_if="clio_panel === 'camera'"):
            _camera(v3, html)
        with html.Div(v_if="clio_panel === 'info'"):
            _info(v3, html)


def _none_selected(html: Any) -> None:
    html.Div("Select an object in the pipeline.", v_if="!clio_sel", classes="py-2")


def _pipeline(v3: Any, html: Any) -> None:
    with html.Div(raw_attrs=['role="list"', 'aria-label="Pipeline"']):
        with html.Div(
            v_for="row in clio_objects",
            key="row.id",
            classes=("['clio-row', {'clio-active': row.id === clio_active}]",),
            style=("{paddingLeft: (row.depth * 14) + 'px'}",),
            raw_attrs=['role="listitem"'],
        ):
            with v3.VBtn(
                icon=True,
                variant="text",
                density="comfortable",
                size="small",
                click="trigger('clio_visible', [row.id, !row.visible])",
                raw_attrs=[
                    ":aria-label=\"(row.visible ? 'Hide ' : 'Show ') + row.label\"",
                    ":aria-pressed=\"row.visible ? 'true' : 'false'\"",
                ],
            ):
                v3.VIcon(
                    icon=("row.visible ? 'mdi-eye' : 'mdi-eye-off-outline'",),
                    size="small",
                )
                v3.VTooltip(
                    activator="parent",
                    text=("row.visible ? 'Hide' : 'Show'",),
                    location="end",
                )
            v3.VBtn(
                "{{ row.label }}",
                variant="text",
                density="comfortable",
                size="small",
                classes="clio-name",
                click="trigger('clio_select', [row.id])",
                raw_attrs=[
                    ":title=\"row.id + ' ' + row.type\"",
                    ":aria-current=\"row.id === clio_active ? 'true' : 'false'\"",
                ],
            )
            with v3.VBtn(
                icon=True,
                variant="text",
                density="comfortable",
                size="small",
                v_if="row.removable",
                click="trigger('clio_remove', [row.id])",
                raw_attrs=[":aria-label=\"'Remove ' + row.label\""],
            ):
                v3.VIcon("mdi-delete-outline", size="small")
                v3.VTooltip(activator="parent", text="Remove", location="end")


def _filters(v3: Any, html: Any) -> None:
    _none_selected(html)
    with html.Div(v_if="clio_sel"):
        html.Div("On {{ clio_sel.label }}", classes="text-caption")
        with html.Div(
            classes="clio-grid", raw_attrs=['role="group"', 'aria-label="Filters"']
        ):
            with html.Span(v_for="f in clio_offered", key="f.name"):
                with v3.VBtn(
                    icon=True,
                    variant="tonal",
                    density="comfortable",
                    size="small",
                    disabled=("f.disabled",),
                    click="trigger('clio_filter_form', [f.name])",
                    classes=("{'clio-pressed': clio_filter === f.name}",),
                    raw_attrs=[
                        ':aria-label="f.tip"',
                        ":aria-pressed=\"clio_filter === f.name ? 'true' : 'false'\"",
                    ],
                ):
                    v3.VIcon(icon=("f.icon",), size="small")
                    v3.VTooltip(activator="parent", text=("f.tip",), location="bottom")
        v3.VBtn(
            "Edit {{ clio_sel.label }}",
            v_if="clio_sel.editable && clio_editing !== clio_active",
            variant="tonal",
            size="small",
            prepend_icon="mdi-pencil",
            classes="mt-2",
            click="trigger('clio_filter_edit', [clio_active])",
        )
        with html.Form(
            v_if="clio_filter",
            classes="clio-form",
            submit="$event.preventDefault(); trigger('clio_filter_apply', [clio_form])",
            raw_attrs=[':aria-label="clio_form_title"'],
        ):
            html.Div("{{ clio_form_title }}", classes="font-weight-bold")
            with html.Div(v_for="f in clio_form_fields", key="f.key"):
                v3.VSelect(
                    v_if="f.kind === 'array' || f.kind === 'select'",
                    label=("f.label",),
                    items=("f.items",),
                    v_model=("clio_form[f.key]",),
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    classes="mt-2",
                )
                v3.VTextField(
                    v_if="f.kind === 'number' || f.kind === 'numbers' || f.kind === 'text'",
                    label=("f.label",),
                    v_model=("clio_form[f.key]",),
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    classes="mt-2",
                    raw_attrs=[":inputmode=\"f.kind === 'text' ? 'text' : 'decimal'\""],
                )
                v3.VSlider(
                    v_if="clio_editing && f.kind === 'numbers' && f.range",
                    min=("f.range[0]",),
                    max=("f.range[1]",),
                    step=("(f.range[1] - f.range[0]) / 200",),
                    model_value=("Number(String(clio_form[f.key]).split(/[ ,]+/)[0])",),
                    update_modelValue="clio_form[f.key] = String(Number($event.toPrecision(6)))",
                    end="trigger('clio_filter_apply', [clio_form])",
                    density="compact",
                    hide_details=True,
                    classes="mt-1",
                    raw_attrs=[":aria-label=\"f.label + ' slider'\""],
                )
                v3.VCheckbox(
                    v_if="f.kind === 'bool'",
                    label=("f.label",),
                    v_model=("clio_form[f.key]",),
                    density="compact",
                    hide_details=True,
                )
                with html.Div(v_if="f.kind === 'vector'"):
                    html.Label("{{ f.label }}", classes="clio-label")
                    with html.Div(classes="clio-vec"):
                        v3.VTextField(
                            v_for="(axis, i) in ['X', 'Y', 'Z']",
                            key="axis",
                            v_model=("clio_form[f.key][i]",),
                            density="compact",
                            hide_details=True,
                            variant="outlined",
                            raw_attrs=[
                                ":aria-label=\"f.label + ' ' + axis\"",
                                'inputmode="decimal"',
                            ],
                        )
                    with html.Div(v_if="f.axes", classes="clio-tools"):
                        v3.VBtn(
                            "{{ axis }}",
                            v_for="(axis, i) in ['X', 'Y', 'Z']",
                            key="axis",
                            size="x-small",
                            variant="tonal",
                            click="clio_form[f.key] = [0, 1, 2].map((j) => (j === i ? 1 : 0))",
                            raw_attrs=[":aria-label=\"f.label + ' along ' + axis\""],
                        )
            with html.Div(classes="clio-tools mt-3"):
                v3.VBtn(
                    "{{ clio_editing ? 'Update' : 'Apply' }}",
                    color="primary",
                    size="small",
                    prepend_icon="mdi-check",
                    click="trigger('clio_filter_apply', [clio_form])",
                )
                v3.VBtn(
                    "{{ clio_editing ? 'Done' : 'Cancel' }}",
                    variant="text",
                    size="small",
                    click="clio_filter = ''; clio_editing = null",
                )


def _display(v3: Any, html: Any) -> None:
    _none_selected(html)
    with html.Div(v_if="clio_sel"):
        html.Div("{{ clio_sel.label }}", classes="text-caption")
        v3.VSelect(
            label="Representation",
            items=("clio_shown_as",),
            model_value=("clio_sel.representation",),
            update_modelValue="trigger('clio_display', ['representation', $event])",
            density="compact",
            hide_details=True,
            variant="outlined",
            classes="mt-2",
        )
        v3.VSelect(
            label="Color by",
            items=("clio_sel.color_items",),
            model_value=("clio_sel.color",),
            update_modelValue="trigger('clio_display', ['color_by', $event])",
            density="compact",
            hide_details=True,
            variant="outlined",
            classes="mt-2",
        )
        v3.VSelect(
            v_if="clio_sel.component_items.length",
            label="Component",
            items=("clio_sel.component_items",),
            model_value=("clio_sel.component",),
            update_modelValue="trigger('clio_display', ['component', $event])",
            density="compact",
            hide_details=True,
            variant="outlined",
            classes="mt-2",
        )
        with html.Div(v_if="clio_sel.color", classes="clio-tools mt-2"):
            v3.VSelect(
                label="Color map",
                items=("clio_presets",),
                model_value=("clio_sel.colormap",),
                update_modelValue="trigger('clio_display', ['colormap', $event])",
                density="compact",
                hide_details=True,
                variant="outlined",
            )
            icon_button(
                v3,
                "mdi-arrow-expand-horizontal",
                "Rescale to data range",
                "trigger('clio_display', ['rescale', true])",
            )
            icon_button(
                v3,
                "mdi-palette-swatch-outline",
                "Color legend",
                "trigger('clio_display', ['scalar_bar', !clio_sel.scalar_bar])",
                pressed="clio_sel.scalar_bar",
            )
        html.Label("Opacity {{ clio_opacity.toFixed(2) }}", classes="clio-label")
        v3.VSlider(
            v_model=("clio_opacity",),
            min=0,
            max=1,
            step=0.05,
            density="compact",
            hide_details=True,
            end="trigger('clio_display', ['opacity', $event])",
            raw_attrs=['aria-label="Opacity"'],
        )


def _camera(v3: Any, html: Any) -> None:
    with html.Div(classes="clio-tools"):
        icon_button(
            v3,
            "mdi-fit-to-screen-outline",
            "Reset camera",
            "trigger('clio_camera', ['fit'])",
        )
        icon_button(
            v3,
            "mdi-perspective-less",
            "Parallel projection",
            "trigger('clio_camera', ['parallel', !clio_parallel])",
            pressed="clio_parallel",
        )
    with html.Div(
        classes="clio-axes", raw_attrs=['role="group"', 'aria-label="Look along"']
    ):
        for direction in DIRECTIONS:
            name = direction.upper().replace("-", "−")
            with v3.VBtn(
                name,
                variant="tonal",
                size="small",
                click=f"trigger('clio_camera', ['direction', '{direction}'])",
                raw_attrs=[f'aria-label="Look along {name}"'],
            ):
                v3.VTooltip(
                    activator="parent", text=f"Look along {name}", location="bottom"
                )


def _info(v3: Any, html: Any) -> None:
    _none_selected(html)
    with html.Div(v_if="clio_sel"):
        html.Div("{{ clio_sel.id }} {{ clio_sel.label }}", classes="font-weight-bold")
        with html.Table(classes="clio-info mt-1"):
            with html.Tr():
                html.Th("Type")
                html.Td("{{ clio_sel.data_type }}")
            with html.Tr():
                html.Th("Points")
                html.Td("{{ clio_sel.points }}")
            with html.Tr():
                html.Th("Cells")
                html.Td("{{ clio_sel.cells }}")
            with html.Tr():
                html.Th("Memory")
                html.Td("{{ clio_sel.memory }}")
            with html.Tr(v_for="b in clio_sel.bounds", key="b.axis"):
                html.Th("{{ b.axis }}")
                html.Td("{{ b.low }} to {{ b.high }}")
        with html.Table(classes="clio-info mt-2", v_if="clio_sel.arrays.length"):
            with html.Tr():
                html.Th("Field")
                html.Th("On")
                html.Th("Range")
            with html.Tr(v_for="a in clio_sel.arrays", key="a.where + a.name"):
                html.Td(
                    "{{ a.name }}{{ a.components > 1 ? ' (' + a.components + ')' : '' }}"
                )
                html.Td("{{ a.where }}")
                html.Td("{{ a.range }}")
        with html.Table(classes="clio-info mt-2", v_if="clio_sel.properties.length"):
            with html.Tr(v_for="p in clio_sel.properties", key="p.key"):
                html.Th("{{ p.key }}")
                html.Td("{{ p.value }}")
