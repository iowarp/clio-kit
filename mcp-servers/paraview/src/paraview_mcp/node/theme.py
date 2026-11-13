"""The view follows its host's look: light or dark, and the host's colours.

An MCP Apps host may send ``hostContext.theme`` and the standard CSS variables
in ``hostContext.styles.variables`` (``--color-background-primary``,
``--color-text-primary``, ``--font-sans``, ...). The view page passes them to
this page, which sets them on its root, gives Vuetify the same surface, text
and border colours, and tells the node half the render view's background: a
neutral between the host's background and text. A host that sends no
variables gets the fallbacks below; one that sends no theme gets the system's.

The background is how the view is drawn, not part of the scene: it is not in
the manifest and no command changes it.
"""

from __future__ import annotations

from typing import Any

# Fallbacks for a host that sends no variables, as the MCP Apps spec asks.
FALLBACK_CSS = """
:root {
  color-scheme: light dark;
  --color-background-primary: light-dark(#f7f7f5, #1e1f22);
  --color-text-primary: light-dark(#1f2328, #e6e8eb);
  --color-text-secondary: light-dark(#59606b, #a9b0ba);
  --color-border-primary: light-dark(#d4d6da, #3a3d44);
  --font-sans: system-ui, sans-serif;
}
"""

THEME_JS = r"""
(function () {
  var root = document.documentElement;
  var dark = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var host = { theme: null, variables: {} };
  var applied = {};
  var sheet = document.createElement('style');
  document.head.appendChild(sheet);
  var probe = document.createElement('span');
  probe.style.display = 'none';
  document.body.appendChild(probe);

  var paint = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  // Any CSS colour (light-dark(), oklch(), a hex) as sRGB bytes.
  function rgb(expression) {
    probe.style.color = '';
    probe.style.color = expression;
    var computed = getComputedStyle(probe).color;
    if (!computed || !paint) return null;
    paint.clearRect(0, 0, 1, 1);
    paint.fillStyle = '#000';
    paint.fillStyle = computed;
    paint.fillRect(0, 0, 1, 1);
    var px = paint.getImageData(0, 0, 1, 1).data;
    return px[3] ? [px[0], px[1], px[2]] : null;
  }
  function triplet(c) { return c.map(function (v) { return Math.round(v); }).join(','); }

  function apply() {
    var trame = window.trame;
    if (!trame || !trame.state || !trame.state.set) { setTimeout(apply, 50); return; }
    var theme = host.theme || (dark && dark.matches ? 'dark' : 'light');
    Object.keys(applied).forEach(function (k) {
      if (!(k in host.variables)) root.style.removeProperty(k);
    });
    applied = {};
    Object.keys(host.variables).forEach(function (k) {
      var v = host.variables[k];
      if (/^--[A-Za-z0-9-]+$/.test(k) && typeof v === 'string') {
        root.style.setProperty(k, v);
        applied[k] = true;
      }
    });
    root.style.colorScheme = theme;
    var bg = rgb('var(--color-background-primary)') || [247, 247, 245];
    var fg = rgb('var(--color-text-primary)') || [31, 35, 40];
    var border = rgb('var(--color-border-primary)') || fg;
    function given(name) { return host.variables[name] ? rgb('var(' + name + ')') : null; }
    var accent = given('--color-ring-primary');
    var tipBack = given('--color-background-inverse') || fg;
    var tipText = given('--color-text-inverse') || bg;
    var danger = given('--color-text-danger');
    var vars = '--v-theme-background:' + triplet(bg) + '!important;'
      + '--v-theme-surface:' + triplet(bg) + '!important;'
      + '--v-theme-on-background:' + triplet(fg) + '!important;'
      + '--v-theme-on-surface:' + triplet(fg) + '!important;'
      + '--v-theme-surface-variant:' + triplet(tipBack) + '!important;'
      + '--v-theme-on-surface-variant:' + triplet(tipText) + '!important;'
      + '--v-border-color:' + triplet(border) + '!important;'
      + (accent ? '--v-theme-primary:' + triplet(accent) + '!important;' : '')
      + (danger ? '--v-theme-error:' + triplet(danger) + '!important;' : '');
    sheet.textContent = '.v-application,.v-theme--light,.v-theme--dark{' + vars + '}'
      + '.v-application{background:rgb(' + triplet(bg) + ')!important;color:rgb(' + triplet(fg)
      + ')!important;font-family:var(--font-sans, system-ui, sans-serif)}';
    // The render view: a neutral a tenth of the way from the background to the text.
    var view = bg.map(function (b, i) { return (b + (fg[i] - b) * 0.1) / 255; });
    var ink = fg.map(function (v) { return v / 255; });
    if (trame.state.get('clio_theme') !== theme) trame.state.set('clio_theme', theme);
    var before = trame.state.get('clio_view_colors') || [[], []];
    var now = [view, ink];
    var moved = now.some(function (c, j) {
      return c.some(function (v, i) { return Math.abs(v - ((before[j] || [])[i] || 0)) > 0.002; });
    });
    if (moved) trame.state.set('clio_view_colors', now);
  }

  if (dark && dark.addEventListener) dark.addEventListener('change', apply);
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (e.source !== window.parent || !d || d.type !== 'clio.paraview.theme') return;
    host.theme = d.theme === 'dark' || d.theme === 'light' ? d.theme : null;
    host.variables = d.variables && typeof d.variables === 'object' ? d.variables : {};
    apply();
  });
  apply();
})();
"""


def _unit(color: Any) -> list[float] | None:
    try:
        rgb = [max(0.0, min(1.0, float(v))) for v in color][:3]
    except (TypeError, ValueError):
        return None
    return rgb if len(rgb) == 3 else None


def apply_to_view(
    simple: Any,
    view: Any,
    background: Any,
    foreground: Any,
    sources: list[Any] | None = None,
) -> bool:
    """Draw the render view on the theme's neutral, with lines and text in its text colour.

    ParaView's colour palette takes the colours, so objects shown later use them
    too; objects already shown in a solid colour (an outline, say) are given the
    new line colour. Returns whether anything was applied.
    """
    back, fore = _unit(background), _unit(foreground)
    if back is None or fore is None:
        return False
    palette = simple.GetSettingsProxy("ColorPalette")
    if palette is not None:
        names = palette.ListProperties()
        for name, value in (("Background", back), ("Foreground", fore), ("Text", fore)):
            if name in names:
                setattr(palette, name, value)
        if "BackgroundColorMode" in names:
            palette.BackgroundColorMode = "Single Color"
    names = view.ListProperties()
    if "UseColorPaletteForBackground" in names:
        view.UseColorPaletteForBackground = 0
    if "BackgroundColorMode" in names:
        view.BackgroundColorMode = "Single Color"
    view.Background = back
    if "OrientationAxesLabelColor" in names:
        view.OrientationAxesLabelColor = fore
    for source in sources or []:
        rep = simple.GetDisplayProperties(source, view)
        if "AmbientColor" in rep.ListProperties():
            rep.AmbientColor = fore
    return True
