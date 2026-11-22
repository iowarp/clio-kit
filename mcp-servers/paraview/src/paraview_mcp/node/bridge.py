"""The page side of a view: the script trame's page runs inside the frame that
hosts it, and the token a request to the node half must carry."""

from __future__ import annotations

from typing import Any

TOKEN_HEADERS = ("X-Clio-Service-Token",)

# Posts the bridge state and the link state to the page that frames trame.
BRIDGE_JS = r"""
(function () {
  function post(message) {
    try { window.parent.postMessage(message, '*'); } catch (e) { /* no parent */ }
  }
  var down = null;
  // A click that did not move is a pick; ParaView finds what is under it.
  document.addEventListener('pointerdown', function (e) {
    down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
  }, true);
  document.addEventListener('pointerup', function (e) {
    var start = down;
    down = null;
    if (!start || !window.trame || !e.target.closest || !e.target.closest('.clio-view')) return;
    if (Math.abs(e.clientX - start.x) + Math.abs(e.clientY - start.y) > 4) return;
    var canvas = Array.prototype.find.call(document.querySelectorAll('.clio-view canvas'), function (c) {
      var r = c.getBoundingClientRect();
      return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    });
    if (!canvas) return;
    var box = canvas.getBoundingClientRect();
    var camera = null;
    try { camera = window.trame.refs.view.getCamera(); } catch (err) { camera = null; }
    window.trame.trigger('clio_pick', [e.clientX - box.left, box.height - (e.clientY - box.top),
      box.width, box.height, camera]);
  }, true);
  function start() {
    var trame = window.trame;
    if (!trame || !trame.state || !trame.state.watch) { setTimeout(start, 50); return; }
    trame.state.watch(['clio_bridge'], function (bridge) {
      if (bridge) post({ type: 'clio.paraview.state', state: bridge });
    });
    var client = trame.client;
    if (client && client.onConnectionClose) {
      client.onConnectionClose(function () { post({ type: 'clio.paraview.link', up: false }); });
      client.onConnectionError(function () { post({ type: 'clio.paraview.link', up: false }); });
    }
    post({ type: 'clio.paraview.link', up: true });
    // The view page answers with the host's theme and colours (theme.py).
    post({ type: 'clio.paraview.ready' });
    // A frame that mounts or resizes (a tab moved, the canvas maximized) asks for a
    // fresh picture: in remote mode nothing else would draw it until the next drag.
    // The remote view subscribes to images only once it has mounted and sized itself,
    // so ask again when the view's box changes and a little after the page starts.
    var redraw = null;
    function ask() {
      clearTimeout(redraw);
      redraw = setTimeout(function () { trame.trigger('clio_redraw'); }, 500);
    }
    [300, 1500, 4000].forEach(function (ms) { setTimeout(ask, ms); });
    window.addEventListener('resize', ask);
    (function watch() {
      var box = document.querySelector('.clio-view');
      if (!box || !window.ResizeObserver) { setTimeout(watch, 200); return; }
      new ResizeObserver(ask).observe(box);
    })();
    // A remote picture can land well after the page starts (a pvserver over a
    // forwarded link), and again after each resize; say when a full-size one shows
    // for the current size, so a host that looked early and saw nothing looks again.
    var drawnAt = '';
    var toldAt = 0;
    var untold = 0;
    (function watchPicture() {
      var box = document.querySelector('.clio-view');
      if (!box) { setTimeout(watchPicture, 200); return; }
      box.addEventListener('load', function (e) {
        var img = e.target;
        if (!img || img.tagName !== 'IMG') return;
        var rect = img.getBoundingClientRect();
        var size = Math.round(rect.width) + 'x' + Math.round(rect.height);
        if (rect.width < 1 || img.naturalWidth < rect.width * 0.9) return;
        // The node half counts the pictures its viewers really show, told at
        // most every two seconds while a drag streams them.
        untold += 1;
        var now = Date.now();
        if (size !== drawnAt || now - toldAt > 2000) {
          trame.trigger('clio_drawn', [size, untold]);
          toldAt = now;
          untold = 0;
        }
        if (size === drawnAt) return;
        drawnAt = size;
        post({ type: 'clio.paraview.drawn' });
      }, true);
    })();
    // The pvserver went: say so in place of the picture, and stop reconnecting.
    trame.state.watch(['clio_lost'], function (text) {
      if (text) post({ type: 'clio.paraview.lost', text: text });
    });
    if (trame.state.get && trame.state.get('clio_lost')) {
      post({ type: 'clio.paraview.lost', text: trame.state.get('clio_lost') });
    }
  }
  start();
})();
"""


def token_from(request: Any) -> str | None:
    """The token a request carries, from the bearer header or the CLIO header."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    for name in TOKEN_HEADERS:
        if request.headers.get(name):
            return request.headers[name].strip()
    return None
