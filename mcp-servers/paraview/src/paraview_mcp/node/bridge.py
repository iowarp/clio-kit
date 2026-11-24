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
    // On a closed link trame swaps its page for a "Connection closed" notice on the
    // next render, which would hide the last picture the view promises to keep. The
    // close runs before that render, so the picture is copied out of trame's page
    // first and laid over the notice; a reconnect reloads the page and drops it.
    var lastPicture = null;
    function keepPicture() {
      if (document.getElementById('clio-kept-picture')) return;
      var source = lastPicture && lastPicture.isConnected ? lastPicture
        : document.querySelector('.clio-view canvas');
      if (!source) return;
      var rect = source.getBoundingClientRect();
      if (rect.width < 10 || rect.height < 10) return;
      var copy = document.createElement('canvas');
      // At the canvas's own pixel size: its picture may be letterboxed in its box,
      // so it is fitted the same way rather than stretched to the box.
      copy.width = source.width || Math.round(rect.width * (window.devicePixelRatio || 1));
      copy.height = source.height || Math.round(rect.height * (window.devicePixelRatio || 1));
      try { copy.getContext('2d').drawImage(source, 0, 0, copy.width, copy.height); } catch (e) { return; }
      copy.style.cssText = 'position:absolute;left:' + rect.left + 'px;top:' + rect.top + 'px;width:'
        + rect.width + 'px;height:' + rect.height + 'px;object-fit:contain';
      var cover = document.createElement('div');
      cover.id = 'clio-kept-picture';
      cover.style.cssText = 'position:fixed;inset:0;z-index:2147483647;background:'
        + (getComputedStyle(document.body).backgroundColor || 'transparent');
      cover.appendChild(copy);
      document.body.appendChild(cover);
      // The view page keeps it too: a reload of this page, or a page that cannot
      // load once ParaView is gone, would drop the copy laid over it here.
      try {
        post({ type: 'clio.paraview.picture', url: copy.toDataURL('image/png'),
          left: rect.left, top: rect.top, width: rect.width, height: rect.height,
          background: cover.style.background });
      } catch (e) { /* a picture that cannot be read stays in this page only */ }
    }
    function linkDown() {
      keepPicture();
      post({ type: 'clio.paraview.link', up: false });
    }
    var client = trame.client;
    if (client && client.onConnectionClose) {
      client.onConnectionClose(linkDown);
      client.onConnectionError(linkDown);
    }
    post({ type: 'clio.paraview.link', up: true });
    // The page drew a scene it was sent, or could not (vtkjs_patch.py): the node
    // half sends the next one then.
    window.addEventListener('vtk-scene-loaded', function (e) {
      var detail = (e && e.detail) || {};
      trame.trigger('clio_scene_loaded', [!!detail.ok, String(detail.error || '').slice(0, 300)]);
    });
    // The view page answers with the host's theme and colours (theme.py).
    post({ type: 'clio.paraview.ready' });
    // A frame that mounts or resizes (a tab moved, the canvas maximized, the host
    // remounted the app) asks for a fresh picture at its own size: in remote mode
    // nothing else would draw it until the next drag. The images are shared by every
    // page of the view, and a page that went mid-drag or at another size leaves that
    // shared state behind, so the ask carries this page's size and says whether it has
    // a picture yet. Until a full-size picture shows, it asks again, backing off.
    var drawnAt = '';
    var shown = false;
    var redraw = null;
    var again = null;
    var tries = 0;
    function boxSize() {
      var box = document.querySelector('.clio-view');
      var r = box ? box.getBoundingClientRect() : null;
      return r ? [Math.round(r.width), Math.round(r.height)] : [0, 0];
    }
    function send() {
      var size = boxSize();
      if (size[0] < 10 || size[1] < 10 || document.hidden) return;
      if (trame.state.get && trame.state.get('viewMode') !== 'remote') return;
      if (trame.state.get && trame.state.get('clio_lost')) return;
      trame.trigger('clio_redraw', [size[0], size[1], !shown]);
      clearTimeout(again);
      again = setTimeout(function () {
        if (shown) return;
        tries += 1;
        send();
      }, Math.min(8000, 1000 * 2 ** tries));
    }
    function ask() {
      clearTimeout(redraw);
      shown = false;
      tries = 0;
      redraw = setTimeout(send, 300);
    }
    ask();
    window.addEventListener('resize', ask);
    document.addEventListener('visibilitychange', ask);
    trame.state.watch(['viewMode'], ask);
    (function watch() {
      var box = document.querySelector('.clio-view');
      if (!box || !window.ResizeObserver) { setTimeout(watch, 200); return; }
      new ResizeObserver(ask).observe(box);
    })();
    // A remote picture can land well after the page starts (a pvserver over a
    // forwarded link), and again after each resize; say when a full-size one shows
    // for the current size, so a host that looked early and saw nothing looks again.
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
        shown = true;
        lastPicture = img;
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
      if (!text) return;
      keepPicture();
      post({ type: 'clio.paraview.lost', text: text });
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
