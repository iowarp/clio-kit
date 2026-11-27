"""Attaching the front to a node half: prepare the folder, start or find it.

The node half lives in ``~/.clio/paraview/<server version>/`` on the machine
where ParaView runs:

- ``site/``: trame, trame-vtk and trame-vuetify, built for that machine's pvpython;
- ``code/``: the node half's code, copied over the same connection;
- ``node.json``: the running node half's pid, port and token (mode 0600);
  ``node-<namespace>.json`` when a server in another network namespace already
  runs one here;
- ``scenes/<id>/manifest.json``: each scene, written after every commit.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

from . import client, config, native
from .config import Settings
from .errors import PREFER_FORWARD, ParaViewError, no_paraview
from .link import Link, LinkError, NodeClient

# build_script, platform_tags, StartLook and wait_for_start live beside the setup
# steps now; attach still offers them to its callers.
from .node_setup import (
    build_script,  # noqa: F401
    code_bundle,
    local_folder,
    node_base,
    node_slot,
    parse_markers,
    platform_tags,  # noqa: F401
    probe_script,
)
from .preparing import NodePreparing
from .starting import StartLook, gone_error, wait_for_start  # noqa: F401
from .transport import Transport, Watch, transport_for

#: SSH failures that a forwarded pvserver endpoint would avoid.
_SSH_CODES = frozenset({"ssh_failed", "unreachable", "forward_failed"})


class Attacher(NodePreparing):
    """Keeps at most one link to a node half and re-attaches when it drops.

    Once linked, a node half that has gone with its pvserver (``pvserver_lost``)
    is reported, with the reason it left, instead of being replaced by an empty
    one. A node half that went while its pvserver still runs (it was killed, or
    crashed) is started again through the same start, keeping its token and
    ports, and its open scenes are opened again from their manifests, so a view
    the host serves reconnects to the same scenes. The retries are said in
    :meth:`status` and go on, backing off, until one starts, the pvserver is seen
    to have gone, or the person stops them.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.link: Link | None = None
        self.local_port: int | None = None
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._heart: threading.Thread | None = None
        #: ParaView runs on this Windows machine: steps run in Python, not a shell.
        self._native = False
        #: Set to end a node half's start that is being waited for.
        self.cancel = threading.Event()
        #: Told each new line a starting node half logs.
        self.progress: Callable[[str], None] | None = None
        #: The running call's stop and progress, while ``ensure`` serves it.
        self._call: tuple[threading.Event | None, Callable[[str], None] | None] = (
            None,
            None,
        )
        #: Whether this front has had a node half; one that goes is then reported.
        self.had_link = False
        #: The last scene opened through this attacher, for the lost-node advice.
        self.scene: str | None = None
        self.base = node_base(settings.pvserver)
        #: Each open scene's view ports, node-side and here, in opening order.
        self.views: dict[str, tuple[int | None, int | None]] = {}
        #: The local port each node-side port is forwarded to, across links.
        self.local_for: dict[int, int] = {}
        #: The token and port a node half started in place of a gone one keeps.
        self._reuse: dict[str, Any] | None = None
        #: While a gone node half is started again: its tries and what they said.
        self.relaunch: dict[str, Any] | None = None
        #: Set by the person to stop starting a gone node half again.
        self.relaunch_stop = threading.Event()
        self._relauncher: threading.Thread | None = None
        #: Why starting it again ended for good, once it did.
        self.ended: ParaViewError | None = None
        #: Scenes that came back on another port, or did not come back: their
        #: views must be opened again, and why.
        self.reopen: dict[str, str] = {}

    # -- public ----------------------------------------------------------------
    def ensure(
        self,
        opening: bool = False,
        stop: threading.Event | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Link:
        """A healthy link, attaching first if there is none.

        Unless ``opening``, a node half that was linked and has gone raises the
        reason it went, rather than starting an empty one in its place. A node half
        that is busy is waited for, with each wait told to ``progress``, until it
        answers or ``stop`` is set.
        """
        with self.lock:
            self._call = (stop, progress)
            try:
                link = self.link
                try:
                    if link is not None:
                        if link.transport.alive() and self._answers(link.client):
                            return link
                        if self._relaunchable():
                            return self._restart(link)
                        self._drop()
                    elif self.ended is not None and not opening:
                        raise self.ended
                    self.link = self._attach(opening)
                except ParaViewError as exc:
                    # A linked node half that went and could not be started again
                    # has ended: its live view says why rather than that none was
                    # ever open.
                    if link is not None and self.link is None and not opening:
                        self.ended = exc
                    raise
                self.had_link = True
                if opening:
                    self.ended = None
                    self.relaunch_stop.clear()
                self._start_heartbeat()
                return self.link
            finally:
                self._call = (None, None)

    def remember(self, scene: str, remote: Any, local: int | None) -> None:
        """Note a scene's view ports once open_view opened it, so a node half
        started in place of a gone one opens it again on the same ports."""
        try:
            port = int(remote)
        except (TypeError, ValueError):
            port = None
        self.views.pop(scene, None)
        self.views[scene] = (port, local)
        self.reopen.pop(scene, None)

    def stop_relaunch(self) -> bool:
        """Stop starting a gone node half again; whether a relaunch was going on."""
        going = self.relaunch is not None
        self.relaunch_stop.set()
        if going:
            self.cancel.set()
        return going

    def close(self) -> None:
        self._stop.set()
        with self.lock:
            self._drop()

    def status(self, scene: str | None = None) -> dict[str, Any]:
        """Whether the linked node half runs, without waiting on it.

        For a live view that cannot reach ParaView: ``running`` (it answers),
        ``busy`` (it holds its loop, or a start or check is under way),
        ``relaunching`` (it went while its pvserver runs, and is being started
        again), ``lost`` (it went, with why) or ``closed`` (no node half was
        linked). A ``scene`` whose view did not come back as it was is ``lost``
        with what to do.
        """
        relaunch = self.relaunch
        if relaunch is not None:
            return self._relaunching(relaunch)
        if not self.lock.acquire(blocking=False):
            return {"state": "busy", "text": "ParaView is starting or busy."}
        try:
            link = self.link
            if link is None:
                if self.ended is not None:
                    return {
                        "state": "lost",
                        "code": self.ended.code,
                        "text": self.ended.message,
                    }
                if self.had_link:
                    return {
                        "state": "lost",
                        "code": "node_lost",
                        "text": "ParaView's node half went and was not started again "
                        "(its pvserver has likely ended with its job). The scenes are "
                        "saved: open_view on a running pvserver opens them again.",
                    }
                return {"state": "closed", "text": "No ParaView node half is open."}
            try:
                health = link.client.health()
                if health.get("lost"):
                    gone = gone_error(
                        json.dumps({"reason": health["lost"]}),
                        "",
                        link.transport.where,
                        self.scene,
                    )
                    if self._relaunchable():
                        return self._relaunching(self._kick(gone.message, rejoin=True))
                    return {"state": "lost", "code": gone.code, "text": gone.message}
                if scene and scene in self.reopen:
                    return {
                        "state": "lost",
                        "code": "view_moved",
                        "text": self.reopen[scene],
                    }
                return {"state": "running", "text": "ParaView answers."}
            except LinkError as exc:
                if exc.busy:
                    return {
                        "state": "busy",
                        "text": "ParaView is busy with a long step, such as a "
                        "render or a measurement.",
                    }
            if link.transport.alive():
                try:
                    gone = self._gone(link.transport, link.folder, link.slot)
                except ParaViewError as exc:
                    gone = exc
            else:
                gone = gone_error("", "", link.transport.where, self.scene)
            if gone.code in _RELAUNCHED and self._relaunchable():
                rejoin = gone.code == "pvserver_lost"
                return self._relaunching(self._kick(gone.message, rejoin=rejoin))
            return {"state": "lost", "code": gone.code, "text": gone.message}
        finally:
            self.lock.release()

    def describe(self) -> dict[str, Any]:
        target = self.settings.target
        link = self.link
        out: dict[str, Any] = {
            "where": (
                f"on the ParaView server at {self.settings.pvserver}, through "
                f"a client {target.describe()}"
                if self.settings.pvserver
                else target.describe()
            ),
            "address": target.address,
            "via": target.via,
            "prepare": target.prepare,
            "pvserver": self.settings.pvserver,
            "attached": link is not None,
        }
        if self.settings.version:
            out["paraview_version"] = self.settings.version
            out["paraview_client"] = client.installed(self.settings.version)
        if link is not None:
            out.update(
                folder=link.folder,
                pvpython=link.pvpython,
                paraview=link.paraview,
                node_pid=link.node.get("pid"),
                pvserver_session=link.node.get("session"),
                environment_built_from=link.built_from,
                built_this_attach=link.built_now,
                code_copied_this_attach=link.uploaded_now,
                attach_seconds=link.timings,
            )
        return out

    # -- starting a gone node half again -----------------------------------------
    def _relaunchable(self) -> bool:
        """Whether a node half that went is started again: it showed scenes, and
        the person has not stopped that."""
        return bool(self.views) and not self.relaunch_stop.is_set()

    def _relaunching(self, relaunch: dict[str, Any]) -> dict[str, Any]:
        tries = int(relaunch.get("tries") or 0)
        text = (
            (
                "ParaView's connection to its pvserver closed, so it is joining the "
                "pvserver again with the same scenes"
                if relaunch.get("rejoin")
                else "ParaView's node half stopped while its pvserver still runs, so it "
                "is being started again with the same scenes"
            )
            + (f" (try {tries})" if tries else "")
            + "."
        )
        if relaunch.get("last"):
            text += f" Last: {relaunch['last']}"
        return {
            "state": "relaunching",
            "tries": tries,
            "since": relaunch.get("since"),
            "text": text,
        }

    def _kick(self, why: str, rejoin: bool = False) -> dict[str, Any]:
        """Start the background relaunch, once; its state. ``rejoin`` when the node
        half went because its pvserver connection closed: one join that fails then
        means the pvserver has ended."""
        if self.relaunch is not None and self._relauncher is not None:
            return self.relaunch
        relaunch: dict[str, Any] = {
            "tries": 0,
            "since": time.time(),
            "why": why,
            "rejoin": rejoin,
        }
        self.relaunch = relaunch
        self._relauncher = threading.Thread(
            target=self._relaunch_loop, args=(relaunch,), daemon=True
        )
        self._relauncher.start()
        return relaunch

    def rejoin(
        self,
        lost: ParaViewError,
        stop: threading.Event | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Link:
        """A node half joined to the pvserver again in place of one whose pvserver
        connection closed (``lost``), with its scenes, for a call to go on with.

        The path to a pvserver can drop while the pvserver runs on (a port forward
        rebuilt after its owner restarted). A join that fails means the pvserver
        has ended, and raises ``lost`` with what the join said.
        """
        with self.lock:
            if not self._relaunchable():
                raise lost
            self._call = (stop, progress)
            try:
                link = self.link
                if link is not None and self._healthy(link):
                    return link
                return self._restart(link, rejoin=True)
            finally:
                self._call = (None, None)

    def _healthy(self, link: Link) -> bool:
        """Whether ``link``'s node half answers and still has its pvserver."""
        if not (link.transport.alive() and self._answers(link.client)):
            return False
        try:
            return not link.client.health().get("lost")
        except LinkError:
            return False

    def _relaunch_loop(self, relaunch: dict[str, Any]) -> None:
        """Try to start the node half again until it starts, the pvserver is seen to
        have gone, or the person stops it. Each try is said; there is no limit on
        how many, only on a pvserver that takes no client for ``start_stall``."""
        wait = 2.0
        refused_since: float | None = None
        try:
            while not self._stop.is_set():
                if self.relaunch_stop.is_set():
                    self.ended = ParaViewError(
                        "relaunch_stopped",
                        "Starting ParaView's node half again was stopped. The scenes "
                        "are saved: open_view(scene=...) starts it and opens one again.",
                    )
                    return
                relaunch["tries"] = int(relaunch.get("tries") or 0) + 1

                def said(text: str) -> None:
                    relaunch["last"] = text

                try:
                    with self.lock:
                        self._call = (self.relaunch_stop, said)
                        try:
                            link = self.link
                            if link is None or not self._healthy(link):
                                self._restart(link, rejoin=bool(relaunch.get("rejoin")))
                        finally:
                            self._call = (None, None)
                    return
                except ParaViewError as exc:
                    if exc.code == "pvserver_lost":
                        self.ended = exc
                        return
                    if exc.code == "pvserver_join_failed":
                        refused_since = refused_since or time.monotonic()
                        quiet = time.monotonic() - refused_since
                        if quiet > self.settings.start_stall:
                            self.ended = ParaViewError(
                                "pvserver_lost",
                                f"ParaView's node half could not be started again: "
                                f"the pvserver at {self.settings.pvserver} has taken "
                                f"no client for {quiet:.0f} s, so it seems to have "
                                "ended (its job finished, say). The scenes are saved; "
                                "open_view on a running pvserver opens them again. "
                                + exc.message,
                            )
                            return
                    else:
                        refused_since = None
                    relaunch["last"] = exc.message
                except LinkError as exc:
                    relaunch["last"] = f"the new node half did not answer: {exc}"
                self.relaunch_stop.wait(wait)
                wait = min(30.0, wait * 2)
        finally:
            self.relaunch = None
            self._relauncher = None

    def _restart(self, old: Link | None, rejoin: bool = False) -> Link:
        """Start a node half in place of one that went, through the same start,
        keeping its token and ports, and open its scenes again. A node half that
        went with its pvserver raises why instead, unless ``rejoin``: then the new
        node half joins the pvserver again, and raises why only if that fails."""
        lost: ParaViewError | None = None
        if old is not None and old.transport.alive():
            gone = self._gone(old.transport, old.folder, old.slot)
            if gone.code == "pvserver_lost" and rejoin:
                lost = gone
            elif gone.code != "node_lost":
                raise gone
        if old is not None:
            self._reuse = {"token": old.token, "port": int(old.node.get("port") or 0)}
        self._drop()
        try:
            link = self._attach(opening=True)
        except ParaViewError as exc:
            if rejoin and exc.code in _JOIN_ENDED:
                said = (
                    lost.message
                    if lost is not None
                    else "ParaView's connection to its server closed."
                )
                raise ParaViewError(
                    "pvserver_lost",
                    f"{said} ParaView tried to reconnect and could not, so it has "
                    f"ended. {exc.message}",
                ) from exc
            raise
        self.link = link
        self.had_link = True
        self._start_heartbeat()
        self._restore(link, old.token if old is not None else None)
        return link

    def _restore(self, link: Link, token: str | None) -> None:
        """Open each scene the gone node half showed, on the port its view had."""
        stop = self._call[0]
        same_token = token is None or link.token == token
        # The scene the agent's tools act on opens last, so it stays current.
        for scene in sorted(self.views, key=lambda s: s == self.scene):
            remote, local = self.views[scene]
            args: dict[str, Any] = {"scene": scene}
            if remote:
                args["view_port"] = remote
            try:
                result = link.client.command("open_view", args, stop=stop)
            except ParaViewError as exc:
                if exc.code == "cancelled":
                    raise
                self.reopen[scene] = (
                    f"ParaView started again after its node half stopped, but scene "
                    f"{scene} did not open: {exc.message}"
                )
                continue
            now = ((result or {}).get("service") or {}).get("port")
            here = link.port_for(now)
            self.views[scene] = (int(now) if now else None, here)
            if same_token and (local is None or here == local):
                self.reopen.pop(scene, None)
            else:
                self.reopen[scene] = (
                    f"ParaView started again after its node half stopped, and scene "
                    f"{scene} is open again, but its view moved to another port: "
                    f"open_view(scene={scene!r}) shows it again."
                )

    # -- attaching ---------------------------------------------------------------
    def _drop(self) -> None:
        if self.link is not None:
            self.link.transport.close()
            self.link = None

    def _attach(self, opening: bool = True) -> Link:
        started = time.monotonic()
        settings = self.settings
        target = settings.target
        transport = transport_for(target, settings.connect_timeout)
        transport.watch = self._watch()
        try:
            transport.open()
            return self._attach_over(transport, started, opening)
        except ParaViewError as exc:
            transport.close()
            if transport.remote and settings.pvserver and exc.code in _SSH_CODES:
                raise ParaViewError(exc.code, exc.message + PREFER_FORWARD) from exc
            raise
        except BaseException:
            transport.close()
            raise

    def _attach_over(
        self, transport: Transport, started: float, opening: bool = True
    ) -> Link:
        settings = self.settings
        base = self.base
        timings: dict[str, float] = {"connect": time.monotonic() - started}
        folder_here = None if transport.remote else local_folder(config.SERVER_VERSION)
        self._native = not transport.remote and native.wanted()
        probe_err = ""
        if self._native:
            marks = native.probe(str(folder_here), base)
        else:
            probe = transport.run(
                probe_script(config.SERVER_VERSION, folder_here, base),
                label="Checking the ParaView folder",
            )
            marks = parse_markers(probe.stdout)
            probe_err = probe.stderr.strip() or probe.stdout.strip()
        folder = marks.get("DIR")
        pvpython = marks.get("PVPYTHON") or ""
        if not folder:
            raise ParaViewError(
                "probe_failed",
                f"Could not prepare the ParaView folder {transport.where}: {probe_err}",
            )
        env = json.loads(marks["ENV"]) if marks.get("ENV") else None
        if not transport.remote and settings.version:
            pvpython = self._local_pvpython(pvpython, env)
        if not pvpython:
            detail = ""
            if settings.target.prepare and probe_err:
                detail = f"The prepare line said: {probe_err[-300:]}"
            fetchable = sorted(v for v, m in client.BUILDS if m == client.machine())
            if not transport.remote and not settings.version and fetchable:
                detail += (
                    f"{' ' if detail else ''}To fetch Kitware's ParaView to this "
                    f"machine instead, set `version` ('ParaView version') to "
                    f"{' or '.join(fetchable)}; it must match any pvserver it joins."
                )
            raise no_paraview(transport.where, detail)
        if env is None or env.get("pvpython") != pvpython:
            env = self._interrogate(transport, pvpython)
        if settings.version and env.get("paraview") != settings.version:
            raise ParaViewError(
                "paraview_version_mismatch",
                f"The ParaView {transport.where} is {env.get('paraview')}, and "
                f"{settings.version} is wanted: a ParaView joins only a server of its "
                "own version.",
            )
        # A prepare line can bring its own GL libraries, so the probe result is
        # kept per prepare line.
        gl_for = settings.target.prepare or ""
        if "gl_args" not in env or env.get("gl_for") != gl_for:
            env["gl_args"] = self._gl_args(transport, pvpython)
            env["gl_for"] = gl_for
            if env.get("built"):
                self._write_env(transport, folder, env)
        gl_args = [str(a) for a in env.get("gl_args") or []]
        built_from = None
        if not env.get("built"):
            built_from = self._build(transport, folder, env, pvpython)
            env["built"] = built_from
            self._write_env(transport, folder, env)
        timings["prepare"] = time.monotonic() - started
        bundle, code_hash = code_bundle()
        uploaded = marks.get("CODE") != code_hash
        if uploaded:
            self._upload(transport, folder, bundle)
        slot, node = node_slot(marks, base)
        # The rendering switch rides on the node's hash, so a node half started
        # without it is replaced rather than reused.
        node_hash = "+".join([code_hash, *(a.lstrip("-") for a in gl_args)])
        if settings.pvserver:
            # A node half joins one pvserver for its life; another one needs a new node.
            node_hash += "+pv=" + settings.pvserver
        link = None
        if node is not None and node.get("code_hash") == node_hash:
            # Another front with other knobs (one a connector change superseded,
            # say) may still show views through it: share it rather than stop it.
            # Its knobs hold until it ends; the next node half takes these.
            link = self._try_link(transport, folder, node, pvpython, env)
        if link is None and not opening and self.had_link:
            if node is not None and self._matches(node, node_hash):
                waiting = (
                    f" It joined the pvserver at {settings.pvserver}: if that has "
                    "ended (its job finished, say), the node half may be waiting on "
                    "it, and open_view with a running pvserver starts a new one."
                    if settings.pvserver
                    else ""
                )
                raise ParaViewError(
                    "node_unreachable",
                    f"The ParaView node half {transport.where} is still running but "
                    "did not answer, so nothing was done. It may be busy with a "
                    "long render; retry, or call open_view to open the view again."
                    + waiting,
                )
            # The node half this front was linked to has gone: say why, rather
            # than show an empty scene in its place.
            raise self._gone(transport, folder, slot)
        if link is None:
            if node is not None:
                freed = self._stop_node(transport, folder, node, slot)
                if not freed and slot == base:
                    # Another sandbox's node half would not let go: run beside it.
                    slot = native.own_slot(base)
            node = self._start_node(
                transport, folder, pvpython, node_hash, gl_args, slot
            )
            link = self._try_link(transport, folder, node, pvpython, env)
        if link is None:
            raise ParaViewError(
                "node_unhealthy",
                f"The ParaView node half started {transport.where} but does not "
                f"answer. Its log is {folder}/{slot}.log.",
            )
        link.slot = slot
        link.built_from = built_from or env.get("built")
        link.built_now = built_from is not None
        link.uploaded_now = uploaded
        timings["total"] = time.monotonic() - started
        link.timings = {k: round(v, 2) for k, v in timings.items()}
        return link

    def _matches(self, node: dict[str, Any], code_hash: str) -> bool:
        """Whether a running node half was started with today's code and knobs."""
        settings = self.settings
        if node.get("code_hash") != code_hash:
            return False
        if node.get("idle_seconds") not in (None, settings.idle_seconds):
            return False
        threshold = settings.threshold_mb
        return threshold is None or node.get("threshold_mb") == threshold

    def _try_link(
        self,
        transport: Transport,
        folder: str,
        node: dict[str, Any],
        pvpython: str,
        env: dict[str, Any],
    ) -> Link | None:
        try:
            port = transport.forward(int(node["port"]), self.local_port)
        except ParaViewError:
            return None
        client = NodeClient(port, str(node["token"]))
        if not self._answers(client):
            return None
        self.local_port = port
        return Link(
            transport=transport,
            folder=folder,
            local_port=port,
            client=client,
            node=node,
            pvpython=pvpython,
            paraview=str(env.get("paraview", "")),
            local_for=self.local_for,
        )

    def _answers(self, client: NodeClient) -> bool:
        """Whether the node half answers, waiting while it is only busy.

        A software volume render holds the node's loop for tens of seconds. A health
        check that times out then means "busy", not "gone": restarting it would lose
        the open scene. So a busy node half is waited for as long as it stays busy,
        and the wait is said; only a refused link fails. The person ends the wait by
        cancelling the call.
        """
        stop, said = self._call
        began = time.monotonic()
        while True:
            try:
                client.health()
                return True
            except LinkError as exc:
                if not exc.busy:
                    return False
            if self.cancel.is_set() or (stop is not None and stop.is_set()):
                raise ParaViewError(
                    "cancelled",
                    "The call was cancelled while ParaView was busy; nothing was done.",
                )
            tell = said or self.progress
            if tell is not None:
                tell(
                    f"ParaView has not answered for {time.monotonic() - began:.0f} s: "
                    "it is busy (a long render holds it), not gone, so this waits"
                )

    def _watch(self) -> Watch:
        """How setup steps are watched: the running call's cancel and progress."""

        def cancelled() -> bool:
            stop = self._call[0]
            return self.cancel.is_set() or (stop is not None and stop.is_set())

        def tell(text: str) -> None:
            said = self._call[1] or self.progress
            if said is not None:
                said(text)

        return Watch(self.settings.setup_stall, cancelled, tell)

    # -- keeping the link warm ----------------------------------------------------
    def _start_heartbeat(self) -> None:
        if self._heart is not None and self._heart.is_alive():
            return
        self._heart = threading.Thread(target=self._beat, daemon=True)
        self._heart.start()

    def _beat(self) -> None:
        """Keep the link warm, and see a node half that went within a beat, so it
        is started again even while no view or tool asks."""
        while not self._stop.wait(BEAT):
            link = self.link
            if link is None or self.relaunch is not None:
                continue
            try:
                link.client.ping()
            except LinkError as exc:
                if exc.busy or not self._relaunchable():
                    continue
                # status() reads why it went and starts it again when it may.
                self.status()


#: Seconds between the heartbeat's pings.
BEAT = 10.0

#: Why a node half went that has it started again with its scenes.
_RELAUNCHED = ("node_lost", "pvserver_lost")

#: How joining a pvserver again fails when the pvserver has ended.
_JOIN_ENDED = ("pvserver_join_failed", "pvserver_join_stalled", "pvserver_lost")
