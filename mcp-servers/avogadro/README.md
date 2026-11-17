# Avogadro MCP

[Avogadro 2](https://two.avogadro.cc/) in a view, for building and remodelling structures
in 3D: fragments, element changes, optimizing while dragging, crystals, surfaces and
supercells. What the person builds and selects comes back to the agent as files.

Like the KiCad connector, this server **finds and starts** the application; it never
installs it. The agent installs Avogadro with the person's consent through the avogadro
plugin's `install-avogadro` skill (the official 2.1.0 Linux AppImage, checksum checked) and
points the connector's **Where Avogadro is** (`AVOGADRO_COMMAND`) at it.

## How it runs (Linux)

`open_in_avogadro` starts, once per server process:

1. **Xvfb**, a virtual display (`-displayfd` picks a free number; 1440×900).
2. **Avogadro** on it, with its own `TMPDIR` (its RPC socket is `$TMPDIR/avogadro`, so two
   sessions never share one), its own data and config folders, software OpenGL, its window
   sized to the screen (`-qwindowgeometry`), and its working folder set to
   `<workspace>/chemistry/avogadro/`, so its Save dialog starts in the workspace.
3. **x11vnc** on that display, loopback only.
4. A **loopback service** (Starlette on 127.0.0.1, a free port) that refuses any request
   without the session's bearer token. It serves the stream page, the vendored noVNC, `state`
   (what the person last sent) and a WebSocket bridged to x11vnc. The opening result declares
   it as `_meta["ai.iowarp.clio/service"] = {port, token}`; CLIO proxies it under the app's
   route and adds the token, so the view and the model never see the port or token.

The view (`ui://avogadro/view`) frames the stream page and polls `state`. When the person
sends something, it reports it with `ui/update-model-context` (formula, atom count, the
selected atoms by 0-based index and element). Each process runs in its own process group;
`close_avogadro` and server exit stop them and delete the session folder.

### Agent → Avogadro

Avogadro's RPC listener (MoleQueue JSON-RPC 2.0; 4-byte big-endian length framing):
`openFile` opens a structure file, `saveGraphic` writes the PNG `avogadro_snapshot` returns.

### Avogadro → agent: "Send to CLIO"

The server writes an Avogadro Python command into the session
(`<data>/OpenChemistry/Avogadro/commands/send_to_clio.py`), shown as **Extensions > CLIO >
Send to CLIO**. Avogadro runs it with the molecule as CJSON on stdin. It writes that CJSON,
including `atoms.selected`, into the session's handoff folder and returns the molecule
unchanged: when a command's reply has no `cjson`, Avogadro replaces the molecule with an
empty one. `read_avogadro` writes the last one sent as XYZ (and keeps the CJSON) under
`chemistry/avogadro/` and reports the selected atoms.

## Tools

| Tool | What it does |
| --- | --- |
| `avogadro_status()` | Whether Avogadro (and where), Xvfb and x11vnc are here; what to install when not; whether it is running. |
| `open_in_avogadro(path?)` | Start (first time) and open a structure file, or an empty molecule; opens the view. |
| `read_avogadro(output_dir?)` | What the person last sent: XYZ + CJSON files, formula, selected atoms. |
| `avogadro_snapshot(output_dir?)` | A PNG of Avogadro's 3D view, saved and returned as an image. |
| `close_avogadro()` | Stop Avogadro, the display, VNC and the service. |

Errors end with `[reason]`: `not_found`, `not_installed` (says to use install-avogadro),
`unsupported_platform` (not Linux x86-64), `display_missing` (names the packages),
`start_failed` (with the program's last output), `not_running`, `nothing_sent`, `rpc_failed`.

## Needs

Linux x86-64 with `Xvfb` and `x11vnc` (system packages) and Avogadro 2: `AVOGADRO_COMMAND`
(an `avogadro2` or an unpacked AppImage's `AppRun`) or `avogadro2` on PATH. macOS and Windows
are refused with a plain reason: the streamed view needs an X display.

## Tests

```sh
uv run --frozen pytest -m "not integration"   # offline
uv run --frozen pytest -m integration         # Xvfb, x11vnc and avogadro2 present
```

Checked here (Linux x86-64) with Ubuntu's `avogadro` 1.99.0 package (an Avogadro 2
pre-release), used only to work out these semantics: the integration test (start, the token
refused and accepted, the VNC handshake through the WebSocket bridge, an RPC snapshot, a
handoff read back with its selection, a clean stop), and in headless Chromium the stream
page connected through the service, Select > Select All and Extensions > CLIO > Send to CLIO
clicked through the stream, and `state` reporting the molecule with all its atoms selected.
Avogadro 2.1.0, installed by CLIO with the `install-avogadro` skill, streams in a CLIO
conversation with a structure loaded, and its RPC snapshots render. Running it under CLIO
found three things the 1.99 checks did not: 2.1 waits on a first-run dialog before it starts
the RPC listener (the server passes `--skip-dialogs`), a sandbox's deep TMPDIR can put the
RPC socket path over the 107-byte Unix limit (the socket folder then goes under /tmp), and
noVNC's `vendor/pako/lib` must be shipped (the repository ignores `lib/` folders). An edit
sent back with Send to CLIO from 2.1 has not been checked yet.

## NOTICE

`src/avogadro_mcp/novnc/` is noVNC v1.5.0 (git commit
`7fcf9dcfe0cc5b14e3841a4429dc091a6ffca861`), `core/` and `vendor/pako/` unchanged: noVNC core
MPL-2.0, pako MIT (`novnc/LICENSE.txt`, `novnc/vendor/pako/LICENSE`). Avogadro is
BSD-3-Clause (OpenChemistry) and is not distributed here. `view.html` follows the paraview
server's view.
