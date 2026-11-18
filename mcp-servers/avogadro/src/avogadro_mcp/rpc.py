"""Avogadro's RPC listener: MoleQueue JSON-RPC 2.0 over a local socket.

Each message is a QDataStream ``QByteArray``: a 4-byte big-endian length, then the JSON.
The socket is ``$TMPDIR/avogadro``, so each session gives Avogadro its own TMPDIR.
Methods used: ``openFile`` {fileName}, ``loadMolecule`` {format, content},
``saveGraphic`` {fileName}.
"""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any

from .errors import AvogadroError


def call(sock_path: Path, method: str, params: dict[str, Any], timeout: float = 30) -> Any:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(timeout)
            s.connect(str(sock_path))
            s.sendall(struct.pack(">I", len(body)) + body)
            data = b""
            while len(data) < 4 or len(data) < 4 + struct.unpack(">I", data[:4])[0]:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
    except OSError as exc:
        raise AvogadroError("rpc_failed", f"Avogadro did not answer {method}: {exc}.") from exc
    if len(data) < 4:
        raise AvogadroError("rpc_failed", f"Avogadro sent no reply to {method}.")
    reply = json.loads(data[4:].decode(errors="replace"))
    if "error" in reply:
        raise AvogadroError("rpc_failed", f"Avogadro refused {method}: {reply['error']}.")
    return reply.get("result")
