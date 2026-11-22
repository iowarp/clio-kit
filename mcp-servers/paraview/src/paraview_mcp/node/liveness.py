"""Whether the pvserver a node half joined is still there.

ParaView does not notice a pvserver that has gone: renders go on returning, as
blank frames, or the client aborts inside the next one. The node half watches
the data server's socket instead, and stops drawing once the far end closes.
The socket is only peeked at; its flags are never changed, since ParaView
shares them.
"""

from __future__ import annotations

import os
import select
import socket
from typing import Any


def pvserver_socket() -> int | None:
    """The file descriptor of the joined data server's socket, if there is one."""
    try:
        from paraview import servermanager
        from paraview.modules.vtkRemotingCore import vtkPVSession

        session = servermanager.ActiveConnection.Session
        controller = session.GetController(vtkPVSession.DATA_SERVER)
        fd = int(controller.GetCommunicator().GetSocket().GetSocketDescriptor())
    except Exception:  # noqa: BLE001 - a build without them is not watched
        return None
    return fd if fd >= 0 else None


def closed(fd: int) -> bool:
    """Whether the far end of socket ``fd`` has closed it, looking without reading."""
    if os.name == "nt":
        probe = socket.socket(fileno=fd)
        try:
            ready, _, _ = select.select([probe], [], [], 0)
            if not ready:
                return False
            return probe.recv(1, socket.MSG_PEEK) == b""
        except BlockingIOError:
            return False
        except OSError:
            return True
        finally:
            probe.detach()
    try:
        probe = socket.socket(fileno=os.dup(fd))
    except OSError:
        return True
    try:
        return probe.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True
    finally:
        probe.close()


def watchable(fd: Any) -> bool:
    """Whether ``fd`` can be watched without changing it: no default socket timeout."""
    return isinstance(fd, int) and socket.getdefaulttimeout() is None
