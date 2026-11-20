"""HTTP connections that reach only public addresses.

A URL from the model can name an internal service: by a literal private
address, by another spelling of one (``127.1``, ``2130706433``,
``0x7f000001``), or by a DNS name that resolves to one, possibly only on the
second lookup (DNS rebinding). Checking the URL's text is not enough, so the
check runs where the connection is made: :class:`PublicOnlyBackend` resolves
the host, refuses it if any address is loopback, private, link-local,
reserved, multicast or unspecified, and connects to the address it checked.
TLS still verifies the certificate against the host name.

With an HTTP proxy configured in the environment, httpx connects to the proxy
instead, and the proxy resolves the host. A caller that wants a check in that
case too can call :func:`assert_public_url` before each request.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import httpcore
import httpx


class BlockedHostError(Exception):
    """A request named a host that resolves to a non-public address."""


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return not (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def public_addresses(host: str, port: int) -> list[str]:
    """Resolve ``host`` and return its addresses, if every one is public.

    Raises:
        BlockedHostError: if the host does not resolve or any address is not public.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise BlockedHostError(f"{host!r} does not resolve: {exc}") from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    blocked = [address for address in addresses if not _is_public(address)]
    if not addresses or blocked:
        raise BlockedHostError(
            f"{host!r} resolves to a loopback, private or link-local address "
            f"({', '.join(blocked) or 'none'})"
        )
    return addresses


async def assert_public_url(url: str) -> None:
    """Check the host of ``url`` (any spelling) resolves only to public addresses."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip()
    if not host:
        raise BlockedHostError(f"{url!r} names no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    await public_addresses(host, port)


class PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """Connect only to a checked public address of the requested host."""

    def __init__(self) -> None:
        self._inner = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        address = (await public_addresses(host, port))[0]
        return await self._inner.connect_tcp(
            address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> httpcore.AsyncNetworkStream:
        raise BlockedHostError("Unix sockets are not reachable from this tool")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def public_transport() -> httpx.AsyncHTTPTransport:
    """An httpx transport whose direct connections reach only public addresses."""
    transport = httpx.AsyncHTTPTransport()
    # httpx does not take a network backend; its httpcore pool does.
    transport._pool._network_backend = PublicOnlyBackend()
    return transport
