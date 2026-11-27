"""Result models of the remote-site tools (spack_site_query)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from .backend import SPACK_RESULT_SCHEMA, SpackFindResult
from .discovery import SpackInfoResult

RELAY_SERVER: Final = "clio-relay"
_LOGIN_NODE_NOTE: Final = (
    "Login-node work only: find, info, spec and module avail. Nothing installs or "
    "builds here, and nothing keeps running after the relay call. The Spack is "
    "prepared by jarvis spack prepare (through clio-relay); package installs go "
    "through a jarvis install job (jarvis ppl install ... +submit)."
)

SiteStatus = Literal[
    "relay_call_required", "needs_approval", "done", "failed", "refused", "declined"
]


class RelayExecArguments(BaseModel):
    """Arguments for clio-relay's ``exec`` tool, to pass unchanged."""

    model_config = ConfigDict(extra="forbid")

    site: str
    command: list[str]
    timeout_s: float
    max_output_bytes: int


class RelayAsk(BaseModel):
    """A pending clio-relay ask the person answers before exec runs."""

    model_config = ConfigDict(extra="forbid")

    ask_id: str | None = None
    message: str | None = None
    page: str | None = None


class RelayCall(BaseModel):
    """One clio-relay tool call the agent makes next."""

    model_config = ConfigDict(extra="forbid")

    server: Literal["clio-relay"] = RELAY_SERVER
    tool: Literal["exec"] = "exec"
    arguments: RelayExecArguments


class SpackSpecNode(BaseModel):
    """One node of ``spack spec -I -l``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    hash: str
    status: Literal["installed", "upstream", "external", "missing"]
    depth: int
    spec: str


class SpackSpecResult(BaseModel):
    """Concretized tree on the site, with what would still be built."""

    model_config = ConfigDict(extra="forbid")

    spec: str
    nodes: list[SpackSpecNode] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    to_build: list[str] = Field(
        default_factory=list,
        description="Missing nodes; a jarvis install job builds them on a compute node.",
    )
    warnings: list[str] = Field(default_factory=list)


class SiteModule(BaseModel):
    """One module from ``module -t avail``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    directory: str | None = None


class SiteModulesResult(BaseModel):
    """Modules the site's login shell offers."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = None
    modules: list[SiteModule] = Field(default_factory=list)


class SpackSiteQueryResult(BaseModel):
    """Result of spack_site_query, either the relay call to make or the parsed answer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["spack.mcp.result.v1"] = SPACK_RESULT_SCHEMA
    operation: Literal["site_query"] = "site_query"
    site: str
    query: Literal["find", "info", "spec", "modules"]
    status: SiteStatus
    relay_call: RelayCall | None = None
    ask: RelayAsk | None = None
    next_step: str
    login_node: str = _LOGIN_NODE_NOTE
    find: SpackFindResult | None = None
    info: SpackInfoResult | None = None
    spec: SpackSpecResult | None = None
    modules: SiteModulesResult | None = None
    error: str | None = None
    stderr_tail: str | None = None
