"""Result models of the remote-site tools (spack_prepare_site, spack_site_query)."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from .backend import SPACK_RESULT_SCHEMA, SpackFindResult
from .discovery import SpackInfoResult

RELAY_SERVER: Final = "clio-relay"
# Spack caps config:build_jobs at the CPUs the process may use (sched_getaffinity),
# so this high cap means every CPU inside the install job.
ALL_CPUS_BUILD_JOBS: Final = 1024
_LOGIN_NODE_NOTE: Final = (
    "Login-node work only: git clone, writing config, compiler find, external find, "
    "find, info and spec. Nothing installs or builds here, and nothing keeps running "
    "after the relay call. Package installs go through a jarvis install job "
    "(jarvis ppl install ... +submit)."
)

SiteStatus = Literal[
    "relay_call_required", "needs_approval", "done", "failed", "refused", "declined"
]


class RelayExecPlan(BaseModel):
    """The plan clio-relay's guard asks the person to approve before a non-read command."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    summary: str


class RelayExecArguments(BaseModel):
    """Arguments for clio-relay's ``exec`` tool, to pass unchanged."""

    model_config = ConfigDict(extra="forbid")

    site: str
    command: list[str]
    timeout_s: float
    max_output_bytes: int
    plan: RelayExecPlan | None = None


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


class JarvisSpackSettings(BaseModel):
    """What jarvis-cd must be told so its install jobs use the CLIO Spack."""

    model_config = ConfigDict(extra="forbid")

    environment: dict[str, str] = Field(
        description=(
            "Export these into jarvis install and run jobs on the site. "
            "JARVIS_SPACK_ROOT selects the CLIO Spack; JARVIS_SPACK_CONFIG makes jarvis "
            "set SPACK_USER_CONFIG_PATH to it and SPACK_DISABLE_LOCAL_CONFIG=1, so ~/.spack "
            "is never read."
        )
    )
    note: str


class SiteToolchain(BaseModel):
    """One compiler spack compiler find registered."""

    model_config = ConfigDict(extra="forbid")

    spec: str
    cc: str | None = None
    cxx: str | None = None
    fc: str | None = None
    full: bool = Field(description="cc, cxx and fc are all set.")
    built_upstream_mpi: bool = Field(description="An installed MPI was built with it.")


class SiteCompilerChoice(BaseModel):
    """The compiler preference written as packages:all:compiler (Spack 0.23 semantics)."""

    model_config = ConfigDict(extra="forbid")

    chosen: str | None = Field(
        default=None, description="First in packages:all:compiler; the solver prefers it."
    )
    source: (
        Literal[
            "person",
            "person_earlier",
            "upstream_mpi",
            "only_full_toolchain",
            "default",
            "no_full_toolchain",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "person: the compiler input; person_earlier: an earlier compiler input; "
            "upstream_mpi: the only full toolchain an installed MPI was built with; "
            "only_full_toolchain: the only one with cc, cxx and fc; default: several "
            "qualify (ask the person); no_full_toolchain: nothing has Fortran."
        ),
    )
    full_toolchain: bool = False
    order: list[str] = Field(default_factory=list, description="packages:all:compiler.")
    toolchains: list[SiteToolchain] = Field(default_factory=list)
    choice_needed: bool = Field(
        default=False,
        description=(
            "More than one full toolchain qualifies. Ask the person once to pick from "
            "candidates, then prepare again with compiler set."
        ),
    )
    candidates: list[str] = Field(default_factory=list)


class SitePrepareReport(BaseModel):
    """What spack_prepare_site found and wrote on the site."""

    model_config = ConfigDict(extra="forbid")

    root: str
    config_dir: str
    spack_version: str
    commit: str | None = None
    pinned: bool
    clone: Literal["cloned", "present"]
    site_spack: str | None = None
    site_version: str | None = None
    upstream: str | None = None
    config_changes: list[str] = Field(default_factory=list)
    compilers: list[str] = Field(default_factory=list)
    externals: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    mpi_buildable: bool
    reuse: bool | None = None
    installed_mpis: list[dict[str, Any]] = Field(
        default_factory=list,
        description="MPIs in the upstream and CLIO install trees; jarvis picks among them.",
    )
    mpi_module_candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Module-only MPIs, recorded as non-default externals so jarvis can offer "
            "them as a choice. Never chosen without the person."
        ),
    )
    warnings: list[str] = Field(default_factory=list)
    compiler: SiteCompilerChoice
    build_jobs: int = Field(
        description=(
            "config:build_jobs in the CLIO Spack. Spack builds with min(build_jobs, CPUs "
            f"the job may use); {ALL_CPUS_BUILD_JOBS} means every CPU the job was given."
        )
    )
    jarvis: JarvisSpackSettings


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


class SpackPrepareSiteResult(BaseModel):
    """Result of spack_prepare_site, either the relay call to make or the report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["spack.mcp.result.v1"] = SPACK_RESULT_SCHEMA
    operation: Literal["prepare_site"] = "prepare_site"
    site: str
    status: SiteStatus
    relay_call: RelayCall | None = None
    approval: str | None = Field(
        default=None,
        description="What the person approves before the relay exec call.",
    )
    ask: RelayAsk | None = Field(
        default=None, description="The relay ask the person answers (needs_approval)."
    )
    next_step: str
    login_node: str = _LOGIN_NODE_NOTE
    report: SitePrepareReport | None = None
    error: str | None = None
    stderr_tail: str | None = None


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
