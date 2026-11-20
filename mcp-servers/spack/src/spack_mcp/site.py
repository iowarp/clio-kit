"""Prepare and research a CLIO Spack on a remote site, reached only through clio-relay.

This server never opens its own connection to a site. Each remote tool works in
two calls. The first call returns the exact clio-relay ``exec`` call to make.
After the agent makes that relay call (with the normal approval), the second
call gets the same arguments plus ``relay_result`` (the ``exec`` tool's
structured result) and returns the parsed answer.

Login nodes are shared, so the commands stay light and short. They clone, write
config, and run ``compiler find``, ``external find``, ``find``, ``info`` and
``spec``. They never install or build, and nothing outlives the relay call.
Package installs run as jarvis install jobs (``jarvis ppl install ... +submit``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from . import site_scripts
from .backend import (
    _FIND_NO_MATCH_DIAGNOSTIC,
    SPACK_RESULT_SCHEMA,
    SpackBackendError,
    SpackFindResult,
    _CommandResult,
    _decode_find_records,
    _package_summary,
    _validated_spec,
)
from .discovery import SpackInfoResult, _parse_spack_info_output

DEFAULT_SITE_HOME: Final = "~/.clio-relay"
RELAY_SERVER: Final = "clio-relay"

# Spack release tags checked against github.com/spack/spack. A release missing
# here is still cloned at its tag; the report then says it is not pinned.
SPACK_RELEASE_PINS: Final = {
    "0.21.3": "6467ddefa3aa41629f9ca04dffab24b01e7d20b2",
    "0.22.2": "594a376c521cc746978571b1181a47bbcff30a21",
    "0.22.5": "8c37ed4534414d99501a91e64e40718e271ff8a2",
    "0.23.0": "c6d4037758140fe15913c29e80cd1547f388ae51",
    "0.23.1": "2bfcc69fa870d3c6919be87593f22647981b648a",
    "1.0.2": "734c5db2121b01c373eed6538e452f18887e9e44",
    "1.1.0": "0c2be44e4ece21eb091ad5de4c97716b7c6d4c87",
}

# Executable-only build tools. MPI is never externalized from PATH.
DEFAULT_BUILD_TOOLS: Final = (
    "autoconf",
    "automake",
    "bison",
    "cmake",
    "diffutils",
    "findutils",
    "flex",
    "gawk",
    "git",
    "gmake",
    "libtool",
    "m4",
    "ninja",
    "perl",
    "pkgconf",
    "tar",
)
MPI_PACKAGES: Final = (
    "cray-mpich",
    "hpcx-mpi",
    "intel-mpi",
    "intel-oneapi-mpi",
    "mpi",
    "mpich",
    "mpt",
    "msmpi",
    "mvapich",
    "mvapich2",
    "openmpi",
    "spectrum-mpi",
)
_SCRIPT_NAME: Final = "clio-spack"
_PACKAGE_NAME: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_MODULE_QUERY: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./+-]{0,199}$")
_RELEASE: Final = re.compile(r"^\d+\.\d+(?:\.\d+)?$")
_SITE_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_SPEC_LINE: Final = re.compile(
    r"^(?P<status>\[\+\]|\[\^\]|\[e\]| - )\s+(?P<hash>[a-z0-9]{7,32})\s{2}"
    r"(?P<indent>\s*)(?P<dep>\^?)(?P<spec>\S.*)$"
)
_SPEC_STATUS: Final = {"[+]": "installed", "[^]": "upstream", "[e]": "external", " - ": "missing"}
_LOGIN_NODE_NOTE: Final = (
    "Login-node work only: git clone, writing config, compiler find, external find, "
    "find, info and spec. Nothing installs or builds here, and nothing keeps running "
    "after the relay call. Package installs go through a jarvis install job "
    "(jarvis ppl install ... +submit)."
)
_GUARDED_NOTE: Final = (
    "clio-relay refused this command. On a Secure (guarded) site relay exec runs only "
    "its read-only list, which has no bash, git or spack, so the CLIO Spack can't be "
    "prepared or queried through exec there yet. Tell the person; don't work around "
    "the guard."
)

SiteStatus = Literal["relay_call_required", "done", "failed", "refused"]


class RelayExecArguments(BaseModel):
    """Arguments for clio-relay's ``exec`` tool, to pass unchanged."""

    model_config = ConfigDict(extra="forbid")

    site: str
    command: list[str]
    timeout_s: float
    max_output_bytes: int


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
    next_step: str
    login_node: str = _LOGIN_NODE_NOTE
    find: SpackFindResult | None = None
    info: SpackInfoResult | None = None
    spec: SpackSpecResult | None = None
    modules: SiteModulesResult | None = None
    error: str | None = None
    stderr_tail: str | None = None


def prepare_site(
    site: str,
    *,
    site_spack: str | None,
    site_home: str = DEFAULT_SITE_HOME,
    root: str | None = None,
    spack_version: str | None = None,
    site_install_tree: str | None = None,
    module_paths: list[str] | None = None,
    build_tools: list[str] | None = None,
    relay_result: dict[str, Any] | None = None,
) -> SpackPrepareSiteResult:
    """Plan the relay exec that prepares a CLIO Spack, or parse its result."""
    site = _site_id(site)
    home = _site_path(site_home, "site_home")
    spack_root = _site_path(root, "root") if root else f"{home}/spack"
    config_dir = f"{home}/spack-config"
    upstream = _site_path(site_spack, "site_spack") if site_spack else ""
    tree = _site_path(site_install_tree, "site_install_tree") if site_install_tree else ""
    if spack_version is not None and not _RELEASE.fullmatch(spack_version):
        raise _invalid("spack_version must be a Spack release such as 0.23.1")
    if not upstream and spack_version is None:
        raise _invalid("give site_spack, or spack_version when the site has no Spack")
    tools = list(DEFAULT_BUILD_TOOLS if build_tools is None else build_tools)
    for tool in tools:
        if not _PACKAGE_NAME.fullmatch(tool):
            raise _invalid(f"build tool {tool!r} is not a Spack package name")
        if tool in MPI_PACKAGES:
            raise _invalid(f"{tool} is an MPI; MPI is never externalized from PATH")
    paths = [_site_path(path, "module_paths") for path in module_paths or []]
    command = [
        "bash",
        "-c",
        site_scripts.PREPARE_SCRIPT,
        _SCRIPT_NAME,
        upstream,
        spack_root,
        config_dir,
        spack_version or "",
        " ".join(f"{name}={sha}" for name, sha in sorted(SPACK_RELEASE_PINS.items())),
        tree,
        ":".join(paths),
        " ".join(name for name in MPI_PACKAGES if name != "mpi"),
        site_scripts.POLICY_PY,
        *tools,
    ]
    call = RelayCall(
        arguments=RelayExecArguments(
            site=site, command=command, timeout_s=900, max_output_bytes=262_144
        )
    )
    if relay_result is None:
        release = f"v{spack_version}" if spack_version else "at the site Spack's release"
        source = (
            f"upstream: the install tree of the site Spack at {upstream}"
            if upstream
            else "no site Spack upstream"
        )
        return SpackPrepareSiteResult(
            site=site,
            status="relay_call_required",
            relay_call=call,
            approval=(
                f"Prepare a CLIO Spack on {site}: shallow-clone Spack {release} into "
                f"{spack_root} unless it is there; write its etc/spack config ({source}, "
                "concretizer reuse, packages:mpi:buildable:false); run spack compiler find "
                f"and spack external find --not-buildable {' '.join(tools) or '(none)'}. "
                "No install, no build, nothing left running. ~/.spack is not read or changed."
            ),
            next_step=(
                "Ask the person with `approval`, call clio-relay exec with "
                "relay_call.arguments unchanged, then call spack_prepare_site again with the "
                "same arguments and relay_result set to the exec structured result."
            ),
        )
    outcome = _relay_outcome(relay_result, command)
    if isinstance(outcome, str):
        return SpackPrepareSiteResult(
            site=site, status="refused", error=outcome, next_step="Tell the person; stop."
        )
    marks = _marks(outcome.stdout)
    errors = marks.get("error", [])
    if outcome.returncode != 0 or errors or "report" not in marks:
        return SpackPrepareSiteResult(
            site=site,
            status="failed",
            error="; ".join(errors) or _exit_text(outcome),
            stderr_tail=_tail(outcome.stderr),
            next_step=(
                "Show the person the error. Nothing was installed. Re-running is safe "
                "once the cause is fixed."
            ),
        )
    return SpackPrepareSiteResult(
        site=site,
        status="done",
        report=_prepare_report(marks),
        next_step=(
            "Research with spack_site_query (find, info, spec, modules). Install through a "
            "jarvis install job with report.jarvis.environment exported; when jarvis "
            "raises an install issue such as ambiguous_provider, ask the person."
        ),
    )


def site_query(
    site: str,
    query: Literal["find", "info", "spec", "modules"],
    *,
    spec: str | None = None,
    site_home: str = DEFAULT_SITE_HOME,
    root: str | None = None,
    relay_result: dict[str, Any] | None = None,
) -> SpackSiteQueryResult:
    """Plan a read-only research command for the site, or parse its result."""
    site = _site_id(site)
    home = _site_path(site_home, "site_home")
    spack_root = _site_path(root, "root") if root else f"{home}/spack"
    config_dir = f"{home}/spack-config"
    term = None if spec is None else _validated_spec(spec)
    if query in {"info", "spec"} and term is None:
        raise _invalid(f"spack {query} needs spec")
    if query == "modules":
        if term is not None and not _MODULE_QUERY.fullmatch(term):
            raise _invalid("a module query is one module name, such as mpich or openmpi/5")
        command = ["module", "-t", "avail", *([term] if term else [])]
        timeout, limit = 60.0, 524_288
    else:
        if query == "info" and not _PACKAGE_NAME.fullmatch(term or ""):
            raise _invalid("spack info takes one package name")
        verb = {
            "find": ["find", "--json", *([term] if term else [])],
            "info": ["info", term or ""],
            "spec": ["spec", "-I", "-l", term or ""],
        }[query]
        command = [
            "bash",
            "-c",
            site_scripts.QUERY_SCRIPT,
            _SCRIPT_NAME,
            spack_root,
            config_dir,
            *verb,
        ]
        timeout = 600.0 if query == "spec" else 120.0
        limit = 2_097_152 if query == "find" else 262_144
    call = RelayCall(
        arguments=RelayExecArguments(
            site=site, command=command, timeout_s=timeout, max_output_bytes=limit
        )
    )
    if relay_result is None:
        return SpackSiteQueryResult(
            site=site,
            query=query,
            status="relay_call_required",
            relay_call=call,
            next_step=(
                "Call clio-relay exec with relay_call.arguments unchanged, then call "
                "spack_site_query again with the same arguments and relay_result set to the "
                "exec structured result."
            ),
        )
    outcome = _relay_outcome(relay_result, command)
    if isinstance(outcome, str):
        return SpackSiteQueryResult(
            site=site, query=query, status="refused", error=outcome, next_step="Tell the person."
        )
    try:
        parsed = _parse_query(query, term, outcome)
    except SpackBackendError as exc:
        return SpackSiteQueryResult(
            site=site,
            query=query,
            status="failed",
            error=f"{exc.code}: {exc.message}" + (f" ({exc.detail})" if exc.detail else ""),
            stderr_tail=_tail(outcome.stderr),
            next_step="Show the person the error, or fix the query and ask again.",
        )
    result = SpackSiteQueryResult(
        site=site, query=query, status="done", next_step="Use the answer."
    )
    return result.model_copy(update=parsed)


def _parse_query(query: str, term: str | None, outcome: _CommandResult) -> dict[str, BaseModel]:
    errors = _marks(outcome.stdout).get("error", [])
    if errors:
        raise SpackBackendError("site_error", "; ".join(errors), operation="site_query")
    if outcome.stdout_truncated:
        raise SpackBackendError(
            "response_too_large",
            "the site's answer was cut off; narrow the query",
            operation="site_query",
        )
    if query == "modules":
        if outcome.returncode != 0:
            raise _failed(outcome)
        return {"modules": _parse_modules(term, outcome.stderr + "\n" + outcome.stdout)}
    if query == "find":
        if outcome.returncode == 1 and _FIND_NO_MATCH_DIAGNOSTIC.fullmatch(outcome.stderr.strip()):
            return {"find": SpackFindResult(query=term, packages=[], count=0)}
        if outcome.returncode != 0:
            raise _failed(outcome)
        packages = sorted(
            (_package_summary(record) for record in _decode_find_records(outcome)),
            key=lambda item: (item.name, item.version or "", item.dag_hash or ""),
        )
        return {"find": SpackFindResult(query=term, packages=packages, count=len(packages))}
    if outcome.returncode != 0:
        raise _failed(outcome)
    if query == "info":
        info = _parse_spack_info_output(outcome.stdout)
        if info is None:
            raise SpackBackendError(
                "invalid_output", "spack info output could not be read", operation="info"
            )
        return {
            "info": SpackInfoResult(
                package=term or "",
                description=info.description,
                versions=info.versions,
                variants=info.variants,
                source="spack_info",
                degraded=not info.complete,
                degraded_reason=None if info.complete else "spack info missed a section",
            )
        }
    return {"spec": _parse_spec_tree(term or "", outcome)}


def _parse_spec_tree(term: str, outcome: _CommandResult) -> SpackSpecResult:
    nodes: list[SpackSpecNode] = []
    for line in outcome.stdout.splitlines():
        match = _SPEC_LINE.match(line)
        if match is None:
            continue
        text = match.group("spec").strip()
        name = re.match(r"[A-Za-z0-9_.-]+", text)
        nodes.append(
            SpackSpecNode(
                name=name.group(0) if name else text,
                hash=match.group("hash"),
                status=cast(Any, _SPEC_STATUS[match.group("status")]),
                depth=len(match.group("indent")) // 4,
                spec=text,
            )
        )
    if not nodes:
        raise SpackBackendError("invalid_output", "spack spec printed no tree", operation="spec")
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.status] = counts.get(node.status, 0) + 1
    warnings = [
        line.strip()
        for line in outcome.stderr.splitlines()
        if line.strip().startswith("==> Warning")
    ][:20]
    return SpackSpecResult(
        spec=term,
        nodes=nodes,
        counts=counts,
        to_build=[node.name for node in nodes if node.status == "missing"],
        warnings=warnings,
    )


def _parse_modules(term: str | None, text: str) -> SiteModulesResult:
    modules: list[SiteModule] = []
    directory: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("@@", "Lmod", 'Use "module')):
            continue
        if line.endswith(":") and line.startswith("/"):
            directory = line[:-1]
            continue
        name = re.sub(r"\s*\((?:D|L|default)\)$", "", line)
        if name.endswith("/") or " " in name:
            continue
        modules.append(SiteModule(name=name, directory=directory))
    return SiteModulesResult(query=term, modules=modules)


def _relay_outcome(relay_result: dict[str, Any], command: list[str]) -> _CommandResult | str:
    """Read clio-relay's exec structured result, refusing one for another command."""
    payload: Any = relay_result
    if isinstance(payload.get("structuredContent"), dict):
        payload = payload["structuredContent"]
    refusal = payload.get("refusal")
    if isinstance(refusal, dict):
        reason = str(refusal.get("reason", "refused"))
        message = str(refusal.get("message", ""))
        prefix = _GUARDED_NOTE + " " if reason == "policy_refused" else ""
        return f"{prefix}relay: {reason}: {message}".strip()
    argv = payload.get("argv")
    if argv is not None and list(argv) != command:
        raise _invalid("relay_result is for a different command; pass this call's exec result")
    stdout, stderr = payload.get("stdout"), payload.get("stderr")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise _invalid("relay_result must be clio-relay exec's structured result")
    code = payload.get("exit_code")
    timed_out = bool(payload.get("timed_out"))
    return _CommandResult(
        argv=tuple(command),
        returncode=124 if timed_out else (code if isinstance(code, int) else -1),
        stdout=stdout,
        stderr=stderr + ("\n[timed out]" if timed_out else ""),
        duration_seconds=float(payload.get("duration_s") or 0.0),
        stdout_truncated=bool(payload.get("stdout_truncated")),
        stderr_truncated=bool(payload.get("stderr_truncated")),
    )


def _marks(stdout: str) -> dict[str, list[str]]:
    marks: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        if line.startswith("@@") and "\t" in line:
            key, value = line[2:].split("\t", 1)
            marks.setdefault(key, []).append(value)
    return marks


def _prepare_report(marks: dict[str, list[str]]) -> SitePrepareReport:
    def first(key: str) -> str | None:
        values = marks.get(key)
        return values[0] if values else None

    data = cast(dict[str, Any], json.loads(marks["report"][-1]))
    root = first("root") or ""
    config_dir = first("config_dir") or ""
    candidates = [value.split(" ", 1)[0] for value in marks.get("mpi_module", [])]
    return SitePrepareReport(
        root=root,
        config_dir=config_dir,
        spack_version=first("version") or "",
        commit=first("commit") or None,
        pinned=first("pinned") is not None,
        clone=cast(Any, first("clone") or "present"),
        site_spack=first("site_spack"),
        site_version=first("site_version"),
        upstream=first("upstream"),
        config_changes=[
            value.split(" ", 1)[1]
            for value in marks.get("config", [])
            if value.startswith("written ")
        ],
        compilers=[str(item) for item in data.get("compilers", [])],
        externals=cast(dict[str, list[dict[str, Any]]], data.get("externals", {})),
        mpi_buildable=bool(data.get("mpi_buildable", True)),
        reuse=data.get("reuse"),
        installed_mpis=cast(list[dict[str, Any]], data.get("installed_mpis", [])),
        mpi_module_candidates=candidates,
        warnings=marks.get("warning", []),
        jarvis=JarvisSpackSettings(
            environment={"JARVIS_SPACK_ROOT": root, "JARVIS_SPACK_CONFIG": config_dir},
            note=(
                "jarvis-cd reads JARVIS_SPACK_ROOT (falling back to SPACK_ROOT) and "
                "JARVIS_SPACK_CONFIG. The CLIO Spack keeps its own config in "
                f"{root}/etc/spack and its caches under {root}/var/clio, so the user's "
                "~/.spack and ~/spack never leak in."
            ),
        ),
    )


def _site_id(site: str) -> str:
    value = site.strip()
    if not _SITE_ID.fullmatch(value):
        raise _invalid("site must be a clio-relay site id")
    return value


def _site_path(value: str | None, field: str) -> str:
    path = (value or "").strip().rstrip("/")
    if not path or not (path.startswith("/") or path == "~" or path.startswith("~/")):
        raise _invalid(f"{field} must be an absolute path or start with ~/")
    if any(ord(char) < 32 or ord(char) == 127 for char in path) or "/../" in f"{path}/":
        raise _invalid(f"{field} must be a plain path")
    if path in {"/", "~"}:
        raise _invalid(f"{field} can't be / or ~")
    return path


def _invalid(message: str) -> SpackBackendError:
    return SpackBackendError("invalid_request", message, operation="validation")


def _failed(outcome: _CommandResult) -> SpackBackendError:
    return SpackBackendError(
        "command_failed",
        _exit_text(outcome),
        operation="site_query",
        returncode=outcome.returncode,
        detail=_tail(outcome.stderr),
    )


def _exit_text(outcome: _CommandResult) -> str:
    if outcome.returncode == 124:
        return "the command timed out"
    return f"the command exited with {outcome.returncode}"


def _tail(text: str, limit: int = 4000) -> str | None:
    value = text.strip()
    if not value:
        return None
    return value if len(value) <= limit else "[cut]\n" + value[-limit:]
