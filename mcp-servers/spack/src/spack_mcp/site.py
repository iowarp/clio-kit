"""Research a site's JARVIS Spack on a remote site, reached only through clio-relay.

This server never opens its own connection to a site. Each remote tool works in
two calls. The first call returns the exact clio-relay ``exec`` call to make.
After the agent makes that relay call (with the normal approval), the second
call gets the same arguments plus ``relay_result`` (the ``exec`` tool's
structured result) and returns the parsed answer.

Login nodes are shared, so the commands stay light and short: ``find``, ``info``,
``spec`` and ``module avail``. They never install or build, and nothing outlives
the relay call. The Spack itself is prepared by ``jarvis spack prepare`` (run
through clio-relay), and packages are installed by jarvis install jobs
(``jarvis ppl install ... +submit``).
"""

from __future__ import annotations

import re
from typing import Any, Final, Literal, cast

from pydantic import BaseModel

from .backend import (
    _FIND_NO_MATCH_DIAGNOSTIC,
    SpackBackendError,
    SpackFindResult,
    _CommandResult,
    _decode_find_records,
    _package_summary,
    _validated_spec,
)
from .discovery import SpackInfoResult, _parse_spack_info_output
from .site_models import (
    RelayAsk,
    RelayCall,
    RelayExecArguments,
    SiteModule,
    SiteModulesResult,
    SpackSiteQueryResult,
    SpackSpecNode,
    SpackSpecResult,
)

DEFAULT_SITE_HOME: Final = "~/.clio-relay"

_PACKAGE_NAME: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_MODULE_QUERY: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./+-]{0,199}$")
_SITE_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_SPEC_LINE: Final = re.compile(
    r"^(?P<status>\[\+\]|\[\^\]|\[e\]| - )\s+(?P<hash>[a-z0-9]{7,32})\s{2}"
    r"(?P<indent>\s*)(?P<dep>\^?)(?P<spec>\S.*)$"
)
_SPEC_STATUS: Final = {"[+]": "installed", "[^]": "upstream", "[e]": "external", " - ": "missing"}
_GUARDED_NOTE: Final = (
    "clio-relay's guard refused this command. A Secure (guarded) site runs Spack "
    "queries only as `env SPACK_DISABLE_LOCAL_CONFIG=1 <spack> find|info|spec` with the "
    "Spack its guard lists (default .clio-relay/spack/bin/spack), and other commands "
    "only with an approved plan. Tell the person; don't work around the guard."
)
_DECLINED_NOTE: Final = "The person did not approve this on the relay ask. Stop; don't retry."


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
        # clio-relay's exec runs from $HOME without a shell, so a ~/ path is given
        # home-relative; the guard matches the Spack program by exact string.
        program = (
            f"{spack_root[2:]}/bin/spack"
            if spack_root.startswith("~/")
            else (f"{spack_root}/bin/spack")
        )
        command = ["env", "SPACK_DISABLE_LOCAL_CONFIG=1", program, *verb]
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
    if isinstance(outcome, _Refused):
        return SpackSiteQueryResult(
            site=site,
            query=query,
            status=outcome.status,
            relay_call=call if outcome.status == "needs_approval" else None,
            ask=outcome.ask,
            error=outcome.error,
            next_step=outcome.next_step("spack_site_query"),
        )
    if query != "modules" and outcome.returncode == 127:
        return SpackSiteQueryResult(
            site=site,
            query=query,
            status="failed",
            error=f"no JARVIS Spack at {spack_root} on {site}",
            stderr_tail=_tail(outcome.stderr),
            next_step=(
                "Prepare the site's JARVIS Spack first: through clio-relay, run "
                "`jarvis spack prepare` (see jarvis docs/spack.md)."
            ),
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


class _Refused(BaseModel):
    """clio-relay answered with a refusal instead of running the command."""

    status: Literal["needs_approval", "declined", "refused"]
    error: str
    ask: RelayAsk | None = None

    def next_step(self, tool: str) -> str:
        if self.status == "needs_approval":
            return (
                "clio-relay asked the person to approve the command (see ask). Once they "
                "answer, call clio-relay exec again with the same relay_call.arguments, "
                f"then call {tool} again with relay_result."
            )
        if self.status == "declined":
            return _DECLINED_NOTE
        return "Tell the person; stop."


def _relay_outcome(relay_result: dict[str, Any], command: list[str]) -> _CommandResult | _Refused:
    """Read clio-relay's exec structured result, refusing one for another command."""
    payload: Any = relay_result
    if isinstance(payload.get("structuredContent"), dict):
        payload = payload["structuredContent"]
    refusal = payload.get("refusal")
    if isinstance(refusal, dict):
        reason = str(refusal.get("reason", "refused"))
        message = str(refusal.get("message", ""))
        data: dict[str, Any] = refusal.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        ask_id = data.get("ask_id")
        error = f"relay: {reason}: {message}".strip()
        if reason == "needs_approval":
            asks = [item for item in data.get("asks") or [] if isinstance(item, dict)]
            match = next((item for item in asks if item.get("ask_id") == ask_id), None)
            first: dict[str, Any] = match or (asks[0] if asks else {})
            return _Refused(
                status="needs_approval",
                error=error,
                ask=RelayAsk(
                    ask_id=None if ask_id is None else str(ask_id),
                    message=None if first.get("message") is None else str(first["message"]),
                    page=None if first.get("page") is None else str(first["page"]),
                ),
            )
        if reason == "policy_refused" and ask_id is not None:
            return _Refused(
                status="declined", error=error, ask=RelayAsk(ask_id=str(ask_id), message=message)
            )
        prefix = _GUARDED_NOTE + " " if reason == "policy_refused" else ""
        return _Refused(status="refused", error=prefix + error)
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
