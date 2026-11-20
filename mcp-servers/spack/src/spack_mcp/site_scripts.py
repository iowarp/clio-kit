"""Bash that spack_prepare_site runs through clio-relay ``exec`` on a login node.

It does light, short work only: a shallow git clone, writing config,
``compiler find`` and ``external find``. It never installs, builds or leaves a
process behind. Relay runs the argv under ``setsid`` from ``$HOME`` and kills
the process group when the call ends. Research queries need no script: they
run Spack directly as ``env SPACK_DISABLE_LOCAL_CONFIG=1 <spack> find|info|spec``,
the shape clio-relay's guard accepts.
"""

from __future__ import annotations

from typing import Final

# Expand ``~`` and isolate Spack from ~/.spack.
_ISOLATION: Final = r"""
set -u
umask 022
say() { printf '@@%s\t%s\n' "$1" "$2"; }
fail() { say error "$1"; exit "${2:-1}"; }
home_path() {
  case "$1" in
    "~") printf '%s' "$HOME" ;;
    "~/"*) printf '%s/%s' "$HOME" "${1#\~/}" ;;
    *) printf '%s' "$1" ;;
  esac
}
isolate() {
  export SPACK_DISABLE_LOCAL_CONFIG=1
  export SPACK_USER_CONFIG_PATH="$1"
  export SPACK_USER_CACHE_PATH="$1/cache"
  unset SPACK_ENV SPACK_ENV_VIEW SPACK_ROOT
}
spack_version() {
  sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$1/lib/spack/spack/__init__.py" 2>/dev/null | head -n 1
}
"""

PREPARE_SCRIPT: Final = (
    _ISOLATION
    + r"""
SITE=$(home_path "$1"); ROOT=$(home_path "$2"); CONFIG=$(home_path "$3")
VERSION=$4; PINS=$5; TREE=$(home_path "$6"); MODPATHS=$7; MPI_NAMES=$8; POLICY=$9
shift 9
isolate "$CONFIG"
mkdir -p "$CONFIG" || fail "cannot create $CONFIG"
say config_dir "$CONFIG"

if [ -n "$SITE" ]; then
  [ -x "$SITE/bin/spack" ] || fail "no site Spack at $SITE"
  SITE_VERSION=$(spack_version "$SITE")
  if [ -z "$SITE_VERSION" ]; then
    SITE_VERSION=$(timeout 60 "$SITE/bin/spack" --version 2>/dev/null | awk '{print $1; exit}')
  fi
  [ -n "$SITE_VERSION" ] || fail "cannot read the site Spack release at $SITE"
  say site_spack "$SITE"
  say site_version "$SITE_VERSION"
  [ -n "$VERSION" ] || VERSION=$SITE_VERSION
  if [ -z "$TREE" ]; then
    TREE=$(timeout 60 "$SITE/bin/spack" python -c \
      'import spack.store; print(spack.store.STORE.root)' 2>/dev/null | tail -n 1)
    [ -n "$TREE" ] || TREE="$SITE/opt/spack"
  fi
  if [ -f "$TREE/.spack-db/index.json" ]; then say upstream "$TREE"
  else fail "site install tree $TREE has no .spack-db/index.json"; fi
fi
[ -n "$VERSION" ] || fail "no Spack release: pass spack_version when the site has no Spack"
case "$VERSION" in *[!0-9.]*|"") fail "bad Spack release $VERSION" ;; esac

PIN=""
for entry in $PINS; do
  [ "${entry%%=*}" = "$VERSION" ] && PIN=${entry#*=}
done

if [ -e "$ROOT/bin/spack" ]; then
  HAVE=$(spack_version "$ROOT")
  [ "$HAVE" = "$VERSION" ] || fail "the CLIO Spack at $ROOT is $HAVE, not $VERSION; nothing changed"
  say clone "present"
elif [ -e "$ROOT" ] && [ -n "$(ls -A "$ROOT" 2>/dev/null)" ]; then
  fail "$ROOT exists and is not a Spack checkout; nothing was changed"
else
  command -v git >/dev/null 2>&1 || fail "git is not on PATH"
  mkdir -p "$(dirname "$ROOT")" || fail "cannot create $(dirname "$ROOT")"
  timeout 600 git -c advice.detachedHead=false clone --quiet --depth 1 \
    --branch "v$VERSION" https://github.com/spack/spack.git "$ROOT" \
    || fail "git clone of Spack v$VERSION into $ROOT failed"
  say clone "cloned"
fi
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
if [ -n "$PIN" ] && [ "$COMMIT" != "$PIN" ]; then
  fail "the CLIO Spack at $ROOT is at $COMMIT, not the v$VERSION pin $PIN"
fi
say root "$ROOT"
say version "$VERSION"
say commit "$COMMIT"
[ -n "$PIN" ] && say pinned "$PIN"
SPACK="$ROOT/bin/spack"

ETC="$ROOT/etc/spack"
mkdir -p "$ETC" || fail "cannot create $ETC"
put() {
  tmp=$(mktemp "$ETC/.$1.XXXXXX") || fail "cannot write in $ETC"
  cat > "$tmp"
  chmod 644 "$tmp"
  if cmp -s "$tmp" "$ETC/$1"; then rm -f "$tmp"; say config "unchanged $1"
  else mv -f "$tmp" "$ETC/$1"; say config "written $1"; fi
}
if [ -n "$SITE" ]; then
  LMOD="$SITE/share/spack/lmod"; TCL="$SITE/share/spack/modules"
  {
    printf 'upstreams:\n  site:\n    install_tree: %s\n' "$TREE"
    if [ -d "$LMOD" ] || [ -d "$TCL" ]; then
      printf '    modules:\n'
      [ -d "$LMOD" ] && printf '      lmod: %s\n' "$LMOD"
      [ -d "$TCL" ] && printf '      tcl: %s\n' "$TCL"
    fi
  } | put upstreams.yaml
fi
put config.yaml <<'EOF'
config:
  misc_cache: $spack/var/clio/misc-cache
  test_stage: $spack/var/clio/test-stage
  build_stage:
  - $tempdir/$user/clio-spack-stage
EOF
put concretizer.yaml <<'EOF'
concretizer:
  reuse: true
EOF
put bootstrap.yaml <<'EOF'
bootstrap:
  root: $spack/var/clio/bootstrap
  trusted:
    spack-install: false
EOF

timeout 300 "$SPACK" compiler find --scope site >&2 || fail "spack compiler find failed"
if [ "$#" -gt 0 ]; then
  timeout 300 "$SPACK" external find --not-buildable --scope site "$@" >&2 \
    || fail "spack external find failed"
fi

if [ -z "$MODPATHS" ]; then
  MODPATHS=$(timeout 30 bash -lc 'printf "\n@@MP=%s\n" "${MODULEPATH-}"' </dev/null 2>/dev/null \
    | sed -n 's/^@@MP=//p')
fi
# A module MPI's install prefix: the bin/ its module puts first on PATH. Only a
# light ``module load`` in a throwaway login shell; nothing is built or kept.
mpi_prefix() {
  timeout 30 bash -lc '
    before=":$PATH:"
    module use "$1" >/dev/null 2>&1
    module load "$2" >/dev/null 2>&1 || exit 1
    exe=$(command -v mpiexec || command -v mpirun) || exit 1
    case "$exe" in /*/bin/mpiexec|/*/bin/mpirun) ;; *) exit 1 ;; esac
    case "$before" in *":${exe%/*}:"*) exit 1 ;; esac
    printf "\n@@PREFIX=%s\n" "${exe%/bin/*}"
  ' _ "$1" "$2" </dev/null 2>/dev/null | sed -n 's/^@@PREFIX=//p' | tail -n 1
}
TAB=$'\t'
CANDIDATES=""
IFS=: read -r -a DIRS <<< "$MODPATHS"
for dir in "${DIRS[@]}"; do
  [ -n "$dir" ] && [ -d "$dir" ] || continue
  dir=${dir%/}
  # Spack's own module trees are already reachable through the upstream.
  if [ -n "$SITE" ]; then
    case "$dir" in "$SITE"/share/spack/*) continue ;; esac
  fi
  case "$dir" in "$ROOT"/share/spack/*) continue ;; esac
  for name in $MPI_NAMES; do
    [ -d "$dir/$name" ] || continue
    for file in "$dir/$name"/*; do
      base=${file##*/}; version=${base%.lua}
      case "$version" in .*|default|*[!A-Za-z0-9._-]*) continue ;; esac
      case "$version" in [0-9]*) ;; *) continue ;; esac
      module="$name/$version"
      prefix=$(mpi_prefix "$dir" "$module")
      if [ -z "$prefix" ] || [ ! -d "$prefix" ]; then
        say warning "MPI module $module ($dir): module load gave no mpiexec prefix; not recorded"
        continue
      fi
      CANDIDATES="$CANDIDATES$name$TAB$version$TAB$module$TAB$prefix
"
      say mpi_module "$module $prefix $dir"
    done
  done
done
export CLIO_MPI_CANDIDATES="$CANDIDATES"

# spack python -c compiles one statement only, so run the policy from a file.
POLICY_FILE=$(mktemp "$CONFIG/.clio-policy.XXXXXX.py") || fail "cannot write in $CONFIG"
printf '%s\n' "$POLICY" > "$POLICY_FILE"
timeout 120 "$SPACK" python "$POLICY_FILE"
STATUS=$?
rm -f "$POLICY_FILE"
[ "$STATUS" -eq 0 ] || fail "writing the Spack MPI policy failed"
"""
)

# Run by the CLIO Spack's own ``spack python``: set ``packages:mpi:buildable:false``,
# record module MPIs as externals with both ``prefix:`` and ``modules:`` (jarvis
# needs the prefix and loads the module) so they can be offered as a choice, and
# report what the merged configuration and the upstream database contain.
POLICY_PY: Final = r"""
import copy, json, os
import spack.config, spack.store
site = "site"
packages = copy.deepcopy(spack.config.CONFIG.get_config("packages", scope=site) or {})
changed = []
mpi = packages.setdefault("mpi", {})
if mpi.get("buildable") is not False:
    mpi["buildable"] = False
    changed.append("packages:mpi:buildable:false")
added = []
for line in os.environ.get("CLIO_MPI_CANDIDATES", "").splitlines():
    parts = line.split("\t")
    if len(parts) != 4 or not parts[3]:
        continue
    name, version, module, prefix = parts
    spec = name + "@" + version
    externals = packages.setdefault(name, {}).setdefault("externals", [])
    same = [i for i in externals if str(i.get("spec", "")).split()[:1] == [spec]]
    if not same:
        externals.append({"spec": spec, "prefix": prefix, "modules": [module]})
        added.append(spec)
        changed.append("packages:" + name + ":externals " + spec)
        continue
    item = same[0]
    modules = list(item.get("modules", []) or [])
    if not item.get("prefix"):
        item["prefix"] = prefix
        changed.append("packages:" + name + ":externals " + spec + " prefix")
    elif item["prefix"] != prefix:
        kept = item["prefix"]
        print("@@warning\t%s keeps prefix %s; module load gives %s" % (spec, kept, prefix))
    if module not in modules:
        item["modules"] = modules + [module]
        changed.append("packages:" + name + ":externals " + spec + " modules")
if changed:
    spack.config.CONFIG.update_config("packages", packages, scope=site)
    for item in changed:
        print("@@config\twritten " + item)
compilers = []
for entry in spack.config.get("compilers", []) or []:
    compiler = entry.get("compiler", {}) if isinstance(entry, dict) else {}
    if compiler.get("spec"):
        compilers.append(str(compiler["spec"]))
externals = {}
for name, settings in sorted((spack.config.get("packages", {}) or {}).items()):
    if not isinstance(settings, dict):
        continue
    for item in settings.get("externals", []) or []:
        externals.setdefault(name, []).append({
            "spec": str(item.get("spec", "")),
            "prefix": item.get("prefix"),
            "modules": list(item.get("modules", []) or []),
            "buildable": settings.get("buildable", True) is not False,
        })
mpis = []
try:
    for spec in spack.store.STORE.db.query("mpi"):
        mpis.append({
            "name": spec.name,
            "version": str(spec.version),
            "hash": spec.dag_hash(7),
            "prefix": str(spec.prefix),
            "upstream": bool(spec.installed_upstream),
        })
except Exception as exc:
    print("@@warning\tcould not list installed MPIs: %s" % exc)
report = {
    "compilers": compilers,
    "externals": externals,
    "mpi_buildable": spack.config.get("packages:mpi:buildable", True) is not False,
    "reuse": spack.config.get("concretizer:reuse", None),
    "installed_mpis": mpis,
    "mpi_externals_added": added,
    "upstreams": sorted((spack.config.get("upstreams", {}) or {}).keys()),
}
print("@@report\t" + json.dumps(report, sort_keys=True))
"""
