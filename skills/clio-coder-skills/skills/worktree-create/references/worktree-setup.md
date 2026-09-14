# Worktree Setup Reference

A git worktree shares the repository's object store and tracked files, but operates as an independent working tree. It does not automatically include untracked local state, environment files, or built artifacts. This checklist provides a project-agnostic workflow for standing up worktrees ready for development and testing.

## 1. Branch Base and Safe Path Derivation

- **Base ref**: Create the branch from the intended base ref (`--base <ref>`, defaulting to the canonical default branch or `HEAD`). Validate the ref with `git rev-parse --verify <ref>`.
- **Worktree root**: Worktrees are placed under `--root <path>` (default: `worktrees/`). Ensure the root path is included in `.gitignore` so it does not appear as untracked in the primary checkout.
- **Safe filesystem paths**: Branch names may contain slashes (e.g., `feat/auth/login`) or characters that require sanitization on disk. Derive filesystem paths under the worktree root safely:
  - Keep paths strictly inside the declared root; reject paths containing directory traversal (`..`).
  - Map branch names to safe relative paths or subdirectories (e.g. `<root>/feat-auth-login` or `<root>/feat/auth/login`).
  - Validate the branch name format using `git check-ref-format --branch <branch>`.

## 2. Gitignored Config & Secret Handling

A fresh worktree lacks the untracked runtime configuration files necessary for the application to boot:
- Candidate files: `.env`, `.env.local`, `.env.<stage>`, local credentials, certificates (`*.pem`), or local tool configs.
- **Rule**: Never copy tracked files.
- **Verification**: Run `git check-ignore <file>` on every candidate file in the main tree before copying it to the worktree.
- If the repository provides `.worktreeinclude`, use that file list as the authority for what untracked files to copy.

## 3. Dependency Installation (Project-Agnostic Detection)

Inspect repository manifests and lockfiles to determine the appropriate package manager:
- **Node.js**:
  - `package-lock.json` -> `npm ci`
  - `pnpm-lock.yaml` -> `pnpm install --frozen-lockfile`
  - `yarn.lock` -> `yarn install --frozen-lockfile`
  - `bun.lockb` / `bun.lock` -> `bun install --frozen-lockfile`
- **Python**:
  - `uv.lock` -> `uv sync`
  - `poetry.lock` -> `poetry install`
  - `Pipfile.lock` -> `pipenv install`
  - `requirements.txt` -> `pip install -r requirements.txt`
- **Rust**:
  - `Cargo.lock` / `Cargo.toml` -> `cargo build`
- **Go**:
  - `go.mod` / `go.sum` -> `go mod download`
- **Ruby**:
  - `Gemfile.lock` -> `bundle install`
- **PHP**:
  - `composer.lock` -> `composer install`

For monorepos, run install in each workspace package directory as defined in the top-level manifest.

## 4. Setup Mode (`--setup auto|none`)

- **`auto`** (default): Runs detected config copy, package installation, codegen/build steps, and health verification.
- **`none`**: Checks out the worktree branch only without copying files or running commands. Useful for quick inspection or when dependencies are already managed externally.

## 5. Service Port Isolation

If the worktree health check or execution starts long-running network daemons:
- Assign unique ports per worktree (`base_port + worktree_index`) via environment variables (e.g. `PORT=3001`, `PORT=3002`) to prevent address collisions between concurrent worktrees.

## 6. Verification and Health Checks

Prefer existing verification defined in CI workflows (`.github/workflows/*`), Makefile targets, or package scripts:
1. Hit an application health check endpoint if defined (e.g. `/healthz`).
2. Run the fast test/build smoke suite (e.g. `npm test`, `cargo test`, `pytest`).
3. Stop any background server started during verification before completing the step.
4. Report health status as PASS or FAIL per worktree. Failures must name the exact command and error.
