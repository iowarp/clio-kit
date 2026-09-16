# Remote and Branch Policy Reference

This reference covers remote classification, branch policies, and publication gates for `ship` and `branch-closeout`.

## Remote Classification

Never assume a remote named `origin` is the contributor's fork or the canonical repository. Different developer setups configure remotes differently:
- **Contributor fork model**: `origin` frequently points to the contributor's fork, while `upstream` points to the canonical organization repository.
- **Direct checkout model**: `origin` points directly to the canonical repository.
- **Custom remote names**: remotes may be named after usernames (e.g. `akougkas`, `upstream`, `github`).

### Deterministic Detection
1. Query the canonical repository via GitHub CLI:
   ```bash
   gh repo view --json nameWithOwner,defaultBranchRef -q '{repo: .nameWithOwner, defaultBranch: .defaultBranchRef.name}'
   ```
2. Query all configured Git remote URLs:
   ```bash
   git remote -v
   ```
3. Identify which remote URL matches the canonical repository. That remote is `<canonical>`.
4. Identify which remote URL points to the contributor's personal fork. That remote is `<fork>`.
5. If only one remote exists and its URL matches canonical repository policy:
   - Check if project instructions declare a canonical-main-only policy (e.g., `CONTRIBUTING.md`).
   - If canonical-main-only, the local clone cannot push topic branches to this remote.

## Default Branch Detection

Detect the default target base branch rather than hardcoding `main`:
1. Check `gh repo view --json defaultBranchRef -q .defaultBranchRef.name`.
2. Or inspect `git symbolic-ref refs/remotes/<canonical>/HEAD`.
3. Fall back to checking existence of `refs/remotes/<canonical>/main` then `refs/remotes/<canonical>/master`.

## Branch Policies

### 1. Canonical-Main-Only Policy (e.g., Clio Coder CONTRIBUTING.md)
- **Rules**:
  - The canonical repository hosts only the default branch (`main`).
  - Maintainer topic, integration, and release candidate branches (`v043`) stay strictly local.
  - Maintainers update canonical `main` only via an authorized, locally gated fast-forward (`git merge --ff-only`).
  - Contributors must push topic branches to their personal fork remote and open a PR against canonical `main`.
  - Never push a topic branch or candidate branch to canonical `origin`.
- **Refusal Guard**:
  - If a push to canonical is requested for any branch other than `main` (or fully qualified annotated release tags), the skill must refuse the push, explain the policy, and either require a fork remote or halt for local maintainer integration.

### 2. Canonical Topic Branch Policy (Feature Branch Workflows)
- In repositories where developers have direct write access to canonical feature branches:
  - Pushes to topic branches on the canonical remote are permitted.
  - Direct pushes to the protected default branch (`main`/`master`) remain guarded.

## Safe Push Semantics

1. Always use explicit refspecs to avoid pushing untracked or misconfigured refs:
   ```bash
   git push -u <remote> refs/heads/<topic>:refs/heads/<topic>
   ```
2. Never push with dirty or uncommitted changes in the worktree.
3. Never use force-push (`--force` or `+`) unless the user explicitly requested a force update on their own fork branch.
4. Open pull requests with explicit canonical target repo and fork head:
   ```bash
   gh pr create --repo <canonical-owner/repo> --base <base> --head <fork-owner>:<topic>
   ```
