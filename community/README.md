# Community contributions

Shared skills use standard `SKILL.md` folders and can be installed for Codex or
other compatible agents with `clio-kit skill install --target DIRECTORY`.
`clio-kit skill validate DIRECTORY` checks standalone skill contributions.
The plugin entries and marketplace federation described below currently use
Claude Code's native `.claude-plugin` format; they are not universal manifests.


Plugins, skills and MCP servers that live in **someone else's repository** and
appear in the CLIO Kit marketplace. One file here per contribution.

Your code stays yours. You release on your own schedule, and your updates reach
users through catalogue refresh and plugin updates without a CLIO package release.
Publishers must bump plugin versions for content changes; refreshing the catalogue
alone does not necessarily replace an installed plugin.

## What belongs where

| You have | Where it goes |
|---|---|
| A skill for CLIO Kit's own servers | A PR into `skills/`, not here — it names our tool names, so it has to move when those move |
| Your own MCP server, in any language | An entry here, pointing at your repo |
| Your own plugin, skills and servers together | An entry here |
| Your own marketplace | An entry here with `kind = "marketplace"` — see [Federated marketplaces](#federated-marketplaces) for what that does and does not do |

## Adding an entry

Create `entries/<name>.toml`. The filename must match the `name` field.

```toml
name        = "materials-lab"
kind        = "plugin"          # or "marketplace"; defaults to "plugin"
description = "Crystal structure and diffraction skills for materials workflows."
category    = "materials-science"
maintainer  = "some-lab"
keywords    = ["materials", "crystallography"]

[source]
type = "github"
repo = "some-lab/materials-agent-skills"
```

Then open a pull request. We review ownership, entry shape, native plugin validation, and a minimal install
and component check. Indexing does not certify all external implementation code.
Maintained CLIO contributions additionally require implementation review and
acceptance tests. External code and update ownership remain with the publisher.

## Source types

**`github`** — the whole repository is the plugin.

```toml
[source]
type = "github"
repo = "owner/repo"
ref  = "v2.0.0"                              # optional: branch or tag
sha  = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"    # optional: exact commit, wins over ref
```

**`git-subdir`** — the plugin lives in a subdirectory of a larger repository.
Fetched with a sparse clone, so a monorepo costs no more than the plugin.

```toml
[source]
type = "git-subdir"
url  = "https://github.com/acme/monorepo.git"
path = "tools/claude-plugin"
ref  = "v2.0.0"                              # optional
```

**`npm`** — a published **Claude plugin package** containing its manifest and
components. A raw npm MCP server is not a plugin: wrap its executable in
`.mcp.json` using `clio-kit plugin init my-plugin --mcp-command npx
--mcp-arg=-y --mcp-arg=@lab/server`. Go and other servers can likewise be
wrapped using their actual executable or `clio-kit server run` descriptor.
Not valid for `kind = "marketplace"`.

```toml
[source]
type     = "npm"
package  = "@acme/claude-plugin"
version  = "^2.0.0"                          # optional
registry = "https://npm.example.com"         # optional, for a private registry
```

`package` must be a name the registry resolves. A local path, folder or tarball
does not work: the client appends a version to whatever you write, so
`./my-plugin.tgz` is looked up as `./my-plugin.tgz@latest`. Test against a real
publish, even a prerelease tag, rather than a file on disk.

**Omitting `version` selects the package's latest published version.** Installed
plugin updates still follow the client's version/update rules. Bump the plugin
manifest version on each content release, refresh the marketplace, update the
plugin, and reload it. Third-party auto-updates are not enabled by default.

## Federated marketplaces

If you run your own marketplace, submit it the same way as a plugin, naming the
other kind — the command reads your `.claude-plugin/marketplace.json` and
writes the entry:

```bash
clio-kit plugin submit /path/to/your-marketplace \
  --repo some-lab/materials-marketplace --kind marketplace
```

Your catalogue needs a `metadata.description`: it is what a user reads before
adding it, and an entry without one is refused when we merge. The entry it
writes is a single file naming the whole repository:

```toml
name        = "materials-lab"
kind        = "marketplace"
description = "A materials-science catalogue: crystallography servers and skills."
maintainer  = "some-lab"

[source]
type = "github"
repo = "some-lab/materials-marketplace"
```

`clio-kit marketplace refresh --root /path/to/clio-kit` fetches each indexed
catalogue and merges its plugins into our native `marketplace.json`. Relative
plugin sources become Git subdirectory sources pinned to the fetched commit;
plugin implementations remain in their owners' repositories. The adjacent
`federation.lock.json` records provenance and supports reproducible generation.
Name conflicts fail the entire refresh; identical direct/indexed sources are
deduplicated. Removing an external entry removes its imported listings on the
next refresh. It does not silently uninstall an existing user's plugin.

```bash
clio-kit marketplace refresh --root .
claude plugin marketplace update clio-kit
claude plugin update iowarp-dev-setup@clio-kit
```

For a GitHub-hosted catalogue, maintainers can run the federation refresh
workflow and commit the resulting catalogue without publishing a new launcher.
Users must update installed plugins and reload them after refreshing. External
marketplaces must be Git repositories with `.claude-plugin/marketplace.json`;
unsupported source forms or escaping relative paths fail with a diagnostic.
`clio-kit marketplaces` also lists the original collections for direct access.

## What a skill has to clear

`clio-kit plugin validate` enforces these. They are not style preferences: a
skill's description is carried in **every** session whether or not it fires, so
a vague one is a permanent tax on every user.

**Blocking** — the submission is refused:

- frontmatter parses, and `name` matches the folder it lives in
- recorded scenarios exist (`evals.md`, or an `evals/` directory). A skill with
  none is untested by definition
- the description opens with `Use when` and names the situation, rather than
  restating what the body says
- the description carries a `Triggers on` clause quoting the literal phrases a
  user types, because that is what the match runs against

**Advisory** — reported, never used to reject:

- no `Not for X; use Y` boundary. Skills covering neighbouring ground hijack
  each other, but a first skill with nothing to collide against is legitimately
  unbounded
- a description over 500 characters, reported with its real size

`clio-kit plugin init` scaffolds a skill that already satisfies all of this, so
the starting point passes and you edit from there.

## Trying something before you index it

An entry becomes discoverable after a marketplace update, so try
a contribution locally first. Nothing below touches this repository or your own
Claude Code config.

**A bare skill folder is not installable.** A `SKILL.md` on its own — the shape
most skill catalogues publish — has no `plugin.json`, so nothing can install it.
Wrap it first:

```bash
clio-kit plugin init /tmp/trial            # scaffold a plugin
rm -rf /tmp/trial/skills/example-workflow  # drop the placeholder
cp -r <their-skill-folder> /tmp/trial/skills/
clio-kit plugin validate /tmp/trial
claude plugin validate /tmp/trial --strict
```

**Then install it into a throwaway config**, so your real one is untouched:

```bash
export CLAUDE_CONFIG_DIR="$(mktemp -d /tmp/clio-trial.XXXXXX)"
claude plugin marketplace add /tmp/trial-marketplace
claude plugin install <name>@<marketplace> --scope user
claude plugin details <name>@<marketplace>   # skills found, and what they cost
```

`plugin details` shows installed components and a context estimate. Skill
names/descriptions support selection; full bodies load when used. Judge quality
with recorded scenarios and observed results, not a token estimate alone.

**What to look at before indexing:**

- Does every skill carry recorded scenarios (`evals.md`)? A skill with none is
  untested by definition. Ours are required to have them.
- Is the description triggers-only? A description restating what the body says
  adds discovery text without helping selection.
- Does it declare boundaries against skills we already ship? Twenty skills with
  overlapping domains will hijack each other without "Not for X; use Y".
- Is its description concise and distinct from the existing skills?

## Rules the generator enforces

Generation fails, rather than publishing something broken, when:

- the filename and the `name` field disagree
- `name` starts with `clio-`, which is reserved for plugins generated from this
  repository's own servers, bundles and skills
- `name` collides with a generated plugin or another community entry
- `description` is missing — it is what a user reads before installing
- `[source]` names an unknown type, omits a field that type requires, or carries
  a field that type does not use

## Pin your source if your users need stability

Without `ref` or `sha`, an entry tracks your default branch. Publish version
bumps so users can update installed plugins after refreshing the marketplace.
Pin a source when consumers need a reviewed revision rather than that moving
branch. Imported relative sources are pinned by the federation snapshot.

## What we ask of you

Keep the repository reachable and the plugin installable. If you stop
maintaining it, open a PR removing the entry — a listing that fails to install
is worse for a user than no listing.

Nothing here is reviewed line by line on every update, and users can see that:
each entry carries `metadata.indexed`, so the catalogue distinguishes what we
maintain from what we point at.

## Contributor commands

`plugin init` creates a skills-only starter by default. Add `--agent` for a
read-only agent and `--mcp-command` / repeated `--mcp-arg` for a real MCP wrapper.
`plugin validate` checks structure; native `claude plugin validate --strict`
checks client compatibility. `plugin submit ... --output entry.toml` prepares
a reviewable entry; `plugin submit ... --open-pr` uses authenticated `gh` to
fork, create a branch, push the one-file contribution, and open its PR. No
GitHub write occurs without `--open-pr`.
