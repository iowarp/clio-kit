---
name: credentials
description: 'Use when this workflow is requested: Handles API keys, tokens, and facility
  credentials (kerberos tickets, globus and scheduler tokens, ssh agents, netrc) before
  a tool needs one, after an auth error, when a new secret is supplied, or when one
  may have leaked. Not for provider target configuration; use clio-coder auth and
  target settings. Triggers on "API key", "credential missing".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/meta/credentials
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Credentials

Work with secrets without ever seeing them. Clio structurally blocks typed
reads and writes of secret-shaped paths (`.env` variants, `~/.ssh/`, `~/.aws/`,
key and pem files, netrc, kubeconfigs) by default. That net is on your side:
never attempt a shell workaround (`cat`, `less`, `head`, `xxd`, `base64`,
`printenv`, `env`, `echo $VAR`). Every attempt is audited, and any value that
slips into output lands permanently in the transcript and evidence bundles.

## Presence Check

First identify the exact credential name the task actually needs: run the
failing command or read its error output and docs. Error messages name the
variable and often the issuer; never guess a name and never assume it matches
an example. Then check presence, in strict preference order:

1. Ask the user to confirm the credential is configured (`ask_user` when
   active). Zero file access.
2. `credential_present` with the exact credential name and source. Prefer
   `source: "environment"` for process variables and pass `file: ".env"` when
   the task names an env file. It returns only present/absent and which source
   was checked.
3. `grep -sq "^NAME=" <envfile>` exactly, only if the typed tool is
   unavailable. Quiet mode, no output flags. The exit code answers the
   question; the value never enters context. The `-q` flag is the protocol,
   not a style choice.
4. There is no fourth option. Never verify by printing.

## Missing Credential

Never ask the user to paste a secret into chat. Generate a fill-in command
they run in their own terminal, and always say that typing is hidden:

```bash
printf "Enter STRESSLAB_API_KEY (typing hidden): " && read -s val && echo \
  && echo "STRESSLAB_API_KEY=$val" >> .env && echo Saved.
```

Always include where to obtain the value (issuer registration link or
instructions). After the user confirms, re-check presence with `grep -sq`.
Use `credential_present` for that re-check when it is available.

## Consuming Credentials

- Scripts read from the env file (dotenv) or the process environment.
- Never export values inline, never pass secrets as CLI arguments (argv is
  audited and receipted), never interpolate them into code or config.
- Never embed tokens in scheduler job scripts; scheduler logs and evidence
  archives outlive the job.
- Provider keys that Clio manages live in the auth store: route those through
  `clio-coder auth` and target settings, not env files.

## Facility and HPC Surfaces

- Kerberos: `klist -s` checks ticket presence by exit code only.
- SSH: `ssh-add -l` lists fingerprints, not keys; that is safe. Never copy
  private keys between machines as part of a task.
- Globus, scheduler, and facility tokens follow the same env-file protocol.
- `.netrc` for data transfer is zero-access by default; leave it that way.

## Leak Containment

If a secret value appears in output or context anyway:

1. Stop the current task line immediately.
2. Name which credential leaked and where it landed (transcript, evidence,
   receipt) without repeating the value.
3. Tell the user to rotate it at the issuer now; the leaked value must be
   treated as burned.
4. Warn that deleting messages does not unshare an exported bundle; anything
   already exported carries the value.
5. Resume only after the user acknowledges.

## Worked Example

All names below are illustrative; use the credential name, env file, and
issuer that your task actually reports.

Task: run `fetch_data.py`. Running it fails with "missing STRESSLAB_API_KEY
(register at https://stresslab.example/keys)", which names the credential.

1. Presence: user is active, so ask "Is STRESSLAB_API_KEY configured in .env?"
   User is unsure. Use `credential_present` with `name: "STRESSLAB_API_KEY"`
   and `file: ".env"`; it reports absent.
2. Missing: send the `read -s` fill-in command above plus the issuer link
   ("create a key at https://stresslab.example/keys"). User runs it, confirms.
3. Re-check: `credential_present` reports present from `.env`. Run the script;
   it loads the key via dotenv. The value never appeared in chat, argv, or
   logs.
4. If the script had printed the key in a stack trace: stop, report
   "STRESSLAB_API_KEY leaked into the transcript via the traceback", instruct
   rotation at the issuer, warn about exported evidence, wait for
   acknowledgment.

## Gotchas

- An auth failure is not evidence the key is absent. Check presence first,
  then scope and expiry through the issuer's dashboard or a harmless API
  probe, never by printing the value.
- `grep VAR file` without `-q` prints the matching line. That is a leak.
- A "temporary" `echo $KEY` for debugging is permanent in the transcript.
- Rotating after a leak is not optional; assume anything in context is
  compromised.
