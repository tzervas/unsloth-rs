# Fleet standards (tzervas)

Applied from the workstation pack under `plans/fleet-standards/pack/`.

## Workflows

| Workflow | When | Runner |
|----------|------|--------|
| `fleet-ci.yml` | push/PR to main|dev | thin caller of `tzervas/ap-workflows/.github/workflows/reusable-ci-autodetect.yml@v0.1` on `[self-hosted, linux, x64, podman, rust]`; hosted `Test Suite (GitHub-hosted)` on `ubuntu-latest` (no `needs: detect`) |
| `fleet-security.yml` | push/PR + weekly | same labels (`rust` selects the fleet work image) |
| `close-issues-on-main.yml` | PR closed→main | GitHub-hosted (API-only) |
| `reopen-issues-closed-off-main.yml` | PR merged off-main with Closes | same |

Action pins follow `tzervas/ap-workflows` `pins/actions.yml` (`actions/checkout@v7`). Do **not** `--all-features` (pulls `cuda`).

The `rust` `runs-on` label is the composition point for the fleet work image. See [ap-workflows RUNNER-IMAGES](https://github.com/tzervas/ap-workflows/blob/main/docs/RUNNER-IMAGES.md).

## Issue close policy

- **`dev` / feature merges:** `Refs #n` only — issues stay open
- **`main` merges:** `Closes #n` / `Fixes #n`
- **Epics:** close only when delivery PR to main includes `Closes #<epic>`

## Badges

README badges use GitHub Actions SVG for **trunk** branch — live status, not static green.

## Copilot

Automatic Copilot code reviews are **disabled** for fleet-managed repos. Do not request Copilot on PRs.

## Permissions

Workflows use minimum `permissions:` blocks (contents read; issues write only for close/reopen jobs).
