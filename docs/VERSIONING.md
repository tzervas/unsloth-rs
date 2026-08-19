# Versioning and releases

## Current version

**`1.0.4`** in `Cargo.toml` / `.cz.toml`. Newest git tag as of this file:
`v1.0.3`. Tag `v1.0.4` via the release workflow (`bump=none`), not by hand.

`unsloth-rs` is **published on crates.io**, so 1.x is real. That drives every
rule below.

## This repo does **not** use `major_version_zero`

The fleet default is `major_version_zero = true` because most repos are 0.x.
**Do not add that key here.** It pins major forever and demotes BREAKING to
MINOR. A dependent on `unsloth-rs = "1.0"` would silently take a breaking
1.1.0. At 1.x, MAJOR is the breaking position. No agent may cut 2.0.0.

## Bump table (ordinary semver)

| Change                        | Bump      | Example           |
| ----------------------------- | --------- | ----------------- |
| `fix:`                        | PATCH     | 1.0.4 → 1.0.5     |
| `feat:`                       | MINOR     | 1.0.4 → 1.1.0     |
| `feat!:` / `BREAKING CHANGE:` | **MAJOR** | 1.0.4 → **2.0.0** |

Consumers pin the major: `unsloth-rs = "1.0"`. Do not pin an exact patch in
install examples.

## Version files

[`.cz.toml`](../.cz.toml) `version_files` must move together:

- `Cargo.toml`
- `README.md` (`**Version:**`)
- `PUBLISHING.md` (status table)
- `.cz.toml` `version`

Do not hand-edit those — use the tool:

```bash
cz version --project          # what cz thinks we are
cz bump --yes --dry-run       # show the next bump, change nothing
cz bump --yes --increment patch --files-only --changelog
```

`--files-only` updates files and changelog without creating a local tag.
Tags are created by [`.github/workflows/release.yml`](../.github/workflows/release.yml)
after the bump PR merges (`bump=none`). `main` requires PRs (`protec-main`);
a `cz bump` that pushes a tag+commit straight to `main` is rejected.

## Release workflow

One dispatch, idempotent. Default is a **dry run**.

```bash
# Plan only (default)
gh workflow run release.yml -R tzervas/unsloth-rs

# Tag + GitHub Release for the version already in Cargo.toml (no bump)
gh workflow run release.yml -R tzervas/unsloth-rs \
  -f bump=none -f dry_run=false -f publish_crate=false

# Next patch: open a bump PR (does not tag). Merge it, then bump=none.
gh workflow run release.yml -R tzervas/unsloth-rs \
  -f bump=patch -f dry_run=false -f publish_crate=false
```

`bump=major` also needs `allow_major=true`. crates.io publish is **off** unless
`publish_crate=true` and `CARGO_REGISTRY_TOKEN` is set. A GitHub Release is not
a registry publication.

## Publication status

| Surface | Version |
| --- | --- |
| `Cargo.toml` / `.cz.toml` | `1.0.4` |
| Newest git tag (before this release workflow) | `v1.0.3` |
| crates.io `unsloth-rs` | historically `1.0.2` (1.0.2 tarball had a case collision; 1.0.3 was tagged and never published) |

When you claim a version is released, say *where*.
