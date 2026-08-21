# Versioning and releases

## Source of truth

The live crate version is **`Cargo.toml` `package.version`**.
`.cz.toml` `tool.commitizen.version` must match it.

```bash
cz version --project
```

This file is **not** a pin. Do not copy a `x.y.z` from here into `Cargo.toml`.
When you claim a version shipped, say *where* (git tag vs crates.io). A GitHub
Release is not a registry publication.

`unsloth-rs` is published on crates.io, so 1.x is real. That drives the rules
below.

## This repo does not use `major_version_zero`

The fleet default is `major_version_zero = true` because most repos are 0.x.
**Do not add that key here.** It pins major forever and demotes BREAKING to
MINOR. A dependent on `unsloth-rs = "1"` accepts `>=1.0.0, <2.0.0`. At 1.x,
MAJOR is the breaking position. No agent may cut 2.0.0.

```
BREAKING, major_version_zero = true   ->  1.2.10 -> 1.3.0   (MINOR)
BREAKING, major_version_zero absent   ->  1.2.10 -> 2.0.0   (MAJOR)
```

## Bump table (ordinary semver)

| Change                        | Bump      | Example           |
| ----------------------------- | --------- | ----------------- |
| `fix:`                        | PATCH     | 1.0.4 → 1.0.5     |
| `feat:`                       | MINOR     | 1.0.4 → 1.1.0     |
| `feat!:` / `BREAKING CHANGE:` | **MAJOR** | 1.0.4 → **2.0.0** |

Examples are illustrative. Consumers pin the major: `unsloth-rs = "1"`.
Do not pin an exact patch in install examples.

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
cz bump --yes --increment patch --files-only
```

`--files-only` updates version files without creating a local tag.
Do **not** pass `--changelog`: commitizen rewrites Keep a Changelog into
its own format. Move `## [Unreleased]` into `## [x.y.z]` by hand.
Tags are created by [`.github/workflows/release.yml`](../.github/workflows/release.yml)
after the bump PR merges (`bump=none`). `main` requires PRs;
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
`publish_crate=true` and `CARGO_REGISTRY_TOKEN` is set.

## Publication status

Read **`Cargo.toml`** and `git tag --list 'v*'` — not this paragraph. Historical
registry notes: crates.io `1.0.2` tarball had a case collision (`ROADMAP.md` +
`roadmap.md`); `1.0.3` was tagged. A git tag is not a crates.io upload.
