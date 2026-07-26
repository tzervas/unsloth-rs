# Versioning and releases

## Current version

**`1.0.3`** — tagged `v1.0.3`.

`unsloth-rs` is **published on crates.io**, which makes it one of a small number of repos in this
fleet whose 1.x is real rather than aspirational. That fact drives every rule below.

## This repo does **not** use `major_version_zero` — and must not

The fleet convention is `major_version_zero = true` in every `.cz.toml`, because nearly every
repo in the fleet is 0.x and that key is what keeps it there. **This repo is an exception.**

`major_version_zero` does not stop applying once a project passes 1.0. It pins the major
*permanently*, and it does so by demoting breaking changes to MINOR bumps. Measured on a fixture
at version 1.2.10:

```
BREAKING, major_version_zero = true   ->  bump: 1.2.10 -> 1.3.0   (MINOR)
BREAKING, major_version_zero absent   ->  bump: 1.2.10 -> 2.0.0   (MAJOR)
```

A dependent on `unsloth-rs = "1.0"` accepts anything `>=1.0.0, <2.0.0`. Shipping a
breaking change as a minor bump would land it **silently in every consumer's next build** — a
semver violation against a published artifact that cannot be unpublished, only yanked.

So: do not add that key here. `.cz.toml` carries the same warning inline.

## At 1.x, MAJOR is the breaking position

| Change                        | Bump      | Example |
| ----------------------------- | --------- | ------- |
| `fix:`                        | PATCH     | 1.0.3 → 1.0.4 |
| `feat:`                       | MINOR     | 1.0.3 → 1.1.0 |
| `feat!:` / `BREAKING CHANGE:` | **MAJOR** | 1.0.3 → **2.0.0** |

This is ordinary semver, and it is the *opposite* of the 0.x repos in this fleet, where the major
is pinned and MINOR carries breaking changes. If you move between repos here, check which regime
you are in before reasoning about a bump.

Consumers pin the **major**: `unsloth-rs = "1.0"` tracks every compatible release up to but
excluding `2.0.0`. Pinning an exact patch (`"1.0.3"`) in a dependency example is a mistake —
it freezes consumers out of compatible fixes.

### A major bump still needs a human

`cz bump` will happily compute `2.0.0` from a `feat!:` commit. **No agent may cut or propose a
2.0.0 release**, exactly as no agent may cut a 1.0.0 elsewhere in the fleet. A major bump against
a published package breaks every dependent by design; that is a maintainer decision, made
deliberately, with a migration note.

## Version files

[`.cz.toml`](../.cz.toml) lists every place the version appears under `version_files`, so
`cz bump` moves them together and they cannot drift:

- `Cargo.toml`
- `README.md`
- `.cz.toml` itself — `version`

Do not hand-edit any of these — run the tool:

```bash
cz bump --yes --dry-run     # show what would happen, change nothing
cz bump                     # move every version file + create the tag
cz version --project        # what this project currently claims to be
```

The `version` key in `.cz.toml` is the version cz bumps *from*, so it must track the newest
released tag. If it lags behind the tag list, the next `cz bump` re-mints a version that already
has a tag.

## Publication status — the repo is ahead of the registry

**A GitHub Release is not a registry publication.** A git tag with notes publishes nothing
consumable; only a crates.io upload produces an artifact a dependent can resolve. For this
project the two currently **disagree**:

| | Version |
| --- | --- |
| `Cargo.toml` in this repo | `1.0.3` |
| Newest git tag | `v1.0.3` |
| **crates.io `unsloth-rs`** | **`1.0.2`** |

`1.0.3` is tagged but **has never been published to crates.io**. Anyone reading the tag list
would reasonably conclude it shipped. It did not — `cargo add unsloth-rs` still resolves `1.0.2`.
When you claim a version is released, say *where*.

## Release steps

1. Land work on `dev` via a work branch — never straight to `main`.
2. `cz bump` on the release branch: this moves every version file and creates the tag locally.
3. Open the release PR `dev` → `main`. Merge with a **merge commit**, never a squash.
4. Push the tag; build the GitHub Release.
5. **Publishing to crates.io is a separate, deliberate step.** It is not automatic, and until it
   runs, the version is not released to consumers.
