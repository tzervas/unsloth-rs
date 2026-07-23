# Publishing to crates.io

## Publication Status

| Field | Value |
|-------|--------|
| **Cargo.toml version** | `1.0.3` |
| **Git tags (local/origin)** | `v1.0.0`, `v1.0.1`, `v1.0.2` (and older alpha); **tag `v1.0.3` after publish** |
| **docs.rs** | Builds for 1.0.2 historically; expect 1.0.3 after publish |
| **crates.io tarball 1.0.2** | **Broken** — dual-path case collision (`ROADMAP.md` + `roadmap.md`) |
| **Fix in this tree (1.0.3)** | Only `ROADMAP.md` remains; `cargo package` succeeds |
| **Last docs refresh** | 2026-07-22 (gap-close Wave-3/4 honesty + packaging) |

**Do not `cargo publish` from this workstream unless the release train explicitly opens a publish gate.** Prefer dry-run / `--list` only.

## Product positioning (must stay honest)

- Crate is **Candle/CubeCL transformer building blocks**, not a full Unsloth product port.
- Do **not** advertise Unsloth “2× training” or “~70% VRAM” claims for this crate without measured evidence from this repo.
- Training/LoRA lives in sister crates (peft-rs, qlora-rs, axolotl-rs).

## Pre-publication checklist

### Metadata

- [x] `Cargo.toml`: name, version, authors, description, license, repository, keywords, categories
- [x] `README.md` with honest scope + non-goals
- [x] `LICENSE` (MIT)
- [x] `CHANGELOG.md` section for **1.0.3**
- [x] Packaging case fix verified (`cargo package --allow-dirty`)

### Packaging (case collision)

crates.io packages on a case-sensitive archive model that still rejects
case-colliding paths (Windows / macOS consumers). **Never ship both
`ROADMAP.md` and `roadmap.md`.**

```bash
# Must succeed and list only one roadmap path
cargo package --allow-dirty --list | rg -i roadmap

# Dry-run package (does not publish)
cargo package --allow-dirty
```

If `cargo package` reports a duplicate-path conflict, stop — do not publish.

### Quality gates (CPU default)

```bash
cargo test --workspace --no-default-features
cargo test --workspace   # default features (currently empty default)
cargo clippy --all-targets -- -D warnings   # optional stricter gate
cargo fmt -- --check
cargo doc --no-deps
```

### CUDA (optional; not required for crates.io CPU package)

GPU builds are **environment-gated**. Missing device nodes or toolkit/arch
mismatch is **`FAIL_ENV`**, not a green CI pass. See [GPU_SETUP.md](GPU_SETUP.md)
and [DEBT.md](DEBT.md).

```bash
# Often required when host reports CC 12.0 but nvcc is older (≤ sm_90)
CUDA_COMPUTE_CAP=90 cargo check --features cuda
```

## GPG signing (maintainer)

- **Name**: Tyler Zervas
- **Email**: tz-dev@vectorweight.com
- **Username**: tzervas

```bash
gpg --list-secret-keys --keyid-format=long tz-dev@vectorweight.com
git config user.signingkey <YOUR_GPG_KEY_ID>
git config commit.gpgsign true
git config tag.gpgSign true
```

### Signed tags

```bash
git tag -s v1.0.3 -m "v1.0.3: packaging case fix + docs honesty"
git push origin v1.0.3
git tag -v v1.0.3
```

## Publishing workflow (when release train opens)

### 1. Prepare

1. Confirm `Cargo.toml` version **1.0.3** and `CHANGELOG.md` section.
2. Confirm single roadmap filename and honest README.
3. Run CPU tests + `cargo package --allow-dirty --list`.

### 2. Package verify

```bash
cargo package --allow-dirty
cargo package --list
ls -lh target/package/unsloth-rs-*.crate
```

### 3. Publish (only with explicit gate)

```bash
# Requires crates.io token; do not run from gap-close agents by default
cargo publish --dry-run
# cargo publish
```

### 4. Post-publish

- Confirm crates.io page unpacks without path errors.
- Confirm docs.rs build for the new version.
- Note any remaining GPU/env debt in DEBT.md (do not claim GPU green).
- Optional: human-only yank of broken **1.0.2** after 1.0.3 is confirmed live.

## Historical failure mode (1.0.2)

```
Invalid path 'unsloth-rs-1.0.2/roadmap.md':
Duplicate path conflicts with 'ROADMAP.md'
```

**Root cause:** two roadmap files differing only by case.
**Remediation:** keep `ROADMAP.md` only; ternary phased notes live under
`TERNARY_GPU_IMPLEMENTATION.md` / other docs — not a second `roadmap.md`.

## References

- [GPU_SETUP.md](GPU_SETUP.md) — `CUDA_COMPUTE_CAP`, FAIL_ENV
- [DEBT.md](DEBT.md) — residual GPU gates
- [CHANGELOG.md](CHANGELOG.md)
