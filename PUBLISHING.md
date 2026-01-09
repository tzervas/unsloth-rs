# Publishing to crates.io

## Publication Status

**Current Version**: `0.1.0` (pre-release)
**Publication Status**: Ready for alpha/beta release
**Last Updated**: January 9, 2026

## Pre-Publication Checklist

### ✅ Completed Items
- [x] Cargo.toml metadata (name, version, authors, description, license, repository, keywords, categories)
- [x] README.md with usage examples and status badges
- [x] LICENSE file (MIT)
- [x] Comprehensive test suite (114 unit tests + 34 integration tests)
- [x] Documentation (`cargo doc` generates full API docs)
- [x] Benchmarking suite
- [x] Quality gates (86% clippy warning reduction)

### 📋 Pre-Publication Requirements

1. **Version Tagging**
   - Decide on version number (suggest `0.1.0-alpha.1` for first alpha release)
   - Update `Cargo.toml` version field
   - Create git tag: `git tag -s v0.1.0-alpha.1 -m "Alpha release 0.1.0-alpha.1"`

2. **Documentation Review**
   - Ensure all public APIs have documentation
   - Run `cargo doc --open` to verify docs render correctly
   - Add examples to critical functions

3. **Final Testing**
   ```bash
   cargo test --all-features
   cargo clippy --all-features -- -D warnings
   cargo fmt -- --check
   cargo bench
   ```

4. **Package Verification**
   ```bash
   cargo package --allow-dirty
   cargo package --list
   ```

## GPG Signing Configuration

### Maintainer Identity
- **Name**: Tyler Zervas
- **Email**: tz-dev@vectorweight.com
- **Username**: tzervas
- **GPG Key**: Required for signed releases

### Setting Up GPG Signing

1. **Verify GPG Key Exists**
   ```bash
   gpg --list-secret-keys --keyid-format=long tz-dev@vectorweight.com
   ```

2. **Create GPG Key (if needed)**
   ```bash
   gpg --full-generate-key
   # Select: RSA and RSA (default)
   # Key size: 4096
   # Expiration: 0 (no expiration) or your preference
   # Real name: Tyler Zervas
   # Email: tz-dev@vectorweight.com
   # Comment: unsloth-rs package signing
   ```

3. **Configure Git to Use GPG**
   ```bash
   # Get your GPG key ID
   gpg --list-secret-keys --keyid-format=long tz-dev@vectorweight.com
   
   # Configure git globally
   git config --global user.signingkey <YOUR_GPG_KEY_ID>
   git config --global commit.gpgsign true
   git config --global tag.gpgSign true
   
   # Or configure for this repo only
   git config user.signingkey <YOUR_GPG_KEY_ID>
   git config commit.gpgsign true
   git config tag.gpgSign true
   ```

4. **Export Public Key to GitHub/GitLab**
   ```bash
   gpg --armor --export tz-dev@vectorweight.com
   ```
   Then add to GitHub: Settings → SSH and GPG keys → New GPG key

### Signed Git Tags

Create signed release tags:
```bash
# Alpha release
git tag -s v0.1.0-alpha.1 -m "Alpha release 0.1.0-alpha.1

- Initial alpha release
- Core transformer building blocks
- CPU reference implementations
- GPU infrastructure ready
- 114 unit tests + 34 integration tests passing"

# Push tag
git push origin v0.1.0-alpha.1

# Verify signature
git tag -v v0.1.0-alpha.1
```

## Publishing Workflow

### Step 1: Prepare Release

1. **Update Version**
   ```bash
   # Update Cargo.toml version field
   # For alpha: 0.1.0-alpha.1
   # For beta: 0.1.0-beta.1
   # For stable: 0.1.0
   ```

2. **Update CHANGELOG.md** (create if needed)
   ```markdown
   # Changelog
   
   ## [0.1.0-alpha.1] - 2026-01-09
   
   ### Added
   - Multi-head attention with GQA support
   - Flash Attention infrastructure (CPU fallback ready)
   - Ternary quantization system
   - RoPE, RMSNorm, SwiGLU implementations
   - Memory tracking and VRAM estimation
   - Mixed precision training support
   - Comprehensive test suite (148 tests)
   
   ### Notes
   - Alpha release for early testing
   - GPU kernels require CUDA feature flag
   - Production use not recommended yet
   ```

3. **Final Quality Check**
   ```bash
   cargo test --all-features
   cargo clippy --all-features -- -D warnings
   cargo doc --no-deps --open
   ```

### Step 2: Package and Verify

```bash
# Create package (dry run)
cargo package --allow-dirty

# Verify package contents
cargo package --list

# Check package size
ls -lh target/package/unsloth-rs-*.crate
```

### Step 3: Publish to crates.io

You're already logged in via `cargo login`. To publish:

```bash
# Publish (dry run first)
cargo publish --dry-run

# Actual publish
cargo publish

# For alpha/beta releases, consider:
cargo publish --allow-dirty  # if there are uncommitted changes
```

### Step 4: Post-Publication

1. **Create GitHub Release**
   - Go to repository releases page
   - Create release from tag `v0.1.0-alpha.1`
   - Add release notes from CHANGELOG.md
   - Mark as "pre-release" for alpha/beta

2. **Verify Publication**
   ```bash
   # Check on crates.io
   open https://crates.io/crates/unsloth-rs
   
   # Test installation
   cargo install unsloth-rs --version 0.1.0-alpha.1
   ```

3. **Announce**
   - Update README.md with installation instructions
   - Consider announcing in Rust ML communities if appropriate

## Versioning Strategy

### Alpha Releases (0.1.0-alpha.x)
- **Purpose**: Early testing, API exploration, feedback gathering
- **Stability**: APIs may change significantly
- **Target Users**: Early adopters, contributors
- **Recommended For**: Testing, experimentation, feedback

### Beta Releases (0.1.0-beta.x)
- **Purpose**: Feature-complete, API stabilization
- **Stability**: APIs unlikely to change (but possible)
- **Target Users**: Early production testing
- **Recommended For**: Pre-production validation

### Stable Releases (0.1.0+)
- **Purpose**: Production use
- **Stability**: Semantic versioning (breaking changes → major version)
- **Target Users**: Production deployments
- **Recommended For**: Production workloads

## Current Recommendation

**Suggested First Release**: `0.1.0-alpha.1`

**Rationale**:
- ✅ Core functionality implemented and tested (148 tests passing)
- ✅ Quality gates established (86% clippy improvement)
- ✅ Documentation complete
- ⚠️ GPU kernels still in development (CubeCL integration)
- ⚠️ Performance optimization ongoing
- ⚠️ API may evolve based on feedback

**Alpha Release Notes**:
```
unsloth-rs 0.1.0-alpha.1 - Early Alpha Release

This is an early alpha release for testing and feedback. The crate provides
Rust implementations of transformer building blocks:

✅ Working:
- Multi-head attention (CPU reference + Candle CUDA backend)
- RoPE, RMSNorm, SwiGLU
- Ternary quantization system
- Memory tracking and VRAM estimation
- Mixed precision training support
- Comprehensive test suite (148 tests)

🚧 In Development:
- Flash Attention CubeCL GPU kernels
- Performance optimization
- Additional fused operations

⚠️ Note: This is an alpha release. APIs may change. Not recommended for 
production use yet. Feedback welcome!
```

## Quick Publish Commands

When ready to publish:

```bash
# 1. Update version in Cargo.toml to "0.1.0-alpha.1"
# 2. Commit changes
git add -A
git commit -S -m "Release 0.1.0-alpha.1"

# 3. Create signed tag
git tag -s v0.1.0-alpha.1 -m "Alpha release 0.1.0-alpha.1"

# 4. Push
git push origin testing
git push origin v0.1.0-alpha.1

# 5. Publish to crates.io
cargo publish

# 6. Create GitHub release from tag
```

## Contact

- **Maintainer**: Tyler Zervas (tzervas)
- **Email**: tz-dev@vectorweight.com
- **Repository**: https://github.com/tzervas/unsloth-rs
