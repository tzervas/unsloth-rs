# Branch Workflow and PR Management

## Branch Structure

This repository maintains four permanent branches that should **never be deleted**:

- `main` - Production-ready releases
- `dev` - Development integration branch
- `testing` - Quality assurance and validation branch
- `experimental` - Experimental features

## Workflow

All feature branches should follow this workflow:

```
feature/* → experimental → testing → dev → main
```

### 1. Experimental Branch

The `experimental` branch is where all new features are first merged:
- Initial feature integration
- Early validation
- Experimental feature testing
- Proof of concept validation

### 2. Testing Branch

Once features are validated in `experimental`, they are promoted to `testing` for:
- Comprehensive code formatting
- Linting checks
- Security audits and patching
- Integration testing
- Quality assurance validation
- Bug fixes and refinements

### 3. Development Branch

Once validated in `testing`, changes are promoted to `dev` for:
- Integration with other features
- Extended testing
- Pre-release validation
- Staging environment deployment

### 4. Main Branch

Production-ready changes are merged from `dev` to `main` for:
- Official releases
- Stable deployments
- Production environment

## PR #16 Workflow

### Current State
- **PR #16**: `feature/unsloth-rs/ternary-gpu-kernels` → `experimental` (✅ Updated)
- Following the correct workflow for new features

### Promotion Path

The feature will follow this promotion path:

1. **Phase 1: Merge to Experimental** (Current)
   - **PR #16**: `feature/unsloth-rs/ternary-gpu-kernels` → `experimental`
   - Initial feature integration and validation
   
2. **Phase 2: Promote to Testing** (After Phase 1)
   - Create new PR: `experimental` → `testing`
   - Comprehensive formatting, linting, and security audit
   - Bug fixes and quality improvements
   - Test suite validation

3. **Phase 3: Promote to Dev** (After Phase 2)
   - Create new PR: `testing` → `dev`
   - Feature integration validation
   - Cross-feature compatibility testing
   
4. **Phase 4: Release to Main** (After Phase 3)
   - Create new PR: `dev` → `main`
   - Production release

## Protected Branches

The following branches must be preserved and never deleted:
- `main`
- `dev`
- `testing`
- `experimental`

## Quality Gates

### Experimental Branch Requirements
- [ ] Feature compiles successfully
- [ ] Basic functionality demonstrated
- [ ] No critical breaking changes
- [ ] Initial testing completed

### Testing Branch Requirements
- [ ] Code formatting (rustfmt)
- [ ] Linting (clippy)
- [ ] Security audit (cargo audit)
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updates
- [ ] Performance regression checks

### Dev Branch Requirements
- [ ] All testing requirements met
- [ ] Feature integration validated
- [ ] Performance benchmarks acceptable
- [ ] Cross-feature compatibility verified
- [ ] Staging deployment successful

### Main Branch Requirements
- [ ] All dev requirements met
- [ ] Release notes prepared
- [ ] Version bumped appropriately
- [ ] Final QA sign-off
- [ ] Production readiness confirmed
