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
feature/* → testing → dev → main
```

### 1. Testing Branch

The `testing` branch is where all new features undergo:
- Comprehensive code formatting
- Linting checks
- Security audits and patching
- Integration testing
- Quality assurance validation

### 2. Development Branch

Once validated in `testing`, changes are promoted to `dev` for:
- Integration with other features
- Extended testing
- Pre-release validation

### 3. Main Branch

Production-ready changes are merged from `dev` to `main` for:
- Official releases
- Stable deployments

## PR #16 Target Update

### Current State
- **PR #16**: `feature/unsloth-rs/ternary-gpu-kernels` → `feature/unsloth-rs/flash-attention-completion`
- The target branch (`flash-attention-completion`) contains 38 commits of work

### Required Change
PR #16 should be retargeted to the `testing` branch to follow the proper workflow:
- **New target**: `testing`
- **Reason**: All new features must undergo comprehensive QA in the testing branch before proceeding to dev

### Steps to Update PR #16

Since the target branch contains significant changes, we need to:

1. **Update PR #16 base branch** to `testing` (requires GitHub web UI or API)
   - Go to PR #16 on GitHub
   - Click "Edit" next to the base branch
   - Change from `feature/unsloth-rs/flash-attention-completion` to `testing`

2. **After PR #16 merges to testing**, create a new PR:
   - **From**: `testing`
   - **To**: `dev`
   - This PR will include all validated changes from testing

3. **After testing → dev merge**, create final PR:
   - **From**: `dev`
   - **To**: `main`
   - This promotes stable changes to production

## Protected Branches

The following branches must be preserved and never deleted:
- `main`
- `dev`
- `testing`
- `experimental`

## Quality Gates

### Testing Branch Requirements
- [ ] Code formatting (rustfmt)
- [ ] Linting (clippy)
- [ ] Security audit (cargo audit)
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updates

### Dev Branch Requirements
- [ ] All testing requirements met
- [ ] Feature integration validated
- [ ] Performance benchmarks acceptable
- [ ] Cross-feature compatibility verified

### Main Branch Requirements
- [ ] All dev requirements met
- [ ] Release notes prepared
- [ ] Version bumped appropriately
- [ ] Final QA sign-off
