# Branch Management & Merge Deconfliction Strategy

**Last Updated**: 2026-01-10  
**Status**: Active Development  
**Current Branch**: `experimental`

## Overview

This document defines the branching strategy, merge order, and conflict resolution patterns for unsloth-rs to enable parallel development of Phases 2-5 while minimizing merge conflicts.

## Branch Structure

```
main (stable releases)
  └── dev (integration testing)
      └── experimental (active development)
          ├── feature/* (short-lived feature branches)
          └── phase/* (phase-specific development)
```

### Branch Purposes

- **`main`**: Stable releases only. Tagged versions. Protected.
- **`dev`**: Integration testing branch. Merges from experimental after validation.
- **`experimental`**: Active development branch. All features merge here first.
- **`feature/*`**: Short-lived branches for specific tasks (1-7 days).
- **`phase/*`**: Longer-lived branches for phase implementations (1-4 weeks).

## Current Development Phases

| Phase | Focus | Shared Files Risk | Status |
|-------|-------|-------------------|--------|
| Phase 2 | GPU Ternary Matmul | Medium | ✅ Complete |
| Phase 3 | Ternary Attention | Medium | ✅ Complete |
| Phase 4 | Advanced Sparsity | Low | ✅ Complete |
| Phase 5 | End-to-end Integration | High | 🔄 In Progress |
| Flash Attention Phase 2 | GPU Profiling | Low | 🔄 Ready for RTX 5080 |
| CI/CD | GitHub Actions | Low | ✅ Complete |

## Shared File Inventory

### High-Conflict Risk Files

These files are frequently modified across multiple phases:

1. **`Cargo.toml`** - Dependencies, features, metadata
   - **Conflict Pattern**: New dependencies, feature flags
   - **Resolution**: Merge dev → experimental daily
   
2. **`src/kernels/mod.rs`** - Module exports
   - **Conflict Pattern**: New module additions
   - **Resolution**: Coordinate module names in TASKS.md
   
3. **`src/kernels/cubecl/interop.rs`** - Tensor conversion utilities
   - **Conflict Pattern**: New conversion functions
   - **Resolution**: Add functions at end of file, use descriptive names
   
4. **`README.md`** - Feature documentation
   - **Conflict Pattern**: Status updates, feature lists
   - **Resolution**: Update only in experimental branch

### Medium-Conflict Risk Files

5. **`src/kernels/ternary/mod.rs`** - Ternary module exports
6. **`src/kernels/ternary/types.rs`** - Shared type definitions
7. **`tests/integration.rs`** - Integration tests
8. **`benches/kernels.rs`** - Benchmarks

### Low-Conflict Risk Files

- Individual kernel implementations (`attention.rs`, `matmul.rs`, etc.)
- Documentation files (`ROADMAP.md`, `TASKS.md`, etc.)
- Configuration presets in `src/kernels/cubecl/config.rs`

## Merge Order & Frequency

### Daily Merges (Fast-Moving)

```
feature/* → experimental (as completed)
```

- Merge immediately when feature is complete and tested
- Use GitHub PR with at least 1 approval
- Run full test suite before merge

### Weekly Merges (Integration)

```
experimental → dev (every Friday, or when milestone complete)
```

- Full regression testing required
- All tests must pass (148+ tests)
- Documentation updated
- CHANGELOG.md updated

### Release Merges (Stable)

```
dev → main (on version release only)
```

- All acceptance criteria met
- Performance benchmarks validated
- Documentation complete
- Version tagged (e.g., `v0.2.0`)

## Conflict Resolution Patterns

### Pattern 1: Cargo.toml Conflicts

**Scenario**: Two branches add different dependencies

```toml
# Branch A adds:
[dependencies]
new-dep-a = "1.0"

# Branch B adds:
[dependencies]
new-dep-b = "2.0"
```

**Resolution**:
1. Accept both dependencies
2. Maintain alphabetical order within each section
3. Group related dependencies (candle*, cubecl*)

### Pattern 2: Module Export Conflicts

**Scenario**: Two branches add different modules

```rust
// Branch A adds:
pub mod new_kernel_a;

// Branch B adds:
pub mod new_kernel_b;
```

**Resolution**:
1. Accept both additions
2. Maintain alphabetical order
3. Add tests for both modules

### Pattern 3: Type Definition Conflicts

**Scenario**: Two branches modify the same struct

```rust
// Branch A adds:
pub struct Config {
    existing_field: u32,
    new_field_a: f32,
}

// Branch B adds:
pub struct Config {
    existing_field: u32,
    new_field_b: bool,
}
```

**Resolution**:
1. Accept both new fields
2. Maintain logical grouping
3. Update Default impl to include both fields
4. Add tests for new fields

### Pattern 4: Documentation Conflicts

**Scenario**: README.md status updates differ

**Resolution**:
1. Use experimental branch version as source of truth
2. Merge status from both branches
3. Remove duplicate entries
4. Maintain consistent formatting

## Conflict Prevention Strategies

### 1. Communication

- **Task Coordination**: Use TASKS.md to claim work
- **File Ownership**: Announce in comments when modifying shared files
- **Daily Sync**: Review experimental branch commits daily

### 2. Code Organization

- **File Modularity**: Keep kernels in separate files
- **Minimal Coupling**: Avoid cross-phase dependencies
- **Interface Stability**: Don't change public APIs mid-phase

### 3. Testing Requirements

- **Per-Feature Tests**: Add tests in same PR as feature
- **Integration Tests**: Add to `tests/integration.rs` at end of file
- **Benchmark Isolation**: Use separate benchmark functions

### 4. Merge Hygiene

- **Small PRs**: Target <500 LOC per PR when possible
- **Frequent Merges**: Don't let branches drift >3 days
- **Rebase Often**: Rebase feature branches on experimental daily

## Phase-Specific Merge Plans

### Phase 2: GPU Ternary Matmul ✅ Complete

**Status**: Merged to experimental  
**Key Files Modified**:
- `src/kernels/ternary/matmul_cubecl.rs` (new)
- `src/kernels/ternary/mod.rs` (exports)
- `tests/integration.rs` (tests added at end)

### Phase 3: Ternary Attention ✅ Complete

**Status**: Merged to experimental  
**Key Files Modified**:
- `src/kernels/ternary/attention_cubecl.rs` (new)
- `src/kernels/ternary/attention.rs` (modified)
- `src/kernels/ternary/mod.rs` (exports)

### Phase 4: Advanced Sparsity ✅ Complete

**Status**: Merged to experimental  
**Key Files Modified**:
- `src/kernels/ternary/matmul_cubecl.rs` (sparse kernel added)
- `src/kernels/ternary/types.rs` (sparsity metadata)

### Phase 5: End-to-end Integration 🔄 In Progress

**Target Merge**: When GPU validation complete  
**Key Files**:
- `src/kernels/ternary/model.rs` (integration layer)
- `tests/integration.rs` (end-to-end tests)
- `README.md` (updated examples)

**Merge Strategy**:
1. Wait for Flash Attention Phase 2 GPU validation
2. Merge both simultaneously to avoid conflicts
3. Update all documentation in single commit

### Flash Attention Phase 2: GPU Profiling 🔄 Ready

**Target Merge**: When RTX 5080 profiling complete  
**Key Files**:
- `benches/kernels.rs` (GPU benchmarks)
- `HANDOFF.md` (performance results)
- `BENCHMARKING.md` (new file)

**Merge Strategy**:
1. Profile on RTX 5080 first
2. Validate on RTX 3090 Ti
3. Document results before merge
4. No conflicts expected (adds new files only)

## CI/CD Integration

### Pre-Merge Checks (GitHub Actions)

✅ All checks must pass before merge:

1. **Tests**: All 148+ tests pass
2. **Clippy**: No warnings with `-D warnings`
3. **Formatting**: `cargo fmt --check` passes
4. **Build**: Builds with all feature combinations
5. **Docs**: `cargo doc` succeeds without warnings

### Merge Automation (Future)

Planned for Issue #9:

- Auto-merge dependabot PRs if tests pass
- Auto-rebase feature branches on experimental
- Conflict detection bot
- Merge queue for experimental branch

## Emergency Conflict Resolution

If a merge conflict cannot be resolved easily:

1. **Create conflict resolution branch**:
   ```bash
   git checkout -b conflict-resolution/describe-conflict experimental
   git merge --no-commit feature/conflicting-branch
   ```

2. **Analyze conflict**:
   - Review both versions
   - Consult original PR authors
   - Check TASKS.md for context

3. **Resolve and test**:
   - Apply resolution patterns above
   - Run full test suite
   - Add regression tests if needed

4. **Document resolution**:
   - Add comment to PR explaining resolution
   - Update this document if new pattern discovered

## Monitoring & Metrics

Track merge health with these metrics:

- **Merge Frequency**: Feature branches should merge within 3 days
- **Conflict Rate**: Target <10% of merges have conflicts
- **Resolution Time**: Conflicts resolved within 1 day
- **Test Pass Rate**: Maintain 100% on experimental branch

## Review Checklist

Before merging any PR to experimental:

- [ ] All tests pass locally
- [ ] CI/CD checks pass
- [ ] No merge conflicts with experimental
- [ ] Shared files updated following patterns above
- [ ] Documentation updated (if public API changed)
- [ ] CHANGELOG.md entry added
- [ ] At least 1 reviewer approval

## Additional Resources

- **ROADMAP.md**: Strategic development plan
- **TASKS.md**: Detailed task list with ownership
- **ISSUE_STATUS.md**: GitHub issue tracking
- **HANDOFF.md**: Phase 2 GPU profiling tasks

## Questions?

For questions about branch strategy or merge conflicts:
1. Review this document
2. Check TASKS.md for current work
3. Check experimental branch commit history
4. Open GitHub discussion if needed

---

**Document Owner**: Project Maintainer  
**Review Frequency**: Weekly during active development  
**Next Review**: When Phase 5 begins merge process
