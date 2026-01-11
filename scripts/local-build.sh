#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright 2026 Tyler Zervas

set -euo pipefail

# Local build and test script for GPU validation
# Performs builds that would be too expensive/complex for CI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔨 unsloth-rs Local Build Script"
echo "=================================="
echo ""

# Parse arguments
BUILD_TYPE="${1:-all}"

function check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  Warning: nvidia-smi not found, CUDA tests will be skipped"
        return 1
    fi
    echo "✓ CUDA available:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    return 0
}

function build_cpu() {
    echo "📦 Building CPU version..."
    cargo build --release
    echo "✓ CPU build complete"
}

function build_cuda() {
    if check_cuda; then
        echo "📦 Building CUDA version..."
        cargo build --release --features cuda
        echo "✓ CUDA build complete"
    else
        echo "⏭️  Skipping CUDA build (no CUDA available)"
    fi
}

function test_cpu() {
    echo "🧪 Running CPU tests..."
    cargo test --release
    echo "✓ CPU tests passed"
}

function test_cuda() {
    if check_cuda; then
        echo "🧪 Running CUDA tests..."
        cargo test --release --features cuda
        echo "✓ CUDA tests passed"
    else
        echo "⏭️  Skipping CUDA tests (no CUDA available)"
    fi
}

function bench_cpu() {
    echo "⏱️  Running CPU benchmarks..."
    cargo bench -- --noplot
    echo "✓ CPU benchmarks complete"
}

function bench_cuda() {
    if check_cuda; then
        echo "⏱️  Running CUDA benchmarks..."
        cargo bench --features cuda -- --noplot
        echo "✓ CUDA benchmarks complete"
    else
        echo "⏭️  Skipping CUDA benchmarks (no CUDA available)"
    fi
}

function build_docker() {
    echo "🐳 Building Docker image (local only)..."
    
    # Check if Dockerfile exists
    if [ ! -f "$PROJECT_ROOT/Dockerfile" ]; then
        echo "⚠️  No Dockerfile found, creating basic one..."
        cat > "$PROJECT_ROOT/Dockerfile" << 'EOF'
# SPDX-License-Identifier: MIT
FROM rust:1.75-slim as builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches
COPY tests ./tests
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/libunsloth_rs.so /usr/local/lib/
COPY --from=builder /build/target/release/libunsloth_rs.a /usr/local/lib/
RUN ldconfig
EOF
    fi
    
    docker build -t unsloth-rs:local .
    echo "✓ Docker image built: unsloth-rs:local"
    echo "  To push to GHCR: docker tag unsloth-rs:local ghcr.io/YOUR_ORG/unsloth-rs:TAG"
}

function prune_builds() {
    echo "🧹 Pruning build artifacts..."
    cargo clean
    if command -v docker &> /dev/null; then
        echo "🐳 Pruning Docker images..."
        docker image prune -f
    fi
    echo "✓ Build artifacts pruned"
}

# Main execution
case "$BUILD_TYPE" in
    "cpu")
        build_cpu
        test_cpu
        ;;
    "cuda")
        build_cuda
        test_cuda
        ;;
    "bench")
        bench_cpu
        bench_cuda
        ;;
    "docker")
        build_docker
        ;;
    "prune")
        prune_builds
        ;;
    "all")
        build_cpu
        build_cuda
        test_cpu
        test_cuda
        echo ""
        echo "✅ All builds complete!"
        echo ""
        echo "Next steps:"
        echo "  - Run benchmarks: ./scripts/local-build.sh bench"
        echo "  - Build Docker: ./scripts/local-build.sh docker"
        echo "  - Clean up: ./scripts/local-build.sh prune"
        ;;
    *)
        echo "Usage: $0 [cpu|cuda|bench|docker|prune|all]"
        echo ""
        echo "Commands:"
        echo "  cpu     - Build and test CPU version"
        echo "  cuda    - Build and test CUDA version"
        echo "  bench   - Run benchmarks (CPU and CUDA if available)"
        echo "  docker  - Build Docker image (local only)"
        echo "  prune   - Clean build artifacts and Docker images"
        echo "  all     - Build and test everything (default)"
        exit 1
        ;;
esac
