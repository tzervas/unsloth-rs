#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright 2026 Tyler Zervas

set -euo pipefail

# GPU profiling script for Flash Attention and other kernels
# Runs comprehensive benchmarks and collects performance metrics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "⚡ GPU Profiling for unsloth-rs"
echo "================================"
echo ""

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. CUDA required for GPU profiling."
    exit 1
fi

# Get GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader)
echo "🎮 GPU Detected: $GPU_INFO"
echo ""

# Parse command
PROFILE_TARGET="${1:-flash-attention}"
OUTPUT_DIR="$PROJECT_ROOT/target/profiling"
mkdir -p "$OUTPUT_DIR"

function profile_flash_attention() {
    echo "📊 Profiling Flash Attention..."
    echo "================================"
    echo ""
    
    # Sequence lengths to test
    SEQ_LENS=(128 256 512 1024 2048)
    BATCH_SIZES=(1 4 8 16)
    
    RESULTS_FILE="$OUTPUT_DIR/flash_attention_results.txt"
    echo "# Flash Attention Profiling Results" > "$RESULTS_FILE"
    echo "# GPU: $GPU_INFO" >> "$RESULTS_FILE"
    echo "# Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    echo "Running benchmarks for various configurations..."
    echo ""
    
    # Run benchmarks with CUDA feature
    cargo bench --features cuda --bench kernels -- flash_attention 2>&1 | tee -a "$RESULTS_FILE"
    
    echo ""
    echo "✅ Flash Attention profiling complete"
    echo "📄 Results saved to: $RESULTS_FILE"
}

function profile_ternary_kernels() {
    echo "📊 Profiling Ternary Kernels..."
    echo "================================"
    echo ""
    
    RESULTS_FILE="$OUTPUT_DIR/ternary_results.txt"
    echo "# Ternary Kernel Profiling Results" > "$RESULTS_FILE"
    echo "# GPU: $GPU_INFO" >> "$RESULTS_FILE"
    echo "# Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    cargo bench --features cuda --bench kernels -- ternary 2>&1 | tee -a "$RESULTS_FILE"
    
    echo ""
    echo "✅ Ternary kernel profiling complete"
    echo "📄 Results saved to: $RESULTS_FILE"
}

function profile_memory_usage() {
    echo "💾 Profiling Memory Usage..."
    echo "================================"
    echo ""
    
    RESULTS_FILE="$OUTPUT_DIR/memory_profile.txt"
    echo "# Memory Usage Profiling Results" > "$RESULTS_FILE"
    echo "# GPU: $GPU_INFO" >> "$RESULTS_FILE"
    echo "# Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # Use nvidia-smi to monitor memory during benchmark
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv -l 1 > "$OUTPUT_DIR/memory_trace.csv" &
    NVIDIA_SMI_PID=$!
    
    # Run memory-intensive benchmarks
    cargo bench --features cuda --bench kernels 2>&1 | tee -a "$RESULTS_FILE"
    
    # Stop nvidia-smi monitoring
    kill $NVIDIA_SMI_PID 2>/dev/null || true
    
    echo ""
    echo "✅ Memory profiling complete"
    echo "📄 Results saved to: $RESULTS_FILE"
    echo "📄 Memory trace: $OUTPUT_DIR/memory_trace.csv"
}

function profile_with_nsys() {
    echo "🔍 Profiling with NVIDIA Nsight Systems..."
    echo "================================"
    echo ""
    
    if ! command -v nsys &> /dev/null; then
        echo "⚠️  Warning: nsys not found. Install NVIDIA Nsight Systems for detailed profiling."
        echo "   Download from: https://developer.nvidia.com/nsight-systems"
        return 1
    fi
    
    NSYS_OUTPUT="$OUTPUT_DIR/nsys_profile"
    
    # Profile a single benchmark run
    nsys profile \
        --output="$NSYS_OUTPUT" \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        cargo bench --features cuda --bench kernels -- --bench --quick
    
    echo ""
    echo "✅ Nsight Systems profiling complete"
    echo "📄 Profile saved to: ${NSYS_OUTPUT}.nsys-rep"
    echo "   Open with: nsys-ui ${NSYS_OUTPUT}.nsys-rep"
}

function compare_cpu_vs_gpu() {
    echo "⚖️  Comparing CPU vs GPU Performance..."
    echo "================================"
    echo ""
    
    RESULTS_FILE="$OUTPUT_DIR/cpu_vs_gpu.txt"
    echo "# CPU vs GPU Comparison" > "$RESULTS_FILE"
    echo "# Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    echo "## CPU Benchmarks" >> "$RESULTS_FILE"
    echo "Running CPU benchmarks..."
    cargo bench --bench kernels 2>&1 | tee -a "$RESULTS_FILE"
    
    echo "" >> "$RESULTS_FILE"
    echo "## GPU Benchmarks" >> "$RESULTS_FILE"
    echo "Running GPU benchmarks..."
    cargo bench --features cuda --bench kernels 2>&1 | tee -a "$RESULTS_FILE"
    
    echo ""
    echo "✅ CPU vs GPU comparison complete"
    echo "📄 Results saved to: $RESULTS_FILE"
}

function generate_summary() {
    echo "📋 Generating Performance Summary..."
    echo "================================"
    echo ""
    
    SUMMARY_FILE="$OUTPUT_DIR/summary.md"
    
    cat > "$SUMMARY_FILE" << EOF
# GPU Profiling Summary

**Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")  
**GPU**: $GPU_INFO  
**CUDA Version**: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

## Results

### Flash Attention
See: \`flash_attention_results.txt\`

### Ternary Kernels
See: \`ternary_results.txt\`

### Memory Usage
See: \`memory_profile.txt\` and \`memory_trace.csv\`

### CPU vs GPU Comparison
See: \`cpu_vs_gpu.txt\`

## Analysis

(Add analysis here after reviewing results)

## Next Steps

- [ ] Update BENCHMARKING.md with validated numbers
- [ ] Update README.md performance claims
- [ ] Document any optimization opportunities
- [ ] Create GitHub issue for any performance gaps

EOF
    
    echo "✅ Summary generated: $SUMMARY_FILE"
    echo ""
    echo "📁 All results in: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Review results in $OUTPUT_DIR"
    echo "  2. Update BENCHMARKING.md with findings"
    echo "  3. Update documentation with validated numbers"
}

# Main execution
case "$PROFILE_TARGET" in
    "flash-attention"|"flash")
        profile_flash_attention
        ;;
    "ternary")
        profile_ternary_kernels
        ;;
    "memory")
        profile_memory_usage
        ;;
    "nsys")
        profile_with_nsys
        ;;
    "compare")
        compare_cpu_vs_gpu
        ;;
    "all")
        profile_flash_attention
        echo ""
        profile_ternary_kernels
        echo ""
        profile_memory_usage
        echo ""
        compare_cpu_vs_gpu
        echo ""
        generate_summary
        ;;
    *)
        echo "Usage: $0 [flash-attention|ternary|memory|nsys|compare|all]"
        echo ""
        echo "Commands:"
        echo "  flash-attention  - Profile Flash Attention kernel (default)"
        echo "  ternary          - Profile ternary quantization kernels"
        echo "  memory           - Profile memory usage during execution"
        echo "  nsys             - Profile with NVIDIA Nsight Systems (requires nsys)"
        echo "  compare          - Compare CPU vs GPU performance"
        echo "  all              - Run all profiling tasks"
        echo ""
        echo "Examples:"
        echo "  ./scripts/profile-gpu.sh flash-attention"
        echo "  ./scripts/profile-gpu.sh all"
        exit 1
        ;;
esac
