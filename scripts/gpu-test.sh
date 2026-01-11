#!/bin/bash
#
# GPU Test Runner for unsloth-rs
#
# This script provides a convenient interface for running GPU-accelerated tests
# for the unsloth-rs library, specifically targeting Flash Attention and other
# GPU-optimized kernels.
#
# REQUIREMENTS:
# - CUDA 12.0 or later
# - Compatible GPU (compute capability 7.0+)
# - At least 4GB VRAM for basic tests
# - Rust toolchain with CUDA feature support
#
# USAGE:
#   ./scripts/gpu-test.sh [COMMAND] [TARGET] [OPTIONS]
#
# COMMANDS:
#   test          Run GPU tests
#   benchmark     Run GPU benchmarks  
#   profile       Profile GPU kernels
#   validate      Validate GPU setup
#   clean         Clean GPU build artifacts
#
# TARGETS:
#   all              All GPU tests
#   flash_attention  Flash Attention tests only
#   ternary          Ternary quantization GPU tests
#   attention        General attention tests
#
# OPTIONS:
#   --verbose        Enable verbose output
#   --release        Use release build for performance tests
#   --iterations N   Number of benchmark iterations (default: 10)
#   --min-vram GB    Minimum VRAM requirement (default: 4)
#
# EXAMPLES:
#   ./scripts/gpu-test.sh validate
#   ./scripts/gpu-test.sh test flash_attention
#   ./scripts/gpu-test.sh benchmark all --release --iterations 20
#   ./scripts/gpu-test.sh test flash_attention --min-vram 8

set -euo pipefail

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ITERATIONS=10
DEFAULT_MIN_VRAM=4

# Parse arguments
COMMAND="${1:-test}"

# For validate command, don't expect a target
if [[ "$COMMAND" == "validate" ]]; then
    TARGET=""
    shift 1 2>/dev/null || true
else
    TARGET="${2:-flash_attention}"
    shift 2 2>/dev/null || true
fi

VERBOSE=false
RELEASE_FLAG=""
ITERATIONS=$DEFAULT_ITERATIONS  
MIN_VRAM_GB=$DEFAULT_MIN_VRAM

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --release)
            RELEASE_FLAG="--release"
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --min-vram)
            MIN_VRAM_GB="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check if CUDA is available
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. CUDA drivers may not be installed."
        return 1
    fi

    if ! nvidia-smi > /dev/null 2>&1; then
        log_error "nvidia-smi failed. GPU may not be available."
        return 1
    fi

    return 0
}

# Get GPU information
get_gpu_info() {
    if ! check_cuda; then
        return 1
    fi

    log_info "GPU Information:"
    nvidia-smi --query-gpu=gpu_name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits | while IFS=', ' read -r name total_mem free_mem compute_cap; do
        echo "  GPU: $name"
        echo "  Total VRAM: ${total_mem}MB ($(echo "scale=1; $total_mem/1024" | bc)GB)"
        echo "  Free VRAM: ${free_mem}MB ($(echo "scale=1; $free_mem/1024" | bc)GB)"
        echo "  Compute Capability: $compute_cap"
        
        # Check minimum VRAM requirement
        total_gb=$(echo "scale=1; $total_mem/1024" | bc)
        if (( $(echo "$total_gb < $MIN_VRAM_GB" | bc -l) )); then
            log_warning "GPU has ${total_gb}GB VRAM, minimum ${MIN_VRAM_GB}GB recommended"
        fi
    done
}

# Validate GPU setup
validate_gpu_setup() {
    log_info "Validating GPU setup..."
    
    if ! check_cuda; then
        log_error "CUDA validation failed"
        return 1
    fi

    get_gpu_info
    
    # Test basic CUDA compilation
    log_info "Testing CUDA feature compilation..."
    cd "$PROJECT_ROOT"
    
    if cargo check --features cuda > /dev/null 2>&1; then
        log_success "CUDA feature compilation: OK"
    else
        log_error "CUDA feature compilation failed"
        log_error "Try: cargo check --features cuda"
        return 1
    fi
    
    # Test basic GPU tensor operations
    log_info "Testing basic GPU operations..."
    if cargo test --features cuda test_gpu_basic_ops --lib > /dev/null 2>&1; then
        log_success "Basic GPU operations: OK"
    else
        log_warning "Basic GPU operations test not found or failed"
    fi

    log_success "GPU setup validation complete"
    return 0
}

# Run specific GPU tests
run_gpu_tests() {
    local target="$1"
    cd "$PROJECT_ROOT"
    
    log_info "Running GPU tests for target: $target"
    
    # Build test command
    local test_cmd="cargo test --features cuda"
    if [[ -n "$RELEASE_FLAG" ]]; then
        test_cmd="$test_cmd $RELEASE_FLAG"
    fi
    
    # Add test filters based on target
    case "$target" in
        "flash_attention")
            test_cmd="$test_cmd test_flash_attention"
            ;;
        "attention")
            test_cmd="$test_cmd attention"
            ;;
        "ternary")
            test_cmd="$test_cmd ternary"
            ;;
        "all")
            test_cmd="$test_cmd --test integration"
            ;;
        *)
            log_error "Unknown test target: $target"
            log_info "Available targets: flash_attention, attention, ternary, all"
            return 1
            ;;
    esac
    
    # Add verbose flag if requested
    if [[ "$VERBOSE" == "true" ]]; then
        test_cmd="$test_cmd -- --nocapture"
    fi
    
    log_info "Executing: $test_cmd"
    
    # Run tests with timeout
    if timeout 300 bash -c "$test_cmd"; then
        log_success "GPU tests completed successfully"
    else
        local exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            log_error "GPU tests timed out (5 minutes)"
        else
            log_error "GPU tests failed (exit code: $exit_code)"
        fi
        return 1
    fi
}

# Run GPU benchmarks
run_gpu_benchmarks() {
    local target="$1"
    cd "$PROJECT_ROOT"
    
    log_info "Running GPU benchmarks for target: $target (iterations: $ITERATIONS)"
    
    # Build benchmark command
    local bench_cmd="cargo bench --features cuda"
    if [[ -n "$RELEASE_FLAG" ]]; then
        bench_cmd="$bench_cmd $RELEASE_FLAG"
    fi
    
    # Set benchmark iterations via environment variable
    export CRITERION_SAMPLE_SIZE="$ITERATIONS"
    
    case "$target" in
        "flash_attention")
            bench_cmd="$bench_cmd -- attention"
            ;;
        "all")
            bench_cmd="$bench_cmd"
            ;;
        *)
            log_error "Unknown benchmark target: $target"
            log_info "Available targets: flash_attention, all"
            return 1
            ;;
    esac
    
    log_info "Executing: $bench_cmd"
    
    if eval "$bench_cmd"; then
        log_success "GPU benchmarks completed"
        log_info "Results saved to target/criterion/"
    else
        log_error "GPU benchmarks failed"
        return 1
    fi
}

# Profile GPU kernels
profile_gpu_kernels() {
    local target="$1"
    
    if ! command -v nsys &> /dev/null; then
        log_error "NVIDIA Nsight Systems (nsys) not found"
        log_info "Install with: sudo apt install nvidia-nsight-systems-cli"
        return 1
    fi
    
    cd "$PROJECT_ROOT"
    
    log_info "Profiling GPU kernels for target: $target"
    
    # Create profile output directory
    mkdir -p target/profiles
    
    local profile_cmd="nsys profile -o target/profiles/gpu_profile_${target}_$(date +%Y%m%d_%H%M%S).nsys-rep"
    profile_cmd="$profile_cmd cargo test --features cuda $RELEASE_FLAG"
    
    case "$target" in
        "flash_attention")
            profile_cmd="$profile_cmd test_flash_attention_gpu_performance"
            ;;
        *)
            log_error "Profiling not yet implemented for target: $target"
            return 1
            ;;
    esac
    
    log_info "Executing: $profile_cmd"
    
    if eval "$profile_cmd"; then
        log_success "GPU profiling completed"
        log_info "Profile saved to target/profiles/"
    else
        log_error "GPU profiling failed"
        return 1
    fi
}

# Clean GPU build artifacts  
clean_gpu_artifacts() {
    cd "$PROJECT_ROOT"
    
    log_info "Cleaning GPU build artifacts..."
    
    cargo clean
    rm -rf target/criterion/
    rm -rf target/profiles/
    
    log_success "GPU artifacts cleaned"
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [COMMAND] [TARGET] [OPTIONS]

COMMANDS:
    test         Run GPU tests
    benchmark    Run GPU benchmarks
    profile      Profile GPU kernels
    validate     Validate GPU setup
    clean        Clean GPU build artifacts
    help         Show this help message

TARGETS:
    all              All GPU tests
    flash_attention  Flash Attention tests only  
    ternary          Ternary quantization GPU tests
    attention        General attention tests

OPTIONS:
    --verbose        Enable verbose output
    --release        Use release build for performance tests
    --iterations N   Number of benchmark iterations (default: $DEFAULT_ITERATIONS)
    --min-vram GB    Minimum VRAM requirement (default: $DEFAULT_MIN_VRAM)

EXAMPLES:
    $0 validate
    $0 test flash_attention
    $0 benchmark all --release --iterations 20
    $0 test flash_attention --min-vram 8 --verbose

EOF
}

# Main execution
main() {
    case "$COMMAND" in
        "test")
            validate_gpu_setup || exit 1
            run_gpu_tests "$TARGET"
            ;;
        "benchmark")
            validate_gpu_setup || exit 1
            run_gpu_benchmarks "$TARGET"
            ;;
        "profile")
            validate_gpu_setup || exit 1
            profile_gpu_kernels "$TARGET"
            ;;
        "validate")
            validate_gpu_setup
            ;;
        "clean")
            clean_gpu_artifacts
            ;;
        "help"|"--help"|"-h")
            print_usage
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            print_usage
            exit 1
            ;;
    esac
}

# Check dependencies
if ! command -v bc &> /dev/null; then
    log_error "bc (calculator) not found. Install with: sudo apt install bc"
    exit 1
fi

# Execute main function
main "$@"