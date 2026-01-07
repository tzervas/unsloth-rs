//! Optimized GPU kernels.

pub mod attention;
pub mod attention_cubecl;
pub mod cubecl;
pub mod rmsnorm;
pub mod rope;
pub mod swiglu;
pub mod ternary;

pub use attention::{FusedAttention, FusedAttentionConfig};
pub use attention_cubecl::{flash_attention_cubecl, has_cubecl_support};
pub use cubecl::{flash_attention_kernel, FlashAttentionConfig};
pub use rmsnorm::RmsNorm;
pub use rope::RotaryEmbedding;
pub use swiglu::SwiGLU;

// Ternary bitsliced operations
pub use ternary::{
    CalibrationMethod, SparsityMetadata, TernaryConfig, TernaryLinear, TernaryPlanes,
    TernaryTensor,
};
