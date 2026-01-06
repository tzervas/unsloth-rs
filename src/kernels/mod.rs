//! Optimized GPU kernels.

pub mod attention;
pub mod attention_cubecl;
pub mod rope;
pub mod rmsnorm;
pub mod swiglu;

pub use attention::{FusedAttention, FusedAttentionConfig};
pub use attention_cubecl::{flash_attention_cubecl, has_cubecl_support};
pub use rope::RotaryEmbedding;
pub use rmsnorm::RmsNorm;
pub use swiglu::SwiGLU;
