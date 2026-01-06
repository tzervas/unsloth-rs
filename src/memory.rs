//! Memory management utilities.

use crate::error::{Result, UnslothError};

/// Memory pool for efficient GPU allocation.
pub struct MemoryPool {
    /// Total allocated bytes
    allocated: usize,
    /// Peak memory usage
    peak: usize,
    /// Memory limit (if set)
    limit: Option<usize>,
}

impl MemoryPool {
    /// Create a new memory pool.
    pub fn new(limit: Option<usize>) -> Self {
        Self {
            allocated: 0,
            peak: 0,
            limit,
        }
    }

    /// Request memory allocation.
    ///
    /// # Errors
    /// Returns OutOfMemory if limit would be exceeded.
    pub fn allocate(&mut self, bytes: usize) -> Result<()> {
        let new_total = self.allocated + bytes;
        
        if let Some(limit) = self.limit {
            if new_total > limit {
                return Err(UnslothError::OutOfMemory {
                    required: new_total,
                    available: limit.saturating_sub(self.allocated),
                });
            }
        }
        
        self.allocated = new_total;
        self.peak = self.peak.max(self.allocated);
        Ok(())
    }

    /// Free memory.
    pub fn free(&mut self, bytes: usize) {
        self.allocated = self.allocated.saturating_sub(bytes);
    }

    /// Get current allocation.
    #[must_use]
    pub fn allocated(&self) -> usize {
        self.allocated
    }

    /// Get peak allocation.
    #[must_use]
    pub fn peak(&self) -> usize {
        self.peak
    }

    /// Reset peak tracking.
    pub fn reset_peak(&mut self) {
        self.peak = self.allocated;
    }
}

/// Gradient checkpointing configuration.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Checkpoint every N layers
    pub checkpoint_every: usize,
    /// Enable checkpointing
    pub enabled: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_every: 1,
            enabled: true,
        }
    }
}

/// Calculate memory requirements for a forward pass.
#[must_use]
pub fn estimate_forward_memory(
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    num_layers: usize,
    checkpoint_config: &CheckpointConfig,
) -> usize {
    let bytes_per_elem = 4; // f32
    
    // Per-layer activation memory
    let activation_per_layer = batch_size * seq_len * hidden_size * bytes_per_elem;
    
    // With checkpointing, only store every N layers
    let stored_layers = if checkpoint_config.enabled {
        (num_layers + checkpoint_config.checkpoint_every - 1) / checkpoint_config.checkpoint_every
    } else {
        num_layers
    };
    
    stored_layers * activation_per_layer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = MemoryPool::new(Some(1000));
        
        assert!(pool.allocate(500).is_ok());
        assert_eq!(pool.allocated(), 500);
        
        assert!(pool.allocate(400).is_ok());
        assert_eq!(pool.allocated(), 900);
        
        // Should fail - would exceed limit
        assert!(pool.allocate(200).is_err());
        
        pool.free(300);
        assert_eq!(pool.allocated(), 600);
    }

    #[test]
    fn test_checkpoint_memory_reduction() {
        let batch = 4;
        let seq = 2048;
        let hidden = 4096;
        let layers = 32;
        
        let no_checkpoint = CheckpointConfig { enabled: false, ..Default::default() };
        let with_checkpoint = CheckpointConfig { enabled: true, checkpoint_every: 4 };
        
        let mem_full = estimate_forward_memory(batch, seq, hidden, layers, &no_checkpoint);
        let mem_checkpoint = estimate_forward_memory(batch, seq, hidden, layers, &with_checkpoint);
        
        // Checkpointing should reduce memory significantly
        assert!(mem_checkpoint < mem_full / 2);
    }
}
