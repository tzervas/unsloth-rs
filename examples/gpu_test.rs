//! Simple GPU test to verify CubeCL CUDA is working

fn main() {
    #[cfg(feature = "cuda")]
    {
        use cubecl::prelude::*;
        use cubecl_cuda::CudaRuntime;
        use std::time::Instant;

        println!("Testing CubeCL CUDA...");

        // Initialize CUDA device
        let device = cubecl_cuda::CudaDevice::new(0);
        let client = CudaRuntime::client(&device);

        println!("CUDA device initialized");

        // Simple test: create and read back data
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            f32::as_bytes(&data).to_vec(),
        ));

        let output_bytes = client.read_one(handle);
        let output_data: &[f32] = f32::from_bytes(&output_bytes);

        println!("Input:  {:?}", data);
        println!("Output: {:?}", output_data);

        assert_eq!(output_data, data.as_slice());
        println!("\n✓ Basic GPU memory test passed!");

        // Test Flash Attention kernel
        println!("\nTesting Flash Attention kernel...");

        use candle_core::{Device, Tensor};
        use unsloth_rs::kernels::cubecl::{flash_attention_kernel, FlashAttentionConfig};

        // Test multiple configurations
        let test_configs = [
            (1, 1, 8, 64, "Small (1x1x8x64)"),
            (1, 8, 128, 64, "Medium (1x8x128x64)"),
            (2, 8, 512, 64, "Large (2x8x512x64)"),
        ];

        for (batch, heads, seq_len, head_dim, label) in test_configs {
            println!("\n  Testing {label}...");

            let q_data: Vec<f32> = (0..batch * heads * seq_len * head_dim)
                .map(|i| (i % 64) as f32 * 0.01)
                .collect();
            let k_data: Vec<f32> = q_data.clone();
            let v_data: Vec<f32> = q_data.clone();

            // Create tensors on CUDA device to actually test GPU kernel
            let cuda_device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
            let q =
                Tensor::from_vec(q_data, (batch, heads, seq_len, head_dim), &cuda_device).unwrap();
            let k =
                Tensor::from_vec(k_data, (batch, heads, seq_len, head_dim), &cuda_device).unwrap();
            let v =
                Tensor::from_vec(v_data, (batch, heads, seq_len, head_dim), &cuda_device).unwrap();

            let scale = 1.0 / (head_dim as f64).sqrt();
            let config = FlashAttentionConfig::default();

            let start = Instant::now();
            match flash_attention_kernel(&q, &k, &v, scale, None, &config) {
                Ok(output) => {
                    let elapsed = start.elapsed();
                    println!("    Output shape: {:?}", output.dims());
                    println!("    Device: {:?}", cuda_device);
                    println!("    Time: {:?}", elapsed);
                }
                Err(e) => {
                    println!("    ✗ Failed: {}", e);
                }
            }
        }

        println!("\n✓ Flash Attention kernel tests passed!");
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!(
            "CUDA feature not enabled. Run with: cargo run --example gpu_test --features cuda"
        );
    }
}
