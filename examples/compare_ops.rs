//! Read fixture tensors and run CustomOp CUDA. Dump rust_*.npy-compatible f32.
//! Output dumps use a host copy; the kernels themselves stay on CudaStorage.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use unsloth_rs::kernels::custom_op::{
    attention_device, chunked_cross_entropy, rmsnorm_custom_op, rope_custom_op, swiglu_custom_op,
};

fn load_f32(path: &Path) -> Vec<f32> {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_i64(path: &Path) -> Vec<i64> {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write_f32(path: &Path, data: &[f32]) {
    let mut buf = Vec::with_capacity(data.len() * 4);
    for v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, buf).expect("write f32");
}

fn timed_ms<T, F: FnMut() -> Result<T, Box<dyn std::error::Error>>>(
    mut f: F,
) -> Result<(T, f64), Box<dyn std::error::Error>> {
    // Drop first-touch CUDA compile / allocator. Not a criterion bench.
    for _ in 0..3 {
        let _ = f()?;
    }
    let t0 = Instant::now();
    let out = f()?;
    Ok((out, t0.elapsed().as_secs_f64() * 1e3))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let work = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/work/out"));
    let device = Device::new_cuda(0).map_err(|e| format!("FAIL_ENV cuda: {e}"))?;
    let mut ms_json = String::new();
    for tag in ["s128", "s512"] {
        let root = work.join(tag);
        if !root.join("x.f32").is_file() {
            continue;
        }
        let x = load_f32(&root.join("x.f32"));
        let w = load_f32(&root.join("w.f32"));
        let hidden = w.len();
        let rows = x.len() / hidden;
        let xt = Tensor::from_vec(x, (rows, hidden), &device)?.to_dtype(DType::F32)?;
        let wt = Tensor::from_vec(w, hidden, &device)?;
        let (y, rms_ms) = timed_ms(|| Ok(rmsnorm_custom_op(&xt, &wt, 1e-5)?))?;
        write_f32(&root.join("rust_rmsnorm.f32"), &y.flatten_all()?.to_vec1::<f32>()?);

        let q = load_f32(&root.join("q.f32"));
        let k = load_f32(&root.join("k.f32"));
        let v = load_f32(&root.join("v.f32"));
        let meta = load_shape_meta(&root);
        let (b, h, s, d) = meta;
        let qt = Tensor::from_vec(q, (b, h, s, d), &device)?;
        let kt = Tensor::from_vec(k, (b, h, s, d), &device)?;
        let vt = Tensor::from_vec(v, (b, h, s, d), &device)?;
        let cos = load_f32(&root.join("cos.f32"));
        let sin = load_f32(&root.join("sin.f32"));
        let cost = Tensor::from_vec(cos, (s, d / 2), &device)?;
        let sint = Tensor::from_vec(sin, (s, d / 2), &device)?;
        let (rq, rope_ms) = timed_ms(|| Ok(rope_custom_op(&qt, &cost, &sint)?))?;
        write_f32(&root.join("rust_rope.f32"), &rq.flatten_all()?.to_vec1::<f32>()?);

        let gate = load_f32(&root.join("gate.f32"));
        let up = load_f32(&root.join("up.f32"));
        let gt = Tensor::from_vec(gate, (b, s, d), &device)?;
        let ut = Tensor::from_vec(up, (b, s, d), &device)?;
        let (sw, swi_ms) = timed_ms(|| Ok(swiglu_custom_op(&gt, &ut)?))?;
        write_f32(&root.join("rust_swiglu.f32"), &sw.flatten_all()?.to_vec1::<f32>()?);

        let logits = load_f32(&root.join("logits.f32"));
        let targets = load_i64(&root.join("targets.i64"));
        let ntok = targets.len();
        let vsz = logits.len() / ntok;
        let lt = Tensor::from_vec(logits, (ntok, vsz), &device)?;
        let tt = Tensor::from_vec(targets, ntok, &device)?;
        let (ce, ce_ms) = timed_ms(|| Ok(chunked_cross_entropy(&lt, &tt, -100, 4096)?))?;
        write_f32(&root.join("rust_ce.f32"), &[ce.to_vec0::<f32>()?]);

        // CUDA CustomOp cuda_fwd is unimplemented; attention_device is GEMM+softmax
        // on CudaStorage (materializes [B,H,S,S], no D2H). Not tiled FA.
        let scale = (d as f64).sqrt().recip();
        let (at, attn_ms) = timed_ms(|| Ok(attention_device(&qt, &kt, &vt, scale, None, true)?))?;
        write_f32(&root.join("rust_attn.f32"), &at.flatten_all()?.to_vec1::<f32>()?);

        ms_json.push_str(&format!(
            "\"{tag}\":{{\"rmsnorm\":{rms_ms:.4},\"rope\":{rope_ms:.4},\"swiglu\":{swi_ms:.4},\"ce\":{ce_ms:.4},\"attn\":{attn_ms:.4}}},"
        ));
    }
    let json = format!(
        "{{\"device\":\"cuda\",\"warmup\":3,\"cuda_compute_cap\":{},\"attn\":\"attention_device GEMM+softmax on CudaStorage, not tiled FA\",\"ms\":{{{}}}}}\n",
        env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "unset".into()),
        ms_json.trim_end_matches(',')
    );
    fs::write(work.join("rust_meta.json"), json)?;
    println!("wrote rust outputs under {}", work.display());
    Ok(())
}

fn load_shape_meta(root: &Path) -> (usize, usize, usize, usize) {
    let p = root.join("shape.json");
    let t = fs::read_to_string(p).expect("shape.json");
    // {"B":2,"H":8,"S":128,"D":64}
    let num = |k: &str| -> usize {
        let key = format!("\"{k}\":");
        let i = t.find(&key).unwrap_or_else(|| panic!("missing {k}")) + key.len();
        t[i..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap_or_else(|e| panic!("dim {k}: {e}"))
    };
    (num("B"), num("H"), num("S"), num("D"))
}
