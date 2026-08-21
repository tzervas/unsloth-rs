//! Read fixture tensors and run `unsloth_rs::ops` on CUDA. Dump rust_*.npy-compatible f32.
//! Output dumps use a host copy; the kernels themselves stay on `CudaStorage`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use unsloth_rs::ops::{
    attention, attention_softcap, attention_window, cross_entropy, geglu, layernorm, rmsnorm, rope,
    rope_with_ids, swiglu,
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

const WARMUP: usize = 5;
const N_SAMPLES: usize = 100;
const WINDOW: usize = 64;
const SOFTCAP: f32 = 50.0;

fn timed_stats<T, F: FnMut() -> Result<T, Box<dyn std::error::Error>>>(
    device: &Device,
    mut f: F,
) -> Result<(T, f64, f64), Box<dyn std::error::Error>> {
    // Drop first-touch CUDA compile / allocator. Not a criterion bench.
    for _ in 0..WARMUP {
        let _ = f()?;
    }
    device.synchronize()?;
    let mut samples = Vec::with_capacity(N_SAMPLES);
    let mut last = None;
    for _ in 0..N_SAMPLES {
        let t0 = Instant::now();
        last = Some(f()?);
        device.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = unsloth_rs::kernels::custom_op::sorted_percentile(&samples, 0.50);
    let p99 = unsloth_rs::kernels::custom_op::sorted_percentile(&samples, 0.99);
    Ok((last.expect("N_SAMPLES > 0"), p50, p99))
}

fn ms_pair(p50: f64, p99: f64) -> String {
    format!("{{\"p50\":{p50:.4},\"p99\":{p99:.4},\"n\":{N_SAMPLES}}}")
}

fn discover_tags(work: &Path) -> Vec<String> {
    let mut tags = Vec::new();
    let Ok(rd) = fs::read_dir(work) else {
        return tags;
    };
    for ent in rd.flatten() {
        let p = ent.path();
        if p.join("shape.json").is_file() && p.join("x.f32").is_file() {
            tags.push(ent.file_name().to_string_lossy().into_owned());
        }
    }
    tags.sort_by_key(|t| t.trim_start_matches('s').parse::<usize>().unwrap_or(0));
    tags
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let work = env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("/work/out"), PathBuf::from);
    let device = Device::new_cuda(0).map_err(|e| format!("FAIL_ENV cuda: {e}"))?;
    let mut ms_json = String::new();
    let mut fail_env = Vec::new();
    for tag in discover_tags(&work) {
        let root = work.join(&tag);
        let x = load_f32(&root.join("x.f32"));
        let w = load_f32(&root.join("w.f32"));
        let hidden = w.len();
        let rows = x.len() / hidden;
        let xt = Tensor::from_vec(x, (rows, hidden), &device)?.to_dtype(DType::F32)?;
        let wt = Tensor::from_vec(w, hidden, &device)?;
        let (y, rms_p50, rms_p99) = timed_stats(&device, || Ok(rmsnorm(&xt, &wt, 1e-5)?))?;
        write_f32(
            &root.join("rust_rmsnorm.f32"),
            &y.flatten_all()?.to_vec1::<f32>()?,
        );

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
        let (rq, rope_p50, rope_p99) = timed_stats(&device, || Ok(rope(&qt, &cost, &sint)?))?;
        write_f32(
            &root.join("rust_rope.f32"),
            &rq.flatten_all()?.to_vec1::<f32>()?,
        );

        let gate = load_f32(&root.join("gate.f32"));
        let up = load_f32(&root.join("up.f32"));
        let gt = Tensor::from_vec(gate, (b, s, d), &device)?;
        let ut = Tensor::from_vec(up, (b, s, d), &device)?;
        let (sw, swi_p50, swi_p99) = timed_stats(&device, || Ok(swiglu(&gt, &ut)?))?;
        write_f32(
            &root.join("rust_swiglu.f32"),
            &sw.flatten_all()?.to_vec1::<f32>()?,
        );

        let logits = load_f32(&root.join("logits.f32"));
        let targets = load_i64(&root.join("targets.i64"));
        let ntok = targets.len();
        let vsz = logits.len() / ntok;
        let lt = Tensor::from_vec(logits, (ntok, vsz), &device)?;
        let tt = Tensor::from_vec(targets, ntok, &device)?;
        let (ce, ce_p50, ce_p99) = timed_stats(&device, || Ok(cross_entropy(&lt, &tt)?))?;
        write_f32(&root.join("rust_ce.f32"), &[ce.to_vec0::<f32>()?]);

        let bias = load_f32(&root.join("bias.f32"));
        let bt = Tensor::from_vec(bias, hidden, &device)?;
        let (ln, ln_p50, ln_p99) = timed_stats(&device, || Ok(layernorm(&xt, &wt, &bt, 1e-5)?))?;
        write_f32(
            &root.join("rust_layernorm.f32"),
            &ln.flatten_all()?.to_vec1::<f32>()?,
        );

        let (gg, geg_p50, geg_p99) = timed_stats(&device, || Ok(geglu(&gt, &ut)?))?;
        write_f32(
            &root.join("rust_geglu.f32"),
            &gg.flatten_all()?.to_vec1::<f32>()?,
        );

        let cache = load_f32(&root.join("cos_cache.f32"));
        let cache_s = cache.len() / (d / 2);
        let costc = Tensor::from_vec(cache, (cache_s, d / 2), &device)?;
        let sintc = Tensor::from_vec(
            load_f32(&root.join("sin_cache.f32")),
            (cache_s, d / 2),
            &device,
        )?;
        let pos = load_i64(&root.join("pos_ids.i64"));
        let pt = Tensor::from_vec(pos, s, &device)?;
        let (rid, rid_p50, rid_p99) =
            timed_stats(&device, || Ok(rope_with_ids(&qt, &costc, &sintc, &pt)?))?;
        write_f32(
            &root.join("rust_rope_with_ids.f32"),
            &rid.flatten_all()?.to_vec1::<f32>()?,
        );

        // Owned tiled SRAM FA on CudaStorage (no [B,H,S,S]). Not Unsloth PTX.
        let scale = (d as f64).sqrt().recip();
        let attn_json =
            match timed_stats(&device, || Ok(attention(&qt, &kt, &vt, scale, None, true)?)) {
                Ok((at, p50, p99)) => {
                    write_f32(
                        &root.join("rust_attn.f32"),
                        &at.flatten_all()?.to_vec1::<f32>()?,
                    );
                    let (aw, wp50, wp99) = timed_stats(&device, || {
                        Ok(attention_window(&qt, &kt, &vt, scale, true, WINDOW)?)
                    })?;
                    write_f32(
                        &root.join("rust_attn_window.f32"),
                        &aw.flatten_all()?.to_vec1::<f32>()?,
                    );
                    let (ac, cp50, cp99) = timed_stats(&device, || {
                        Ok(attention_softcap(
                            &qt, &kt, &vt, scale, None, true, SOFTCAP,
                        )?)
                    })?;
                    write_f32(
                        &root.join("rust_attn_softcap.f32"),
                        &ac.flatten_all()?.to_vec1::<f32>()?,
                    );
                    format!(
                        "\"attn\":{},\"attn_window\":{},\"attn_softcap\":{}",
                        ms_pair(p50, p99),
                        ms_pair(wp50, wp99),
                        ms_pair(cp50, cp99),
                    )
                }
                Err(e) => {
                    let msg = format!("FAIL_ENV {tag} attn: {e}");
                    eprintln!("{msg}");
                    fail_env.push(msg);
                    "\"attn\":null,\"attn_window\":null,\"attn_softcap\":null".to_string()
                }
            };

        let chunk = format!(
            "\"{tag}\":{{\"rmsnorm\":{},\"layernorm\":{},\"rope\":{},\"rope_with_ids\":{},\"swiglu\":{},\"geglu\":{},\"ce\":{},{attn_json}}},",
            ms_pair(rms_p50, rms_p99),
            ms_pair(ln_p50, ln_p99),
            ms_pair(rope_p50, rope_p99),
            ms_pair(rid_p50, rid_p99),
            ms_pair(swi_p50, swi_p99),
            ms_pair(geg_p50, geg_p99),
            ms_pair(ce_p50, ce_p99),
        );
        ms_json.push_str(&chunk);
    }
    let fail_json = if fail_env.is_empty() {
        String::new()
    } else {
        format!(
            ",\"fail_env\":[{}]",
            fail_env
                .iter()
                .map(|s| format!("\"{}\"", s.replace('"', "'")))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let json = format!(
        "{{\"device\":\"cuda\",\"warmup\":{WARMUP},\"n_samples\":{N_SAMPLES},\"window\":{WINDOW},\"softcap\":{SOFTCAP},\"cuda_compute_cap\":{},\"attn\":\"unsloth_rs::ops::attention tiled SRAM FA (owned NVRTC, not Unsloth PTX)\"{fail_json},\"ms\":{{{}}}}}\n",
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
    // {"B":2,"H":8,"S":128,"D":64,...}
    let num = |k: &str| -> usize {
        let key = format!("\"{k}\":");
        let i = t.find(&key).unwrap_or_else(|| panic!("missing {k}")) + key.len();
        t[i..]
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap_or_else(|err| panic!("dim {k}: {err}"))
    };
    (num("B"), num("H"), num("S"), num("D"))
}
