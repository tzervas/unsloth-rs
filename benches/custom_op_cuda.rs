//! Host vs CUDA-event p50/p99 for `CustomOp` kernels (G-UNS-01 slice 1).
//!
//! First NVRTC compile and `get_or_load_custom_func` stay outside every timed
//! region. Criterion output is informational; the recorded artifact is
//! `artifacts/custom_op_cuda.json` (sorted-sample p50/p99).
//!
//! Without `--features cuda` this binary prints `FAIL_ENV` and exits 2.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("FAIL_ENV: missing feature cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(code) = cuda_main() {
        std::process::exit(code);
    }
}

#[cfg(feature = "cuda")]
fn cuda_main() -> Result<(), i32> {
    use std::path::Path;
    use std::time::{Duration, Instant};

    use candle_core::cuda::cudarc::driver::sys::CUevent_flags;
    use candle_core::Device;
    use criterion::{BenchmarkId, Criterion};

    const N_SAMPLES: usize = 100;
    const WARMUP: usize = 5;

    if !Path::new("/dev/nvidia0").exists() {
        eprintln!("FAIL_ENV: missing /dev/nvidia0");
        return Err(2);
    }
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL_ENV: Device::new_cuda(0): {e}");
            return Err(2);
        }
    };
    let stream = match device.as_cuda_device() {
        Ok(d) => d.cuda_stream(),
        Err(e) => {
            eprintln!("FAIL_ENV: as_cuda_device: {e}");
            return Err(2);
        }
    };

    let gpu = nvsmi_query(&["--query-gpu=name", "--format=csv,noheader"])
        .trim()
        .to_string();
    let gpu_compute_cap = nvsmi_query(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .trim()
        .to_string();
    let cuda_compute_cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "unset".into());
    let compute_apps = nvsmi_query(&[
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv",
    ]);
    if let Some(line) = blocked_compute_app(&compute_apps) {
        eprintln!("FAIL_ENV: compute-apps lists llama-server or hypha-control: {line}");
        return Err(2);
    }

    let compile_cached = match measure_compile_cached() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL: cache probe: {e}");
            return Err(1);
        }
    };

    let cases = match build_cases(&device) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL: tensor setup: {e}");
            return Err(1);
        }
    };

    let mut rows = Vec::with_capacity(cases.len());
    for case in &cases {
        if let Err(e) = warmup(&case.work, WARMUP) {
            eprintln!("FAIL: warmup {}: {e}", case.id);
            return Err(1);
        }
        if let Err(e) = stream.synchronize() {
            eprintln!("FAIL: warmup sync: {e}");
            return Err(1);
        }

        let mut host = Vec::with_capacity(N_SAMPLES);
        let mut event = Vec::with_capacity(N_SAMPLES);
        for _ in 0..N_SAMPLES {
            let t0 = Instant::now();
            match case.work.run() {
                Ok(y) => {
                    if let Err(e) = stream.synchronize() {
                        eprintln!("FAIL: host sync {}: {e}", case.id);
                        return Err(1);
                    }
                    host.push(t0.elapsed().as_secs_f64() * 1.0e3);
                    drop(y);
                }
                Err(e) => {
                    eprintln!("FAIL: host op {}: {e}", case.id);
                    return Err(1);
                }
            }

            let start = match stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT)) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("FAIL: record start {}: {e}", case.id);
                    return Err(1);
                }
            };
            match case.work.run() {
                Ok(y) => {
                    let end = match stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT)) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("FAIL: record end {}: {e}", case.id);
                            return Err(1);
                        }
                    };
                    match start.elapsed_ms(&end) {
                        Ok(ms) => event.push(f64::from(ms)),
                        Err(e) => {
                            eprintln!("FAIL: elapsed_ms {}: {e}", case.id);
                            return Err(1);
                        }
                    }
                    drop(y);
                }
                Err(e) => {
                    eprintln!("FAIL: event op {}: {e}", case.id);
                    return Err(1);
                }
            }
        }

        host.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        event.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let row = MeasuredRow {
            op: case.op,
            tag: case.tag,
            elems: case.elems,
            vocab: case.vocab,
            n: case.n,
            launch_bound: case.launch_bound,
            compile_cached,
            host_p50_ms: unsloth_rs::kernels::custom_op::sorted_percentile(&host, 0.50),
            host_p99_ms: unsloth_rs::kernels::custom_op::sorted_percentile(&host, 0.99),
            event_p50_ms: unsloth_rs::kernels::custom_op::sorted_percentile(&event, 0.50),
            event_p99_ms: unsloth_rs::kernels::custom_op::sorted_percentile(&event, 0.99),
        };
        println!(
            "{} host_p50={:.4} host_p99={:.4} event_p50={:.4} event_p99={:.4} elems={} launch_bound={}",
            row.id(),
            row.host_p50_ms,
            row.host_p99_ms,
            row.event_p50_ms,
            row.event_p99_ms,
            row.elems,
            row.launch_bound
        );
        rows.push(row);
    }

    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/custom_op_cuda.json");
    if let Err(e) = write_json(
        &out,
        &gpu,
        &cuda_compute_cap,
        &gpu_compute_cap,
        &compute_apps,
        &rows,
    ) {
        eprintln!("FAIL_IO: write {}: {e}", out.display());
        return Err(1);
    }
    println!("wrote {}", out.display());

    // Criterion is informational only; JSON above is the recorded artifact.
    let mut c = Criterion::default()
        .configure_from_args()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    {
        let mut group = c.benchmark_group("custom_op_cuda_event");
        for case in &cases {
            let work = case.work.clone();
            let stream = stream.clone();
            group.bench_function(BenchmarkId::from_parameter(&case.id), move |b| {
                b.iter_custom(|iters| {
                    let start = stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .expect("record start");
                    for _ in 0..iters {
                        let y = work.run().expect("op");
                        std::hint::black_box(y);
                    }
                    let end = stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .expect("record end");
                    let ms = start.elapsed_ms(&end).expect("elapsed_ms");
                    Duration::from_secs_f64(f64::from(ms) / 1.0e3)
                });
            });
        }
        group.finish();
    }
    c.final_summary();
    Ok(())
}

#[cfg(feature = "cuda")]
fn nvsmi_query(args: &[&str]) -> String {
    std::process::Command::new("nvidia-smi")
        .args(args)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
}

#[cfg(feature = "cuda")]
fn blocked_compute_app(csv: &str) -> Option<String> {
    for line in csv.lines().skip(1) {
        let lower = line.to_ascii_lowercase();
        if lower.contains("llama-server") || lower.contains("hypha-control") {
            return Some(line.trim().to_string());
        }
    }
    None
}

/// First-vs-second NVRTC compile on a unique module. True only if the second
/// call did not increment the compile counter and was wall-clock faster.
#[cfg(feature = "cuda")]
fn measure_compile_cached() -> Result<bool, String> {
    use std::time::Instant;
    use unsloth_rs::kernels::custom_op::nvrtc::{cached_ptx, ptx_compile_count};

    let src = r#"extern "C" __global__ void g_uns01_cache_probe(float *x) { *x = 0.f; }"#;
    let name = "g_uns01_cache_probe";
    let c0 = ptx_compile_count();
    let t0 = Instant::now();
    cached_ptx(name, src).map_err(|e| format!("first compile: {e}"))?;
    let first = t0.elapsed();
    let c1 = ptx_compile_count();
    let t1 = Instant::now();
    cached_ptx(name, src).map_err(|e| format!("second compile: {e}"))?;
    let second = t1.elapsed();
    let c2 = ptx_compile_count();
    Ok(c1 == c0 + 1 && c2 == c1 && second < first)
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct Work {
    kind: WorkKind,
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
enum WorkKind {
    Rms {
        x: candle_core::Tensor,
        w: candle_core::Tensor,
        eps: f32,
    },
    Rope {
        x: candle_core::Tensor,
        cos: candle_core::Tensor,
        sin: candle_core::Tensor,
    },
    Swi {
        gate: candle_core::Tensor,
        up: candle_core::Tensor,
    },
    Ce {
        logits: candle_core::Tensor,
        targets: candle_core::Tensor,
    },
    Attn {
        q: candle_core::Tensor,
        k: candle_core::Tensor,
        v: candle_core::Tensor,
        scale: f32,
        causal: bool,
    },
    FusedCe {
        hidden: candle_core::Tensor,
        weight: candle_core::Tensor,
        targets: candle_core::Tensor,
        chunk: usize,
    },
}

#[cfg(feature = "cuda")]
impl Work {
    fn run(&self) -> candle_core::Result<candle_core::Tensor> {
        use unsloth_rs::kernels::custom_op::{
            attention_custom_op, chunked_cross_entropy, rmsnorm_custom_op, rope_custom_op,
            swiglu_custom_op,
        };
        match &self.kind {
            WorkKind::Rms { x, w, eps } => rmsnorm_custom_op(x, w, *eps),
            WorkKind::Rope { x, cos, sin } => rope_custom_op(x, cos, sin),
            WorkKind::Swi { gate, up } => swiglu_custom_op(gate, up),
            WorkKind::Ce { logits, targets } => chunked_cross_entropy(logits, targets, -100, 4096),
            WorkKind::Attn {
                q,
                k,
                v,
                scale,
                causal,
            } => attention_custom_op(q, k, v, *scale, *causal),
            WorkKind::FusedCe {
                hidden,
                weight,
                targets,
                chunk,
            } => unsloth_rs::ops::fused_linear_ce(hidden, weight, targets, *chunk),
        }
    }
}

#[cfg(feature = "cuda")]
struct Case {
    id: String,
    op: &'static str,
    tag: &'static str,
    elems: usize,
    vocab: Option<usize>,
    n: Option<usize>,
    launch_bound: bool,
    work: Work,
}

#[cfg(feature = "cuda")]
struct MeasuredRow {
    op: &'static str,
    tag: &'static str,
    elems: usize,
    vocab: Option<usize>,
    n: Option<usize>,
    launch_bound: bool,
    compile_cached: bool,
    host_p50_ms: f64,
    host_p99_ms: f64,
    event_p50_ms: f64,
    event_p99_ms: f64,
}

#[cfg(feature = "cuda")]
impl MeasuredRow {
    fn id(&self) -> String {
        format!("{}/{}", self.op, self.tag)
    }
}

#[cfg(feature = "cuda")]
fn build_cases(device: &candle_core::Device) -> candle_core::Result<Vec<Case>> {
    Ok(vec![
        // Launch-bound controls matching compare fixtures (B=2 H=8 S=128|512, D=64).
        rms_case(device, "s128", 2, 128, 64, true)?,
        rms_case(device, "s512", 2, 512, 64, true)?,
        swi_case(device, "s128", 2, 128, 64, true)?,
        swi_case(device, "s512", 2, 512, 64, true)?,
        rope_case(device, "s128", 2, 8, 128, 64, true)?,
        rope_case(device, "s512", 2, 8, 512, 64, true)?,
        ce_case(device, "s128", 2 * 128, 128, true)?,
        ce_case(device, "s512", 2 * 512, 128, true)?,
        attn_case(device, "s128", 2, 8, 128, 64, true)?,
        attn_case(device, "s512", 2, 8, 512, 64, false)?,
        // Larger but still launch-bound. Do not treat as the compute point.
        rms_case(device, "s2048_launch", 2, 2048, 128, true)?,
        swi_case(device, "s2048_launch", 2, 2048, 128, true)?,
        rope_case(device, "s2048_launch", 2, 8, 2048, 128, true)?,
        // Compute-bound (separate tags; hidden/V is not RoPE head-dim).
        rms_case(device, "compute", 1, 4096, 4096, false)?,
        swi_case(device, "compute", 1, 4096, 4096, false)?,
        ce_case(device, "compute", 512, 32768, false)?,
        // Fused linear+CE (no [N,V] logits). Launch-bound matches compare V=128;
        // compute uses the G-UNS-06 smoke-scale vocab tile.
        fused_ce_case(device, "s128", 2 * 128, 64, 128, true)?,
        fused_ce_case(device, "s512", 2 * 512, 64, 128, true)?,
        fused_ce_case(device, "compute", 512, 4096, 32768, false)?,
    ])
}

#[cfg(feature = "cuda")]
fn rms_case(
    device: &candle_core::Device,
    tag: &'static str,
    batch: usize,
    seq: usize,
    hidden: usize,
    launch_bound: bool,
) -> candle_core::Result<Case> {
    use candle_core::{DType, Tensor};
    let x = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), device)?;
    let w = Tensor::ones((hidden,), DType::F32, device)?;
    Ok(Case {
        id: format!("rmsnorm/{tag}"),
        op: "rmsnorm",
        tag,
        elems: batch * seq * hidden,
        vocab: None,
        n: None,
        launch_bound,
        work: Work {
            kind: WorkKind::Rms { x, w, eps: 1e-5 },
        },
    })
}

#[cfg(feature = "cuda")]
fn swi_case(
    device: &candle_core::Device,
    tag: &'static str,
    batch: usize,
    seq: usize,
    hidden: usize,
    launch_bound: bool,
) -> candle_core::Result<Case> {
    use candle_core::Tensor;
    let gate = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), device)?;
    let up = Tensor::randn(0.0f32, 1.0, (batch, seq, hidden), device)?;
    Ok(Case {
        id: format!("swiglu/{tag}"),
        op: "swiglu",
        tag,
        elems: batch * seq * hidden,
        vocab: None,
        n: None,
        launch_bound,
        work: Work {
            kind: WorkKind::Swi { gate, up },
        },
    })
}

#[cfg(feature = "cuda")]
fn rope_case(
    device: &candle_core::Device,
    tag: &'static str,
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    launch_bound: bool,
) -> candle_core::Result<Case> {
    use candle_core::Tensor;
    let x = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, head_dim), device)?;
    let half = head_dim / 2;
    let cos = Tensor::randn(0.0f32, 1.0, (seq, half), device)?;
    let sin = Tensor::randn(0.0f32, 1.0, (seq, half), device)?;
    Ok(Case {
        id: format!("rope/{tag}"),
        op: "rope",
        tag,
        elems: batch * heads * seq * head_dim,
        vocab: None,
        n: None,
        launch_bound,
        work: Work {
            kind: WorkKind::Rope { x, cos, sin },
        },
    })
}

#[cfg(feature = "cuda")]
fn ce_case(
    device: &candle_core::Device,
    tag: &'static str,
    n_tokens: usize,
    vocab: usize,
    launch_bound: bool,
) -> candle_core::Result<Case> {
    use candle_core::{DType, Tensor};
    let logits = Tensor::randn(0.0f32, 1.0, (n_tokens, vocab), device)?;
    let targets = Tensor::zeros((n_tokens,), DType::I64, device)?;
    Ok(Case {
        id: format!("ce/{tag}"),
        op: "ce",
        tag,
        elems: n_tokens * vocab,
        vocab: Some(vocab),
        n: Some(n_tokens),
        launch_bound,
        work: Work {
            kind: WorkKind::Ce { logits, targets },
        },
    })
}

#[cfg(feature = "cuda")]
fn fused_ce_case(
    device: &candle_core::Device,
    tag: &'static str,
    n_tokens: usize,
    dim: usize,
    vocab: usize,
    launch_bound: bool,
) -> candle_core::Result<Case> {
    use candle_core::{DType, Tensor};
    let hidden = Tensor::randn(0.0f32, 1.0, (n_tokens, dim), device)?;
    let weight = Tensor::randn(0.0f32, 1.0, (vocab, dim), device)?;
    let targets = Tensor::zeros((n_tokens,), DType::I64, device)?;
    Ok(Case {
        id: format!("fused_linear_ce/{tag}"),
        op: "fused_linear_ce",
        tag,
        elems: n_tokens * dim,
        vocab: Some(vocab),
        n: Some(n_tokens),
        launch_bound,
        work: Work {
            kind: WorkKind::FusedCe {
                hidden,
                weight,
                targets,
                chunk: 4096,
            },
        },
    })
}

#[cfg(feature = "cuda")]
fn attn_case(
    device: &candle_core::Device,
    tag: &'static str,
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    launch_bound: bool,
) -> candle_core::Result<Case> {
    use candle_core::Tensor;
    let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, head_dim), device)?;
    let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, head_dim), device)?;
    let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq, head_dim), device)?;
    let scale = (head_dim as f32).sqrt().recip();
    Ok(Case {
        id: format!("attn/{tag}"),
        op: "attn",
        tag,
        elems: batch * heads * seq * head_dim,
        vocab: None,
        n: None,
        launch_bound,
        work: Work {
            kind: WorkKind::Attn {
                q,
                k,
                v,
                scale,
                causal: true,
            },
        },
    })
}

#[cfg(feature = "cuda")]
fn warmup(work: &Work, n: usize) -> candle_core::Result<()> {
    for _ in 0..n {
        drop(work.run()?);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn write_json(
    path: &std::path::Path,
    gpu: &str,
    cuda_compute_cap: &str,
    gpu_compute_cap: &str,
    compute_apps: &str,
    rows: &[MeasuredRow],
) -> std::io::Result<()> {
    use std::fmt::Write as _;
    let mut s = String::new();
    writeln!(s, "{{").unwrap();
    writeln!(s, "  \"sacred_bar\": false,").unwrap();
    writeln!(s, "  \"gpu\": {},", json_str(gpu)).unwrap();
    writeln!(s, "  \"cuda_compute_cap\": {},", json_str(cuda_compute_cap)).unwrap();
    writeln!(s, "  \"gpu_compute_cap\": {},", json_str(gpu_compute_cap)).unwrap();
    writeln!(s, "  \"compute_apps\": {},", json_str(compute_apps.trim())).unwrap();
    writeln!(
        s,
        "  \"note\": \"Rust host+event p50/p99 after PTX cache. compile_cached is first-vs-second NVRTC (not a launch-tax close: Mutex + Arc + Candle dispatch remain). First NVRTC compile is outside timed regions. torch/Unsloth host+sync p50/p99 live in artifacts/py-rs-compare.json. fused_linear_ce is vocab-tile GEMM (no [N,V]). C-UNS-06-P50 harvest only; G-UNS-06 stays open. No 2x/VRAM claims. Not a G-UNS-01 close. 5080 numbers do not replace the C2 single-3090Ti sacred bar. cuda_compute_cap is the CUDA_COMPUTE_CAP pin (or unset), not proof of native SM.\","
    )
    .unwrap();
    writeln!(s, "  \"samples\": 100,").unwrap();
    writeln!(s, "  \"ops\": [").unwrap();
    for (i, r) in rows.iter().enumerate() {
        let vocab = match r.vocab {
            Some(v) => v.to_string(),
            None => "null".into(),
        };
        let ntok = match r.n {
            Some(v) => v.to_string(),
            None => "null".into(),
        };
        let _ = writeln!(
            s,
            "    {{\"op\":{},\"tag\":{},\"elems\":{},\"vocab\":{vocab},\"n_tokens\":{ntok},\"host_p50_ms\":{:.6},\"host_p99_ms\":{:.6},\"event_p50_ms\":{:.6},\"event_p99_ms\":{:.6},\"launch_bound\":{},\"compile_cached\":{}}}{}",
            json_str(r.op),
            json_str(r.tag),
            r.elems,
            r.host_p50_ms,
            r.host_p99_ms,
            r.event_p50_ms,
            r.event_p99_ms,
            r.launch_bound,
            r.compile_cached,
            if i + 1 == rows.len() { "" } else { "," }
        );
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, s)
}

#[cfg(feature = "cuda")]
fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                use std::fmt::Write as _;
                write!(out, "\\u{:04x}", u32::from(c)).unwrap();
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
