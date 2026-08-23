use rand::Rng;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

type BlasInt = i32;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

unsafe extern "C" {
    fn openblas_set_num_threads(num_threads: i32);
    fn cblas_dgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: BlasInt,
        n: BlasInt,
        k: BlasInt,
        alpha: f64,
        a: *const f64,
        lda: BlasInt,
        b: *const f64,
        ldb: BlasInt,
        beta: f64,
        c: *mut f64,
        ldc: BlasInt,
    );
}

fn random_matrix(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..n * n).map(|_| rng.r#gen::<f64>()).collect()
}

fn matmul(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    let n_i = n as BlasInt;
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            n_i,
            n_i,
            n_i,
            1.0,
            a.as_ptr(),
            n_i,
            b.as_ptr(),
            n_i,
            0.0,
            c.as_mut_ptr(),
            n_i,
        );
    }
}

/// Multiplies `fixed` by `count` random n x n matrices sequentially, letting OpenBLAS
/// parallelize each individual `dgemm` call across `blas_threads` threads.
fn bench_blas_threads(n: usize, count: usize, blas_threads: i32, fixed: &[f64]) -> Duration {
    unsafe { openblas_set_num_threads(blas_threads) };
    let matrices: Vec<Vec<f64>> = (0..count).map(|_| random_matrix(n)).collect();
    let mut c = vec![0.0f64; n * n];

    let start = Instant::now();
    for m in &matrices {
        matmul(n, fixed, m, &mut c);
    }
    start.elapsed()
}

/// Multiplies `fixed` by `count` random n x n matrices, evenly split across `rust_threads`
/// OS threads, each pinned to single-threaded OpenBLAS.
fn bench_rust_threads(n: usize, count: usize, rust_threads: usize, fixed: &[f64]) -> Duration {
    unsafe { openblas_set_num_threads(1) };
    let matrices: Vec<Vec<f64>> = (0..count).map(|_| random_matrix(n)).collect();
    let fixed = Arc::new(fixed.to_vec());
    let chunk_size = count.div_ceil(rust_threads);

    let start = Instant::now();
    thread::scope(|scope| {
        for chunk in matrices.chunks(chunk_size.max(1)) {
            let fixed = Arc::clone(&fixed);
            scope.spawn(move || {
                unsafe { openblas_set_num_threads(1) };
                let mut c = vec![0.0f64; n * n];
                for m in chunk {
                    matmul(n, &fixed, m, &mut c);
                }
            });
        }
    });
    start.elapsed()
}

fn main() {
    let n = 512; // matrix dimension
    let count = 64; // number of multiplications
    let thread_counts = [1, 2, 4, 8];
    let fixed = random_matrix(n);

    println!("Matrix size: {n}x{n}, multiplications: {count}\n");

    println!("== OpenBLAS internal threading (single Rust thread) ==");
    for &threads in &thread_counts {
        let elapsed = bench_blas_threads(n, count, threads, &fixed);
        println!("blas_threads={threads:<2} time={elapsed:?}");
    }

    println!("\n== Rust-level threading (BLAS pinned to 1 thread) ==");
    for &threads in &thread_counts {
        let elapsed = bench_rust_threads(n, count, threads as usize, &fixed);
        println!("rust_threads={threads:<2} time={elapsed:?}");
    }
}
