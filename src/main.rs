use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

mod matrix;

use matrix::{random_matrices, random_matrix};

type BlasInt = i32;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

struct BenchmarkConfig {
    matrix_size: usize,
    multiplication_count: usize,
    thread_counts: &'static [usize],
}

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
    let matrices = random_matrices(n, count);
    let mut c = vec![0.0f64; n * n];

    let start = Instant::now();
    for m in &matrices {
        matmul(n, fixed, m, &mut c);
    }
    start.elapsed()
}

/// Multiplies `fixed` by `count` random n x n matrices split as evenly as possible across
/// `rust_threads` worker threads, each pinned to single-threaded OpenBLAS.
fn bench_rust_threads(n: usize, count: usize, rust_threads: usize, fixed: &[f64]) -> Duration {
    unsafe { openblas_set_num_threads(1) };
    let matrices = Arc::new(random_matrices(n, count));
    let fixed = Arc::new(fixed.to_vec());
    let worker_count = rust_threads.min(count.max(1));

    // First `remainder` workers take one extra matrix so sizes differ by at most 1.
    let base = count / worker_count;
    let remainder = count % worker_count;

    let start = Instant::now();
    thread::scope(|scope| {
        let mut offset = 0;
        for worker in 0..worker_count {
            let len = base + usize::from(worker < remainder);
            let range = offset..offset + len;
            offset += len;

            let fixed = Arc::clone(&fixed);
            let matrices = Arc::clone(&matrices);
            scope.spawn(move || {
                unsafe { openblas_set_num_threads(1) };
                let mut c = vec![0.0f64; n * n];
                for m in &matrices[range] {
                    matmul(n, &fixed, m, &mut c);
                }
            });
        }
    });
    start.elapsed()
}

fn main() {
    let config = BenchmarkConfig {
        matrix_size: 2048,
        multiplication_count: 256,
        thread_counts: &[1, 2, 4, 8],
    };
    let fixed = random_matrix(config.matrix_size);

    println!(
        "Matrix size: {size}x{size}, multiplications: {count}\n",
        size = config.matrix_size,
        count = config.multiplication_count,
    );

    println!("== OpenBLAS internal threading (single Rust thread) ==");
    for &threads in config.thread_counts {
        let elapsed = bench_blas_threads(
            config.matrix_size,
            config.multiplication_count,
            threads as i32,
            &fixed,
        );
        println!("blas_threads={threads:<2} time={elapsed:?}");
    }

    println!("\n== Rust-level threading (BLAS pinned to 1 thread) ==");
    for &threads in config.thread_counts {
        let elapsed = bench_rust_threads(
            config.matrix_size,
            config.multiplication_count,
            threads,
            &fixed,
        );
        println!("rust_threads={threads:<2} time={elapsed:?}");
    }
}
