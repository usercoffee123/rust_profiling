use rand::Rng;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

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

fn random_matrix(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..n * n).map(|_| rng.r#gen::<f64>()).collect()
}

fn random_matrices(n: usize, count: usize) -> Vec<Vec<f64>> {
    (0..count).map(|_| random_matrix(n)).collect()
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

/// Multiplies `fixed` by `count` random n x n matrices using `rust_threads` worker threads,
/// each pinned to single-threaded OpenBLAS and receiving matrices through a channel.
fn bench_rust_threads(n: usize, count: usize, rust_threads: usize, fixed: &[f64]) -> Duration {
    unsafe { openblas_set_num_threads(1) };
    let matrices = random_matrices(n, count);
    let fixed = Arc::new(fixed.to_vec());
    let worker_count = rust_threads.min(count.max(1));

    let start = Instant::now();
    thread::scope(|scope| {
        let channels: Vec<(Sender<Vec<f64>>, Receiver<Vec<f64>>)> =
            (0..worker_count).map(|_| mpsc::channel()).collect();
        let (senders, receivers): (Vec<_>, Vec<_>) = channels.into_iter().unzip();

        for receiver in receivers {
            let fixed = Arc::clone(&fixed);
            scope.spawn(move || {
                unsafe { openblas_set_num_threads(1) };
                let mut c = vec![0.0f64; n * n];
                for matrix in receiver {
                    matmul(n, &fixed, &matrix, &mut c);
                }
            });
        }

        for (index, matrix) in matrices.into_iter().enumerate() {
            let sender = &senders[index % worker_count];
            sender
                .send(matrix)
                .expect("worker thread unexpectedly stopped");
        }

        drop(senders);
    });
    start.elapsed()
}

fn main() {
    let config = BenchmarkConfig {
        matrix_size: 512,
        multiplication_count: 128,
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
