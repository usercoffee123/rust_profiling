use rand::Rng;

pub fn random_matrix(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..n * n).map(|_| rng.r#gen::<f64>()).collect()
}

pub fn random_matrices(n: usize, count: usize) -> Vec<Vec<f64>> {
    (0..count).map(|_| random_matrix(n)).collect()
}
