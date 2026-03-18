#[cfg(test)]
mod tests {
    #[test]
    fn test_distance() {
        let mut vec = vec![3.0, 4.0];
        let mut sum_sq: f32 = 0.0;
        for val in &vec {
            sum_sq += val * val;
        }
        let norm = sum_sq.sqrt().max(1e-10);
        for val in &mut vec {
            *val /= norm;
        }
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
        let new_norm = (vec[0] * vec[0] + vec[1] * vec[1]).sqrt();
        assert!((new_norm - 1.0).abs() < 1e-6);
    }
}
