pub enum DistanceFunction {
    Euclidean,
}

impl DistanceFunction {
    pub fn get_closure(&self) -> fn(&&[f32], &&[f32]) -> f32 {
        match self {
            DistanceFunction::Euclidean => |sample_a: &&[f32], sample_b: &&[f32]| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            },
        }
    }
}
