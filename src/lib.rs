mod hyperparameters;
mod tsne;
mod utils;

use crate::hyperparameters::Hyperparameters;
use crate::utils::set_panic_hook;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

mod algorithm;
#[cfg(test)]
mod test;
use crate::algorithm::tsne_encoder;

#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// t-distributed stochastic neighbor embedding. Provides a parallel implementation of both the
/// exact version of the algorithm and the tree accelerated one leveraging space partitioning trees.
#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct bhtSNEf32 {
    tsne_encoder: tsne_encoder<f32>,
}

#[wasm_bindgen]
impl bhtSNEf32 {
    #[wasm_bindgen(constructor)]
    pub fn new(data: JsValue, opt: JsValue) -> Self {
        set_panic_hook();
        let converted_data: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(data).unwrap();
        let hyperparameters: Hyperparameters<f32> = serde_wasm_bindgen::from_value(opt).unwrap();

        let mut tsne = tsne_encoder::new(converted_data, hyperparameters);

        tsne.barnes_hut_data(|sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });
        Self { tsne_encoder: tsne }
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// `epochs` - the maximum number of fitting iterations. Must be positive
    pub fn step(&mut self, epochs: usize) -> Result<JsValue, JsValue> {
        self.tsne_encoder.run(epochs);

        let embeddings: Vec<f32> = self.tsne_encoder.embeddings();
        let samples: Vec<Vec<f32>> = embeddings
            .chunks(self.tsne_encoder.no_dims)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(serde_wasm_bindgen::to_value(&samples)?)
    }
}

#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct bhtSNEf64 {
    tsne_encoder: tsne_encoder<f64>,
}

#[wasm_bindgen]
impl bhtSNEf64 {
    #[wasm_bindgen(constructor)]
    pub fn new(data: JsValue, opt: JsValue) -> Self {
        set_panic_hook();
        let converted_data: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data).unwrap();
        let hyperparameters: Hyperparameters<f64> = serde_wasm_bindgen::from_value(opt).unwrap();

        let mut tsne = tsne_encoder::new(converted_data, hyperparameters);

        tsne.barnes_hut_data(|sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        });
        Self { tsne_encoder: tsne }
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// `epochs` - Sets epochs, the maximum number of fitting iterations.
    pub fn step(&mut self, epochs: usize) -> Result<JsValue, JsValue> {
        self.tsne_encoder.run(epochs);

        let embeddings: Vec<f64> = self.tsne_encoder.embeddings();
        let samples: Vec<Vec<f64>> = embeddings
            .chunks(self.tsne_encoder.no_dims)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(serde_wasm_bindgen::to_value(&samples)?)
    }
}
