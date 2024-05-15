mod utils;

#[cfg(test)]
mod test;

use wasm_bindgen::prelude::*;

#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;
mod tsne;

use crate::utils::set_panic_hook;
pub(crate) use num_traits::{cast::AsPrimitive, Float};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

/// t-distributed stochastic neighbor embedding. Provides a parallel implementation of both the
/// exact version of the algorithm and the tree accelerated one leveraging space partitioning trees.

#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct bhtSNE {
    tsne_encoder: tsne_encoder<f32>,
}
#[wasm_bindgen]
impl bhtSNE {
    #[wasm_bindgen(constructor)]
    pub fn new(data: JsValue) -> Self {
        set_panic_hook();

        let converted_data: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(data).unwrap();
        let d: usize = converted_data[0].len();

        let flattened_array: Vec<f32> = converted_data
            .into_par_iter()
            .flat_map(|inner_vec| inner_vec.into_par_iter())
            .collect();

        let tsne = tsne_encoder::new(flattened_array, d);
        Self { tsne_encoder: tsne }
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// `epochs` - Sets epochs, the maximum number of fitting iterations.
    pub fn step(&mut self, epochs: usize) -> Result<JsValue, JsValue> {
        self.tsne_encoder.epochs = epochs;
        self.tsne_encoder.barnes_hut(|sample_a, sample_b| {
            sample_a
                .iter()
                .zip(sample_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        });

        let embeddings: Vec<f32> = self.tsne_encoder.y.iter().map(|x| x.0).collect();
        let samples: Vec<Vec<f32>> = embeddings
            .chunks(self.tsne_encoder.no_dims)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(serde_wasm_bindgen::to_value(&samples)?)
    }

    /// Sets a new learning rate.
    ///
    /// # Arguments
    ///
    /// `learning_rate` - new value for the learning rate.
    pub fn learning_rate(&mut self, learning_rate: f32) {
        self.tsne_encoder.learning_rate = learning_rate;
    }

    /// Sets new epochs, i.e the maximum number of fitting iterations.
    ///
    /// # Arguments
    ///
    /// `epochs` - new value for the epochs.
    pub fn epochs(&mut self, epochs: usize) {
        self.tsne_encoder.epochs = epochs;
    }

    /// Sets a new momentum.
    ///
    /// # Arguments
    ///
    /// `momentum` - new value for the momentum.
    pub fn momentum(&mut self, momentum: f32) {
        self.tsne_encoder.momentum = momentum;
    }

    /// Sets a new final momentum.
    ///
    /// # Arguments
    ///
    /// `final_momentum` - new value for the final momentum.
    pub fn final_momentum(&mut self, final_momentum: f32) {
        self.tsne_encoder.final_momentum = final_momentum;
    }

    /// Sets a new momentum switch epoch, i.e. the epoch after which the algorithm switches to
    /// `final_momentum` for the map update.
    ///
    /// # Arguments
    ///
    /// `momentum_switch_epoch` - new value for the momentum switch epoch.
    pub fn momentum_switch_epoch(&mut self, momentum_switch_epoch: usize) {
        self.tsne_encoder.momentum_switch_epoch = momentum_switch_epoch;
    }

    /// Sets a new stop lying epoch, i.e. the epoch after which the P distribution values become
    /// true, as defined in the original implementation. For epochs < `stop_lying_epoch` the values
    /// of the P distribution are multiplied by a factor equal to `12.0`.
    ///
    /// # Arguments
    ///
    /// `stop_lying_epoch` - new value for the stop lying epoch.
    pub fn stop_lying_epoch(&mut self, stop_lying_epoch: usize) {
        self.tsne_encoder.stop_lying_epoch = stop_lying_epoch;
    }

    /// Sets a new theta, which determines the accuracy of the approximation. Must be **strictly greater than
    /// 0.0**. Large values for θ increase the speed of the algorithm but decrease its accuracy.
    /// For small values of θ it is less probable that a cell in the space partitioning tree will
    /// be treated as a single point. For θ equal to 0.0 the method degenerates in the exact
    /// version.
    ///
    /// # Arguments
    ///
    /// * `theta`  - new value for the theta.
    pub fn theta(&mut self, theta: f32) {
        assert!(
            theta > 0.0f32,
            "error: theta value must be greater than 0.0.
            A value of 0.0 corresponds to using the exact version of the algorithm."
        );
        self.tsne_encoder.theta = theta;
    }

    /// Sets a new value for the embedding dimension.
    ///
    /// # Arguments
    ///
    /// `embedding_dim` - new value for the embedding space dimensionality.
    pub fn embedding_dim(&mut self, embedding_dim: u8) {
        self.tsne_encoder.embedding_dim = embedding_dim;
    }

    /// Sets a new perplexity value.
    ///
    /// # Arguments
    ///
    /// `perplexity` - new value for the perplexity. It's used so that the bandwidth of the Gaussian
    ///  kernels, is set in such a way that the perplexity of each the conditional distribution *Pi*
    ///  equals a predefined perplexity *u*.
    ///
    /// A good value for perplexity lies between 5.0 and 50.0.
    pub fn perplexity(&mut self, perplexity: f32) {
        self.tsne_encoder.perplexity = perplexity;
    }
}

struct TsneBuilder<'data, U>
where
    U: Send + Sync,
{
    data: &'data [U],
}

impl<'data, U> TsneBuilder<'data, U>
where
    U: Send + Sync,
{
    pub fn new(data: &'data [U]) -> Self {
        Self { data }
    }
}

#[allow(non_camel_case_types)]
pub struct tsne_encoder<T>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
{
    data: Vec<T>,
    d: usize,
    theta: T,
    no_dims: usize,
    learning_rate: T,
    epochs: usize,
    momentum: T,
    final_momentum: T,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    embedding_dim: u8,
    perplexity: T,
    p_values: Vec<tsne::Aligned<T>>,
    p_rows: Vec<usize>,
    p_columns: Vec<usize>,
    y: Vec<tsne::Aligned<T>>,
    dy: Vec<tsne::Aligned<T>>,
    uy: Vec<tsne::Aligned<T>>,
    gains: Vec<tsne::Aligned<T>>,
}

impl<T> tsne_encoder<T>
where
    T: Float
        + Send
        + Sync
        + AsPrimitive<usize>
        + Sum
        + DivAssign
        + AddAssign
        + MulAssign
        + SubAssign,
{
    pub fn new(data: Vec<T>, d: usize) -> Self {
        Self {
            data,
            d,
            theta: T::from(0.5).unwrap(),
            no_dims: 2,
            learning_rate: T::from(200.0).unwrap(),
            epochs: 1000,
            momentum: T::from(0.5).unwrap(),
            final_momentum: T::from(0.8).unwrap(),
            momentum_switch_epoch: 250,
            stop_lying_epoch: 250,
            embedding_dim: 2,
            perplexity: T::from(20.0).unwrap(),
            p_values: Vec::new(),
            p_rows: Vec::new(),
            p_columns: Vec::new(),
            y: Vec::new(),
            dy: Vec::new(),
            uy: Vec::new(),
            gains: Vec::new(),
        }
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// * `theta` - determines the accuracy of the approximation. Must be **strictly greater than
    /// 0.0**. Large values for θ increase the speed of the algorithm but decrease its accuracy.
    /// For small values of θ it is less probable that a cell in the space partitioning tree will
    /// be treated as a single point. For θ equal to 0.0 the method degenerates in the exact
    /// version.
    ///
    /// * `metric_f` - metric function.
    ///
    ///
    /// **Do note that** `metric_f` **must be a metric distance**, i.e. it must
    /// satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
    pub fn barnes_hut<F: Fn(&&[T], &&[T]) -> T + Send + Sync>(&mut self, metric_f: F) -> &mut Self {
        let samples: Vec<&[T]> = self.data.chunks(self.d).collect::<Vec<&[T]>>();
        let tsne_builder = TsneBuilder::new(&samples);

        let data = tsne_builder.data;
        let n_samples = tsne_builder.data.len(); // Number of samples in data.

        let theta = self.theta;

        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &n_samples);

        let embedding_dim = self.embedding_dim as usize;
        // Number of  points ot consider when approximating the conditional distribution P.
        let n_neighbors: usize = (T::from(3.0).unwrap() * self.perplexity).as_();
        // NUmber of entries in gradient and gains matrices.
        let grad_entries = n_samples * embedding_dim;
        // Number of entries in pairwise measures matrices.
        let pairwise_entries = n_samples * n_neighbors;

        // Prepare buffers
        tsne::prepare_buffers(
            &mut self.y,
            &mut self.dy,
            &mut self.uy,
            &mut self.gains,
            &grad_entries,
        );
        // The P distribution values are restricted to a subset of size n_neighbors for each input
        // sample.
        self.p_values.resize(pairwise_entries, T::zero().into());

        // This vector is used to keep track of the indexes for each nearest neighbors of each
        // sample. There's a one to one correspondence between the elements of p_columns
        // an the elements of p_values: for each row i of length n_neighbors of such matrices it
        // holds that p_columns[i][j] corresponds to the index sample which contributes
        // to p_values[i][j]. This vector is freed inside symmetrize_sparse_matrix.
        let mut p_columns: Vec<tsne::Aligned<usize>> = vec![0.into(); pairwise_entries];

        // Computes sparse input similarities using a vantage point tree.
        {
            // Distances buffer.
            let mut distances: Vec<tsne::Aligned<T>> = vec![T::zero().into(); pairwise_entries];

            // Build ball tree on data set. The tree is freed at the end of the scope.
            let tree = tsne::VPTree::new(data, &metric_f);

            // For each sample in the dataset compute the perplexities using a vantage point tree
            // in parallel.
            {
                let perplexity = &self.perplexity; // Immutable borrow must be outside.
                self.p_values
                    .par_chunks_mut(n_neighbors)
                    .zip(distances.par_chunks_mut(n_neighbors))
                    .zip(p_columns.par_chunks_mut(n_neighbors))
                    .zip(data.par_iter())
                    .enumerate()
                    .for_each(
                        |(index, (((p_values_row, distances_row), p_columns_row), sample))| {
                            // Writes the indices and the distances of the nearest neighbors of sample.
                            tree.search(
                                sample,
                                index,
                                n_neighbors + 1, // The first NN is sample itself.
                                p_columns_row,
                                distances_row,
                                &metric_f,
                            );
                            debug_assert!(!p_columns_row.iter().any(|i| i.0 == index));
                            tsne::search_beta(p_values_row, distances_row, perplexity);
                        },
                    );
            }
        }

        // Symmetrize sparse P matrix.
        tsne::symmetrize_sparse_matrix(
            &mut self.p_rows,
            &mut self.p_columns,
            p_columns,
            &mut self.p_values,
            n_samples,
            &n_neighbors,
        );

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Prepares buffers for Barnes-Hut algorithm.
        let mut positive_forces: Vec<tsne::Aligned<T>> = vec![T::zero().into(); grad_entries];
        let mut negative_forces: Vec<tsne::Aligned<T>> = vec![T::zero().into(); grad_entries];
        let mut forces_buffer: Vec<tsne::Aligned<T>> = vec![T::zero().into(); grad_entries];
        let mut q_sums: Vec<tsne::Aligned<T>> = vec![T::zero().into(); n_samples];

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        let mut means: Vec<T> = vec![T::zero(); embedding_dim];

        // Main Training loop.
        for epoch in 0..self.epochs {
            {
                // Construct space partitioning tree on current embedding.
                let tree = tsne::SPTree::new(&embedding_dim, &self.y, &n_samples);
                // Check if the SPTree is correct.
                debug_assert!(tree.is_correct(), "error: SPTree is not correct.");

                // Computes forces using the Barnes-Hut algorithm in parallel.
                // Each chunk of positive_forces and negative_forces is associated to a distinct
                // embedded sample in y. As a consequence of this the computation can be done in
                // parallel.
                positive_forces
                    .par_chunks_mut(embedding_dim)
                    .zip(negative_forces.par_chunks_mut(embedding_dim))
                    .zip(forces_buffer.par_chunks_mut(embedding_dim))
                    .zip(q_sums.par_iter_mut())
                    .zip(self.y.par_chunks(embedding_dim))
                    .enumerate()
                    .for_each(
                        |(
                            index,
                            (
                                (
                                    ((positive_forces_row, negative_forces_row), forces_buffer_row),
                                    q_sum,
                                ),
                                sample,
                            ),
                        )| {
                            tree.compute_edge_forces(
                                index,
                                sample,
                                &self.p_rows,
                                &self.p_columns,
                                &self.p_values,
                                forces_buffer_row,
                                positive_forces_row,
                            );
                            tree.compute_non_edge_forces(
                                index,
                                &theta,
                                negative_forces_row,
                                forces_buffer_row,
                                q_sum,
                            );
                        },
                    );
            }

            // Compute final Barnes-Hut t-SNE gradient approximation.
            // Reduces partial sums of Q distribution.
            let q_sum: T = q_sums.par_iter_mut().map(|sum| sum.0).sum();
            self.dy
                .par_iter_mut()
                .zip(positive_forces.par_iter_mut())
                .zip(negative_forces.par_iter_mut())
                .for_each(|((grad, pf), nf)| {
                    grad.0 = pf.0 - (nf.0 / q_sum);
                    pf.0 = T::zero();
                    nf.0 = T::zero();
                });
            // Zeroes Q-sums.
            q_sums.par_iter_mut().for_each(|sum| sum.0 = T::zero());

            // Updates the embedding in parallel with gradient descent.
            tsne::update_solution(
                &mut self.y,
                &self.dy,
                &mut self.uy,
                &mut self.gains,
                &self.learning_rate,
                &self.momentum,
            );

            // Make solution zero-mean.
            tsne::zero_mean(&mut means, &mut self.y, &n_samples, &embedding_dim);

            // Stop lying about the P-values if the time is right.
            if epoch == self.stop_lying_epoch {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if epoch == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }
        }
        // Clears buffers used for fitting.
        tsne::clear_buffers(&mut self.dy, &mut self.uy, &mut self.gains);
        self
    }
}
