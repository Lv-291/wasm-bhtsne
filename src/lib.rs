mod tsne;

mod distance_functions;
use distance_functions::DistanceFunction;
mod utils;

use js_sys::Array;

use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};

use wasm_bindgen::prelude::wasm_bindgen;

/// t-distributed stochastic neighbor embedding. Provides a parallel implementation of both the
/// exact version of the algorithm and the tree accelerated one leveraging space partitioning trees.
#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct tSNE {
    data: Vec<f32>,
    d: usize,
    learning_rate: f32,
    momentum: f32,
    final_momentum: f32,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    embedding_dim: usize,
    perplexity: f32,
    p_values: Vec<tsne::Aligned<f32>>,
    p_rows: Vec<usize>,
    p_columns: Vec<usize>,
    y: Vec<tsne::Aligned<f32>>,
    dy: Vec<tsne::Aligned<f32>>,
    uy: Vec<tsne::Aligned<f32>>,
    gains: Vec<tsne::Aligned<f32>>,
    distance_f: fn(&&[f32], &&[f32]) -> f32,
    positive_forces: Vec<tsne::Aligned<f32>>,
    negative_forces: Vec<tsne::Aligned<f32>>,
    forces_buffer: Vec<tsne::Aligned<f32>>,
    q_sums: Vec<tsne::Aligned<f32>>,
    means: Vec<f32>,
    n_samples: usize,
    theta: f32,
}

#[wasm_bindgen]
impl tSNE {
    #[wasm_bindgen(constructor)]
    pub fn new(vectors: Array) -> tSNE {
        utils::set_panic_hook();

        let mut tsne = tSNE {
            data: utils::convert_array_to_vec(&vectors),
            d: utils::get_num_cols(&vectors),
            learning_rate: 200.0,
            momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_epoch: 250,
            stop_lying_epoch: 250,
            embedding_dim: 2,
            perplexity: 20.0,
            p_values: Vec::new(),
            p_rows: Vec::new(),
            p_columns: Vec::new(),
            y: Vec::new(),
            dy: Vec::new(),
            uy: Vec::new(),
            gains: Vec::new(),
            distance_f: DistanceFunction::Euclidean.get_closure(),
            positive_forces: Vec::new(),
            negative_forces: Vec::new(),
            forces_buffer: Vec::new(),
            q_sums: Vec::new(),
            means: Vec::new(),
            n_samples: 0,
            theta: 1.0,
        };

        tsne.barnes_hut_data();
        tsne
    }

    fn barnes_hut_data(&mut self) {

        let vectors: Vec<&[f32]> = self.data.chunks(self.d).collect();
        self.n_samples = vectors.len(); // Number of samples in data.

        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &self.n_samples);

        // Number of  points ot consider when approximating the conditional distribution P.
        let n_neighbors: usize = (3.0f32 * self.perplexity) as usize;
        // NUmber of entries in gradient and gains matrices.
        let grad_entries = self.n_samples * self.embedding_dim;
        // Number of entries in pairwise measures matrices.
        let pairwise_entries = self.n_samples * n_neighbors;

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
        self.p_values.resize(pairwise_entries, 0.0f32.into());

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
        self.p_values.resize(pairwise_entries, 0.0f32.into());

        // This vector is used to keep track of the indexes for each nearest neighbors of each
        // sample. There's a one to one correspondence between the elements of p_columns
        // an the elements of p_values: for each row i of length n_neighbors of such matrices it
        // holds that p_columns[i][j] corresponds to the index sample which contributes
        // to p_values[i][j]. This vector is freed inside symmetrize_sparse_matrix.
        let mut p_columns: Vec<tsne::Aligned<usize>> = vec![0.into(); pairwise_entries];

        // Computes sparse input similarities using a vantage point tree.
        {
            // Distances buffer.
            let mut distances: Vec<tsne::Aligned<f32>> = vec![0.0f32.into(); pairwise_entries];

            let metric_f = self.distance_f;

            // Build ball tree on data set. The tree is freed at the end of the scope.
            let tree = tsne::VPTree::new(&vectors, self.distance_f);

            // For each sample in the dataset compute the perplexities using a vantage point tree
            // in parallel.
            {
                let perplexity = &self.perplexity; // Immutable borrow must be outside.
                self.p_values
                    .par_chunks_mut(n_neighbors)
                    .zip(distances.par_chunks_mut(n_neighbors))
                    .zip(p_columns.par_chunks_mut(n_neighbors))
                    .zip(vectors.par_iter())
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
                                metric_f,
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
            self.n_samples,
            &n_neighbors,
        );

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Prepares buffers for Barnes-Hut algorithm.
        self.positive_forces = vec![0.0f32.into(); grad_entries];
        self.negative_forces = vec![0.0f32.into(); grad_entries];
        self.forces_buffer = vec![0.0f32.into(); grad_entries];
        self.q_sums = vec![0.0f32.into(); self.n_samples];

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        self.means = vec![0.0f32; self.embedding_dim];
    }

    #[wasm_bindgen]
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
    pub fn barnes_hut(&mut self, epochs: usize) -> Array {
        // Checks that theta is valid.


        let mut positive_forces = self.positive_forces.clone();
        let mut negative_forces = self.negative_forces.clone();
        let mut forces_buffer = self.forces_buffer.clone();
        let mut q_sums = self.q_sums.clone();
        let mut means = self.means.clone();

        // Main Training loop.
        for epoch in 0..epochs {
            {
                // Construct space partitioning tree on current embedding.
                let tree = tsne::SPTree::new(&self.embedding_dim, &self.y, &self.n_samples);
                // Check if the SPTree is correct.
                debug_assert!(tree.is_correct(), "error: SPTree is not correct.");

                // Computes forces using the Barnes-Hut algorithm in parallel.
                // Each chunk of positive_forces and negative_forces is associated to a distinct
                // embedded sample in y. As a consequence of this the computation can be done in
                // parallel.
                positive_forces
                    .par_chunks_mut(self.embedding_dim)
                    .zip(negative_forces.par_chunks_mut(self.embedding_dim))
                    .zip(forces_buffer.par_chunks_mut(self.embedding_dim))
                    .zip(q_sums.par_iter_mut())
                    .zip(self.y.par_chunks(self.embedding_dim))
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
                                &self.theta,
                                negative_forces_row,
                                forces_buffer_row,
                                q_sum,
                            );
                        },
                    );
            }

            // Compute final Barnes-Hut t-SNE gradient approximation.
            // Reduces partial sums of Q distribution.
            let q_sum: f32 = q_sums.par_iter_mut().map(|sum| sum.0).sum();
            self.dy
                .par_iter_mut()
                .zip(positive_forces.par_iter_mut())
                .zip(negative_forces.par_iter_mut())
                .for_each(|((grad, pf), nf)| {
                    grad.0 = pf.0 - (nf.0 / q_sum);
                    pf.0 = 0.0f32;
                    nf.0 = 0.0f32;
                });
            // Zeroes Q-sums.
            q_sums.par_iter_mut().for_each(|sum| sum.0 = 0.0f32);

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
            tsne::zero_mean(&mut means, &mut self.y, &self.n_samples, &self.embedding_dim);

            // Stop lying about the P-values if the time is right.
            if epoch == self.stop_lying_epoch {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if epoch == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }
        }

        self.positive_forces = positive_forces;
        self.negative_forces = negative_forces;
        self.forces_buffer = forces_buffer;
        self.q_sums = q_sums;
        self.means = means;

        self.embedding()
    }

    #[wasm_bindgen(setter)]
    /// Sets a new learning rate.
    ///
    /// # Arguments
    ///
    /// `learning_rate` - new value for the learning rate.
    pub fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }


    #[wasm_bindgen(setter)]
    pub fn set_theta(&mut self, theta: f32) {
        assert!(
            theta > 0.0,
            "error: theta value must be greater than 0.0.
            A value of 0.0 corresponds to using the exact version of the algorithm."
        );
        self.theta = theta;
    }

    #[wasm_bindgen(setter)]
    /// Sets a new momentum.
    ///
    /// # Arguments
    ///
    /// `momentum` - new value for the momentum.
    pub fn set_momentum(&mut self, momentum: f32) {
        self.momentum = momentum;
    }

    #[wasm_bindgen(setter)]
    /// Sets a new final momentum.
    ///
    /// # Arguments
    ///
    /// `final_momentum` - new value for the final momentum.
    pub fn set_final_momentum(&mut self, final_momentum: f32) {
        self.final_momentum = final_momentum;
    }

    #[wasm_bindgen(setter)]
    /// Sets a new momentum switch epoch, i.e. the epoch after which the algorithm switches to
    /// `final_momentum` for the map update.
    ///
    /// # Arguments
    ///
    /// `momentum_switch_epoch` - new value for the momentum switch epoch.
    pub fn set_momentum_switch_epoch(&mut self, momentum_switch_epoch: usize) {
        self.momentum_switch_epoch = momentum_switch_epoch;
    }

    #[wasm_bindgen(setter)]
    /// Sets a new stop lying epoch, i.e. the epoch after which the P distribution values become
    /// true, as defined in the original implementation. For epochs < `stop_lying_epoch` the values
    /// of the P distribution are multiplied by a factor equal to `12.0`.
    ///
    /// # Arguments
    ///
    /// `stop_lying_epoch` - new value for the stop lying epoch.
    pub fn set_stop_lying_epoch(&mut self, stop_lying_epoch: usize) {
        self.stop_lying_epoch = stop_lying_epoch;
    }

    #[wasm_bindgen(setter)]
    /// Sets a new value for the embedding dimension.
    ///
    /// # Arguments
    ///
    /// `embedding_dim` - new value for the embedding space dimensionality.
    pub fn set_embedding_dim(&mut self, embedding_dim: usize) {
        self.embedding_dim = embedding_dim;
    }

    #[wasm_bindgen(setter)]
    /// Sets a new perplexity value.
    ///
    /// # Arguments
    ///
    /// `perplexity` - new value for the perplexity. It's used so that the bandwidth of the Gaussian
    ///  kernels, is set in such a way that the perplexity of each the conditional distribution *Pi*
    ///  equals a predefined perplexity *u*.
    ///
    /// A good value for perplexity lies between 5.0 and 50.0.
    pub fn set_perplexity(&mut self, perplexity: f32) {
        self.perplexity = perplexity;
    }

    #[wasm_bindgen]
    /// Returns the computed embedding.
    pub fn embedding(&self) -> Array {
        let result_data: Vec<f32> = self.y.iter().map(|x| x.0).collect();
        utils::convert_to_array_of_arrays(result_data, self.embedding_dim)
    }
}
