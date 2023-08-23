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
    epochs: usize,
    momentum: f32,
    final_momentum: f32,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    embedding_dim: usize,
    perplexity: f32,
    p_values: Vec<tsne::Aligned<f32>>,
    p_rows: Vec<usize>,
    p_columns: Vec<usize>,
    q_values: Vec<tsne::Aligned<f32>>,
    y: Vec<tsne::Aligned<f32>>,
    dy: Vec<tsne::Aligned<f32>>,
    uy: Vec<tsne::Aligned<f32>>,
    gains: Vec<tsne::Aligned<f32>>,
    distance_f: fn(&&[f32], &&[f32]) -> f32,
}
#[wasm_bindgen]
impl tSNE {
    #[wasm_bindgen(constructor)]
    pub fn new(vectors: Array) -> tSNE {
        utils::set_panic_hook();

        tSNE {
            data: utils::convert_array_to_vec(&vectors),
            d: utils::get_num_cols(&vectors),
            learning_rate: 200.0,
            epochs: 1000,
            momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_epoch: 250,
            stop_lying_epoch: 250,
            embedding_dim: 2,
            perplexity: 20.0,
            p_values: Vec::new(),
            p_rows: Vec::new(),
            p_columns: Vec::new(),
            q_values: Vec::new(),
            y: Vec::new(),
            dy: Vec::new(),
            uy: Vec::new(),
            gains: Vec::new(),
            distance_f: DistanceFunction::Euclidean.get_closure(),
        }
    }

    #[wasm_bindgen]
    /// Performs a parallel exact version of the t-SNE algorithm. Pairwise distances between samples
    /// in the input space will be computed accordingly to the supplied function `distance_f`.
    ///
    /// # Arguments
    ///
    /// `distance_f` - distance function.
    ///
    /// **Do note** that such a distance function needs not to be a metric distance, i.e. it is not
    /// necessary for it so satisfy the triangle inequality. Consequently, the squared euclidean
    /// distance, and many other, can be used.
    pub fn exact(&mut self) -> Array {
        let vectors: Vec<&[f32]> = self.data.chunks(self.d).collect();
        let n_samples = vectors.len(); // Number of samples in data.

        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &n_samples);

        let embedding_dim = self.embedding_dim as usize;
        // NUmber of entries in gradient and gains matrices.
        let grad_entries = n_samples * embedding_dim;
        // Number of entries in pairwise measures matrices.
        let pairwise_entries = n_samples * n_samples;

        // Prepares the buffers.
        tsne::prepare_buffers(
            &mut self.y,
            &mut self.dy,
            &mut self.uy,
            &mut self.gains,
            &grad_entries,
        );
        // Prepare distributions matrices.
        self.p_values.resize(pairwise_entries, 0.0f32.into()); // P.
        self.q_values.resize(pairwise_entries, 0.0f32.into()); // Q.

        // Alignment prevents false sharing.
        let mut distances: Vec<tsne::Aligned<f32>> = vec![0.0f32.into(); pairwise_entries];
        // Zeroes the diagonal entries. The distances vector is recycled but the elements
        // corresponding to the diagonal entries of the distance matrix are always kept to 0. and
        // never written on. This hold as an invariant through all the algorithm.
        for i in 0..n_samples {
            distances[i * n_samples + i] = 0.0f32.into();
        }

        // Compute pairwise distances in parallel with the user supplied function.
        // Only upper triangular entries, excluding the diagonal are computed: flat indexes are
        // unraveled to pick such entries.

        tsne::compute_pairwise_distance_matrix(
            &mut distances,
            self.distance_f,
            |index| &vectors[*index],
            &n_samples,
        );

        // Compute gaussian perplexity in parallel. First, the conditional distribution is computed
        // for each element. Each row of the P matrix is independent from the others, thus, this
        // computation is accordingly parallelized.
        {
            let perplexity = &self.perplexity;
            self.p_values
                .par_chunks_mut(n_samples)
                .zip(distances.par_chunks(n_samples))
                .for_each(|(p_values_row, distances_row)| {
                    tsne::search_beta(p_values_row, distances_row, perplexity);
                });
        }

        // Symmetrize pairwise input similarities. Conditional probabilities must be summed to
        // obtain the joint P distribution.
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let symmetric = self.p_values[j * n_samples + i].0;
                self.p_values[i * n_samples + j].0 += symmetric;
                self.p_values[j * n_samples + i].0 = self.p_values[i * n_samples + j].0;
            }
        }

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        let mut means: Vec<f32> = vec![0.0f32; embedding_dim];

        // Main fitting loop.
        for epoch in 0..self.epochs {
            // Compute pairwise squared euclidean distances between embeddings in parallel.
            tsne::compute_pairwise_distance_matrix(
                &mut distances,
                |ith: &[tsne::Aligned<f32>], jth: &[tsne::Aligned<f32>]| {
                    ith.iter()
                        .zip(jth.iter())
                        .map(|(i, j)| (i.0 - j.0).powi(2))
                        .sum()
                },
                |index| &self.y[index * embedding_dim..index * embedding_dim + embedding_dim],
                &n_samples,
            );

            // Computes Q.
            self.q_values
                .par_iter_mut()
                .zip(distances.par_iter())
                .for_each(|(q, d)| q.0 = 1.0f32 / (1.0f32 + d.0));

            // Computes the exact gradient in parallel.
            let q_values_sum: f32 = self.q_values.par_iter().map(|q| q.0).sum();

            // Immutable borrow to self must happen outside of the inner sequential
            // loop. The outer parallel loop already has a mutable borrow.
            let y = &self.y;
            self.dy
                .par_chunks_mut(embedding_dim)
                .zip(self.y.par_chunks(embedding_dim))
                .zip(self.p_values.par_chunks(n_samples))
                .zip(self.q_values.par_chunks(n_samples))
                .for_each(
                    |(((dy_sample, y_sample), p_values_sample), q_values_sample)| {
                        p_values_sample
                            .iter()
                            .zip(q_values_sample.iter())
                            .zip(y.chunks(embedding_dim))
                            .for_each(|((p, q), other_sample)| {
                                let m: f32 = (p.0 - q.0 / q_values_sum) * q.0;
                                dy_sample
                                    .iter_mut()
                                    .zip(y_sample.iter())
                                    .zip(other_sample.iter())
                                    .for_each(|((dy_el, y_el), other_el)| {
                                        dy_el.0 += (y_el.0 - other_el.0) * m
                                    });
                            });
                    },
                );

            // Updates the embedding in parallel with gradient descent.
            tsne::update_solution(
                &mut self.y,
                &self.dy,
                &mut self.uy,
                &mut self.gains,
                &self.learning_rate,
                &self.momentum,
            );
            // Zeroes the gradient.
            self.dy.iter_mut().for_each(|el| el.0 = 0.0f32);

            // Make solution zero mean.
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

        self.embedding()
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
    pub fn barnes_hut(&mut self, theta: f32) -> Array {
        // Checks that theta is valid.
        assert!(
            theta > 0.0,
            "error: theta value must be greater than 0.0.
            A value of 0.0 corresponds to using the exact version of the algorithm."
        );

        let vectors: Vec<&[f32]> = self.data.chunks(self.d).collect();
        let n_samples = vectors.len(); // Number of samples in data.

        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &n_samples);

        let embedding_dim = self.embedding_dim;
        // Number of  points ot consider when approximating the conditional distribution P.
        let n_neighbors: usize = (3.0f32 * self.perplexity) as usize;
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
            n_samples,
            &n_neighbors,
        );

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Prepares buffers for Barnes-Hut algorithm.
        let mut positive_forces: Vec<tsne::Aligned<f32>> = vec![0.0f32.into(); grad_entries];
        let mut negative_forces: Vec<tsne::Aligned<f32>> = vec![0.0f32.into(); grad_entries];
        let mut forces_buffer: Vec<tsne::Aligned<f32>> = vec![0.0f32.into(); grad_entries];
        let mut q_sums: Vec<tsne::Aligned<f32>> = vec![0.0f32.into(); n_samples];

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        let mut means: Vec<f32> = vec![0.0f32; embedding_dim];

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
    /// Sets new epochs, i.e the maximum number of fitting iterations.
    ///
    /// # Arguments
    ///
    /// `epochs` - new value for the epochs.
    pub fn set_epochs(&mut self, epochs: usize) {
        self.epochs = epochs;
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
