use crate::tsne;

use crate::hyperparameters::Hyperparameters;
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
    pub(crate) theta: T,
    pub(crate) no_dims: usize,
    learning_rate: T,
    epochs: usize,
    momentum: T,
    final_momentum: T,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    pub(crate) embedding_dim: usize,
    perplexity: T,
    pub(crate) p_values: Vec<tsne::Aligned<T>>,
    pub(crate) p_rows: Vec<usize>,
    pub(crate) p_columns: Vec<usize>,
    pub(crate) y: Vec<tsne::Aligned<T>>,
    dy: Vec<tsne::Aligned<T>>,
    uy: Vec<tsne::Aligned<T>>,
    gains: Vec<tsne::Aligned<T>>,
    positive_forces: Vec<tsne::Aligned<T>>,
    negative_forces: Vec<tsne::Aligned<T>>,
    forces_buffer: Vec<tsne::Aligned<T>>,
    q_sums: Vec<tsne::Aligned<T>>,
    means: Vec<T>,
    n_samples: usize,
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
    pub fn new(data: Vec<Vec<T>>, hyperparameters: Hyperparameters<T>) -> Self {
        let d: usize = data[0].len();

        let flattened_array: Vec<T> = data
            .into_par_iter()
            .flat_map(|inner_vec| inner_vec.into_par_iter())
            .collect();

        assert!(
            hyperparameters.theta > T::from(0.0).unwrap(),
            "error: theta value must be greater than 0.0.
            A value of 0.0 corresponds to using the exact version of the algorithm."
        );

        Self {
            data: flattened_array,
            d,
            theta: hyperparameters.theta,
            no_dims: 2,
            learning_rate: hyperparameters.learning_rate,
            epochs: 0,
            momentum: hyperparameters.momentum,
            final_momentum: hyperparameters.final_momentum,
            momentum_switch_epoch: hyperparameters.momentum_switch_epoch,
            stop_lying_epoch: hyperparameters.stop_lying_epoch,
            embedding_dim: hyperparameters.embedding_dim,
            perplexity: hyperparameters.perplexity,
            p_values: Vec::new(),
            p_rows: Vec::new(),
            p_columns: Vec::new(),
            y: Vec::new(),
            dy: Vec::new(),
            uy: Vec::new(),
            gains: Vec::new(),
            positive_forces: Vec::new(),
            negative_forces: Vec::new(),
            forces_buffer: Vec::new(),
            q_sums: Vec::new(),
            means: Vec::new(),
            n_samples: 0,
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
    pub fn barnes_hut_data<F: Fn(&&[T], &&[T]) -> T + Send + Sync>(&mut self, metric_f: F) {
        let samples: Vec<&[T]> = self.data.chunks(self.d).collect::<Vec<&[T]>>();
        let tsne_builder = TsneBuilder::new(&samples);

        let data = tsne_builder.data;
        self.n_samples = tsne_builder.data.len(); // Number of samples in data.

        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &self.n_samples);

        let embedding_dim = self.embedding_dim;
        // Number of  points ot consider when approximating the conditional distribution P.
        let n_neighbors: usize = (T::from(3.0).unwrap() * self.perplexity).as_();
        // NUmber of entries in gradient and gains matrices.
        let grad_entries = self.n_samples * embedding_dim;
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
            self.n_samples,
            &n_neighbors,
        );

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Prepares buffers for Barnes-Hut algorithm.
        self.positive_forces = vec![T::zero().into(); grad_entries];
        self.negative_forces = vec![T::zero().into(); grad_entries];
        self.forces_buffer = vec![T::zero().into(); grad_entries];
        self.q_sums = vec![T::zero().into(); self.n_samples];

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        self.means = vec![T::zero(); embedding_dim];
        self.epochs = 0;
    }

    // Main Training loop.
    pub fn run(&mut self, epochs: usize) {
        let embedding_dim = self.embedding_dim;

        for _epoch in 0..epochs {
            self.epochs += 1;
            {
                // Construct space partitioning tree on current embedding.
                let tree = tsne::SPTree::new(&embedding_dim, &self.y, &self.n_samples);
                // Check if the SPTree is correct.
                debug_assert!(tree.is_correct(), "error: SPTree is not correct.");

                // Computes forces using the Barnes-Hut algorithm in parallel.
                // Each chunk of positive_forces and negative_forces is associated to a distinct
                // embedded sample in y. As a consequence of this the computation can be done in
                // parallel.
                self.positive_forces
                    .par_chunks_mut(embedding_dim)
                    .zip(self.negative_forces.par_chunks_mut(embedding_dim))
                    .zip(self.forces_buffer.par_chunks_mut(embedding_dim))
                    .zip(self.q_sums.par_iter_mut())
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
            let q_sum: T = self.q_sums.par_iter_mut().map(|sum| sum.0).sum();
            self.dy
                .par_iter_mut()
                .zip(self.positive_forces.par_iter_mut())
                .zip(self.negative_forces.par_iter_mut())
                .for_each(|((grad, pf), nf)| {
                    grad.0 = pf.0 - (nf.0 / q_sum);
                    pf.0 = T::zero();
                    nf.0 = T::zero();
                });
            // Zeroes Q-sums.
            self.q_sums.par_iter_mut().for_each(|sum| sum.0 = T::zero());

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
            tsne::zero_mean(
                &mut self.means,
                &mut self.y,
                &self.n_samples,
                &self.embedding_dim,
            );

            // Stop lying about the P-values if the time is right.
            if self.epochs == self.stop_lying_epoch {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if self.epochs == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }
        }
    }

    pub fn embeddings(&mut self) -> Vec<T> {
        self.y.iter().map(|x| x.0).collect()
    }
}
