
mod tsne;
mod utils;
// mod distance_functions;
// use distance_functions::DistanceFunction

use wasm_bindgen::prelude::*;


use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};
#[cfg(feature = "csv")]
use std::{error::Error, fs::File};
use wasm_bindgen::prelude::wasm_bindgen;

/// t-distributed stochastic neighbor embedding. Provides a parallel implementation of both the
/// exact version of the algorithm and the tree accelerated one leveraging space partitioning trees.
#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct tSNE
{
    data: Vec<f32>,
    d: usize,
    learning_rate: f32,
    epochs: usize,
    momentum: f32,
    final_momentum: f32,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    embedding_dim: u8,
    perplexity: f32,
    p_values: Vec<tsne::Aligned<f32>>,
    p_rows: Vec<usize>,
    p_columns: Vec<usize>,
    q_values: Vec<tsne::Aligned<f32>>,
    y: Vec<tsne::Aligned<f32>>,
    dy: Vec<tsne::Aligned<f32>>,
    uy: Vec<tsne::Aligned<f32>>,
    gains: Vec<tsne::Aligned<f32>>,
}
#[wasm_bindgen]
impl tSNE
{
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, d: usize) -> tSNE {
        utils::set_panic_hook();
        tSNE {
            data,
            d,
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
        }
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
    pub fn set_embedding_dim(&mut self, embedding_dim: u8) {
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
    pub fn embedding(&self) -> Vec<f32> {
        self.y.iter().map(|x| x.0).collect()
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
    pub fn exact(&mut self) {
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
            |sample_a: &&[f32], sample_b: &&[f32]| {
                sample_a
                    .iter()
                    .zip(sample_b.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            },
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
    }
}

// this is the stuff that I haven't implemented in wasm yet
impl tSNE {
    /// Writes the embedding to a csv file. If the embedding space dimensionality is either equal to
    /// 2 or 3 the resulting csv file will have some simple headers:
    ///
    /// * x, y for 2 dimensions.
    ///
    /// * x, y, z for 3 dimensions.
    ///
    /// # Arguments
    ///
    /// * `file_path` - path of the file to write the embedding to.
    ///
    /// # Errors
    ///
    /// Returns an error is something goes wrong during the I/O operations.
    #[cfg(feature = "csv")]
    pub fn write_csv<'a>(&'a mut self, path: &str) -> Result<&'a mut Self, Box<dyn Error>>
    {
        let mut writer = csv::Writer::from_path(path)?;

        // String-ify the embedding.
        let to_write = self
            .y
            .iter()
            .map(|el| el.0.to_string())
            .collect::<Vec<String>>();

        // Write headers.
        match self.embedding_dim {
            2 => writer.write_record(&["x", "y"])?,
            3 => writer.write_record(&["x", "y", "z"])?,
            _ => (), // Write no headers for embedding dimensions greater that 3.
        }
        // Write records.
        for record in to_write.chunks(self.embedding_dim as usize) {
            writer.write_record(record)?
        }
        // Final flush.
        writer.flush()?;
        // Everything went smooth.
        Ok(self)
    }

    /// Loads data from a csv file.
    ///
    /// # Arguments
    ///
    /// * `file_path` - path of the file to load the data from.
    ///
    /// * `has_headers` - whether the file has headers or not. if set to `true` the function will
    /// not parse the first line of the csv file.
    ///
    /// * `skip` - an optional slice that specifies a subset of the file columns that must not be
    /// parsed.
    ///
    /// * `f` - function that converts [`String`] into a data sample. It takes as an argument a single
    /// record field.
    ///
    /// # Errors
    ///
    /// Returns an error is something goes wrong during the I/O operations.
    #[cfg(feature = "csv")]
    pub fn load_csv<T, F: Fn(String) -> f32>(
        path: &str,
        has_headers: bool,
        skip: Option<&[usize]>,
        f: F,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut data: Vec<f32> = Vec::new();

        let file = File::open(path)?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(has_headers)
            .from_reader(file);

        match skip {
            Some(range) => {
                for result in reader.records() {
                    let record = result?;

                    (0..record.len())
                        .filter(|column| !range.contains(column))
                        .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
                }
            }
            None => {
                for result in reader.records() {
                    let record = result?;

                    (0..record.len())
                        .for_each(|field| data.push(f(record.get(field).unwrap().to_string())));
                }
            }
        }
        Ok(data)
    }
}

