use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters<T>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
{
    /// Sets a new learning rate.
    ///
    /// # Arguments
    ///
    /// `learning_rate` - new value for the learning rate.
    #[serde(default = "Hyperparameters::<T>::learning_rate")]
    pub learning_rate: T,
    /// Sets a new momentum.
    ///
    /// # Arguments
    ///
    /// `momentum` - new value for the momentum.
    #[serde(default = "Hyperparameters::<T>::momentum")]
    pub momentum: T,
    /// Sets a new final momentum.
    ///
    /// # Arguments
    ///
    /// `final_momentum` - new value for the final momentum.
    #[serde(default = "Hyperparameters::<T>::final_momentum")]
    pub final_momentum: T,
    /// Sets a new momentum switch epoch, i.e. the epoch after which the algorithm switches to
    /// `final_momentum` for the map update.
    ///
    /// # Arguments
    ///
    /// `momentum_switch_epoch` - new value for the momentum switch epoch.
    #[serde(default = "Hyperparameters::<T>::momentum_switch_epoch")]
    pub momentum_switch_epoch: usize,
    /// Sets a new stop lying epoch, i.e. the epoch after which the P distribution values become
    /// true, as defined in the original implementation. For epochs < `stop_lying_epoch` the values
    /// of the P distribution are multiplied by a factor equal to `12.0`.
    ///
    /// # Arguments
    ///
    /// `stop_lying_epoch` - new value for the stop lying epoch.
    #[serde(default = "Hyperparameters::<T>::stop_lying_epoch")]
    pub stop_lying_epoch: usize,
    /// Sets a new theta, which determines the accuracy of the approximation. Must be **strictly greater than
    /// 0.0**. Large values for θ increase the speed of the algorithm but decrease its accuracy.
    /// For small values of θ it is less probable that a cell in the space partitioning tree will
    /// be treated as a single point. For θ equal to 0.0 the method degenerates in the exact
    /// version.
    ///
    /// # Arguments
    ///
    /// * `theta`  - new value for the theta.
    #[serde(default = "Hyperparameters::<T>::theta")]
    pub theta: T,
    /// Sets a new value for the embedding dimension.
    ///
    /// # Arguments
    ///
    /// `embedding_dim` - new value for the embedding space dimensionality.
    #[serde(default = "Hyperparameters::<T>::embedding_dim")]
    pub embedding_dim: usize,
    /// Sets a new perplexity value.
    ///
    /// # Arguments
    ///
    /// `perplexity` - new value for the perplexity. It's used so that the bandwidth of the Gaussian
    ///  kernels, is set in such a way that the perplexity of each the conditional distribution *Pi*
    ///  equals a predefined perplexity *u*.
    ///
    /// A good value for perplexity lies between 5.0 and 50.0.
    #[serde(default = "Hyperparameters::<T>::perplexity")]
    pub perplexity: T,
}

impl<T> Hyperparameters<T>
where
    T: Float + Send + Sync + Sum + DivAssign + AddAssign + MulAssign + SubAssign,
{
    fn learning_rate() -> T {
        T::from(200.0).unwrap()
    }

    fn momentum() -> T {
        T::from(0.5).unwrap()
    }

    fn final_momentum() -> T {
        T::from(0.8).unwrap()
    }

    fn momentum_switch_epoch() -> usize {
        250
    }

    fn stop_lying_epoch() -> usize {
        250
    }

    fn theta() -> T {
        T::from(0.5).unwrap()
    }

    fn embedding_dim() -> usize {
        2
    }

    fn perplexity() -> T {
        T::from(20.0).unwrap()
    }
}
