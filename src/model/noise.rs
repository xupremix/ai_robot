use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::{Cuda, Device, Kind, Tensor};

use crate::prelude::{ACTION_SPACE, MU, SIGMA, THETA};

pub struct Noise {
    mode: (Kind, Device),
    state: Tensor,
    theta: f64,
    sigma: f64,
    mu: f64,
}

impl Noise {
    pub(crate) fn new() -> Self {
        let mode = if Cuda::is_available() {
            FLOAT_CUDA
        } else {
            FLOAT_CPU
        };
        let state = Tensor::ones([ACTION_SPACE], mode);
        Self {
            mode,
            state,
            theta: THETA,
            sigma: SIGMA,
            mu: MU,
        }
    }

    pub fn sample(&mut self) -> &Tensor {
        let dx = self.theta * (self.mu - &self.state)
            + self.sigma * Tensor::randn(self.state.size(), self.mode);
        self.state += dx;
        &self.state
    }
}
