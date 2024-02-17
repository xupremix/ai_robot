use tch::{Device, Tensor};
use tch::nn::{Adam, linear, Optimizer, OptimizerConfig, seq, Sequential, VarStore};

use crate::prelude::{ACTION_SPACE, CRITIC_LR, OBSERVATION_SPACE};

pub struct Critic {
    vs: VarStore,
    network: Sequential,
    device: Device,
    optimizer: Optimizer,
}

impl Critic {
    pub(crate) fn new() -> Self {
        let device = Device::cuda_if_available();
        let vs = VarStore::new(device.clone());
        let optimizer = Adam::default().build(&vs, CRITIC_LR).unwrap();
        let p = &vs.root();
        let network = seq().add(linear(
            p / "in",
            OBSERVATION_SPACE + ACTION_SPACE,
            1,
            Default::default(),
        ));
        Self {
            network,
            device,
            vs,
            optimizer,
        }
    }

    pub fn forward(&self, obs: &Tensor, actions: &Tensor) -> Tensor {
        let xs = Tensor::cat(&[actions.copy(), obs.copy()], 1);
        xs.to_device(self.device).apply(&self.network)
    }

    pub fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }
    pub fn var_store(&self) -> &VarStore {
        &self.vs
    }
    pub fn var_store_mut(&mut self) -> &mut VarStore {
        &mut self.vs
    }
    pub fn import(&mut self, other: &Self) {
        self.vs.copy(&other.vs).unwrap();
    }
}
