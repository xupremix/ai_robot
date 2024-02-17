use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::nn::{linear, seq, Adam, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{CModule, Cuda, Device, Tensor};

use crate::prelude::{ACTION_SPACE, ACTOR_LR, OBSERVATION_SPACE};

pub struct Actor {
    save_path: String,
    vs: VarStore,
    network: Sequential,
    device: Device,
    optimizer: Optimizer,
}

impl Actor {
    pub(crate) fn new(save_path: String) -> Self {
        let device = Device::cuda_if_available();
        let vs = VarStore::new(device.clone());
        let optimizer = Adam::default().build(&vs, ACTOR_LR).unwrap();
        let p = &vs.root();
        let network = seq().add(linear(
            p / "in",
            OBSERVATION_SPACE,
            ACTION_SPACE,
            Default::default(),
        ));
        Self {
            save_path,
            device,
            network,
            vs,
            optimizer,
        }
    }

    pub fn forward(&self, obs: &Tensor) -> Tensor {
        obs.to_device(self.device).apply(&self.network)
    }

    pub fn save(&mut self) {
        self.vs.freeze();
        let mut forward_fn = |x: &[Tensor]| vec![self.forward(&x[0])];
        let mode = if Cuda::is_available() {
            FLOAT_CUDA
        } else {
            FLOAT_CPU
        };
        // trace the module with a dummy input
        let cmodule = CModule::create_by_tracing(
            "Model",
            "forward",
            &[Tensor::zeros([OBSERVATION_SPACE], mode)],
            &mut forward_fn,
        )
        .unwrap();
        // save the module
        cmodule.save(&self.save_path).unwrap();
        self.vs.unfreeze();
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
