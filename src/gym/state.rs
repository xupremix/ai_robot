use tch::{Device, Kind, Tensor};

pub struct State {
    pub action: i64,
    pub reward: f64,
    pub done: bool,
    pub dir: [f64; 4],
}

impl Default for State {
    fn default() -> Self {
        Self {
            action: -1,
            reward: 0.0,
            done: false,
            dir: [0.0; 4],
        }
    }
}

impl State {
    pub(crate) fn build(&self) -> Tensor {
        Tensor::from_slice(&self.dir)
            .to_kind(Kind::Float)
            .to(Device::cuda_if_available())
    }
}
