use crate::gym::Gym;
use crate::robot::{Eval, FieldSet, MlRobot};
use tch::nn::Module;
use tch::CModule;
use tch::Kind::Float;

impl<S, L> MlRobot<Eval, FieldSet<S>, CModule, FieldSet<L>, Gym> {
    // TODO Switch to return the new tiles
    pub fn step(&mut self) -> bool {
        let action = self
            .model
            .forward(&self.gym.state.borrow().build())
            .softmax(-1, Float)
            .argmax(-1, false)
            .int64_value(&[]);
        self.gym.step(action);
        self.gym.state.borrow().done
    }
    pub fn reset(&mut self) {
        self.gym.reset();
    }
}
