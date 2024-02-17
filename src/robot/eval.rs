use crate::gym::Gym;
use crate::robot::{Eval, FieldSet, MlRobot};
use tch::nn::Module;
use tch::CModule;
use tch::Kind::Float;

impl<S> MlRobot<Eval, FieldSet<S>, CModule, bool, Gym> {
    pub fn step(&mut self) -> bool {
        let obs = self.gym.state.borrow().build();
        let action = self
            .model
            .forward(&obs)
            .softmax(-1, Float)
            .argmax(-1, false)
            .int64_value(&[]);
        if self.log {
            eprintln!("---------------");
            eprintln!("Step:");
            eprintln!("Dir: {:?}", obs);
        }
        self.gym.step(action);
        self.gym.state.borrow().done
    }
    pub fn reset(&mut self) {
        if self.log {
            eprintln!("Resetting the environment");
        }
        self.gym.reset();
    }
}
