use crate::gym::Gym;
use crate::robot::{Eval, FieldSet, MlRobot};
use robotics_lib::event::events::Event;
use tch::nn::Module;
use tch::CModule;
use tch::Kind::Float;

impl<S> MlRobot<Eval, FieldSet<S>, CModule, bool, Gym> {
    pub fn step(&mut self) -> (bool, Vec<Event>) {
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
        let done = self.gym.state.borrow().done;
        let events = self.gym.state.borrow().events.clone();
        self.gym.state.borrow_mut().events = vec![];
        (done, events)
    }
    pub fn reset(&mut self) {
        if self.log {
            eprintln!("Resetting the environment");
        }
        self.gym.reset();
    }
}
