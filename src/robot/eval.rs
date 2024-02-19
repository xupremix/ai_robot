use robotics_lib::event::events::Event;
use robotics_lib::world::tile::Tile;
use tch::nn::Module;
use tch::CModule;
use tch::Kind::Float;

use crate::gym::Gym;
use crate::robot::{Eval, FieldSet, MlRobot};

impl<S> MlRobot<Eval, FieldSet<S>, CModule, bool, Gym> {
    pub fn step(&mut self) -> (bool, Vec<Event>, Vec<(Tile, (usize, usize))>) {
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
        let new_tiles = self.gym.state.borrow().new_tiles.clone();
        self.gym.state.borrow_mut().events = vec![];
        self.gym.state.borrow_mut().new_tiles = vec![];
        (done, events, new_tiles)
    }
    pub fn reset(&mut self) {
        if self.log {
            eprintln!("Resetting the environment");
        }
        self.gym.reset();
    }
}
