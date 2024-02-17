use crate::gym::Gym;
use crate::prelude::Agent;
use crate::robot::{FieldSet, MlRobot, Train};
use tch::Kind::Float;

impl<S, L> MlRobot<Train, FieldSet<S>, Agent, FieldSet<L>, Gym> {
    pub fn save(&mut self) {
        self.model.save();
    }
    pub fn train(
        &mut self,
        epochs: usize,
        max_ep_len: usize,
        batch_size: usize,
        train_iterations: usize,
    ) {
        let mut best_rw = f64::MIN;
        for _ in 0..epochs {
            self.gym.reset();
            for _ in 0..max_ep_len {
                let obs = self.gym.state.borrow().build();
                let actions = self.model.actions(&obs);
                let action = actions.softmax(-1, Float).argmax(-1, true).int64_value(&[]);
                self.gym.step(action);
                self.model.remember(
                    &obs,
                    &actions,
                    &self.gym.state.borrow().reward.into(),
                    &self.gym.state.borrow().build(),
                );
                if self.gym.state.borrow().done {
                    break;
                }
            }
            for _ in 0..train_iterations {
                self.model.train(batch_size);
            }
            let mut acc_rw = 0.0;
            self.gym.reset();
            for _ in 0..max_ep_len {
                let obs = self.gym.state.borrow().build();
                let actions = self.model.actions(&obs);
                let action = actions.softmax(-1, Float).argmax(-1, true).int64_value(&[]);
                self.gym.step(action);
                let reward = self.gym.state.borrow().reward;
                acc_rw += reward;
                if self.gym.state.borrow().done {
                    break;
                }
            }
            if acc_rw > best_rw {
                best_rw = acc_rw;
                eprintln!("Found a better model");
                self.model.save();
            }
        }
    }
    pub fn train_while_discovering(&mut self, epochs: usize, max_ep_len: usize, batch_size: usize) {
        let mut best_rw = f64::MIN;
        for _ in 0..epochs {
            let mut discover = batch_size;
            let mut acc_rw = 0.0;
            self.gym.reset();
            for i in 0..max_ep_len {
                let obs = self.gym.state.borrow().build();
                let actions = self.model.actions(&obs);
                let action = actions.softmax(-1, Float).argmax(-1, true).int64_value(&[]);
                self.gym.step(action);
                let reward = self.gym.state.borrow().reward;
                acc_rw += reward;
                self.model.remember(
                    &obs,
                    &actions,
                    &reward.into(),
                    &self.gym.state.borrow().build(),
                );
                if self.gym.state.borrow().done {
                    eprintln!("The robot completed the task");
                    break;
                }
                if i > discover {
                    self.model.train(batch_size);
                    discover += batch_size;
                }
            }
            if acc_rw > best_rw {
                best_rw = acc_rw;
                eprintln!("Found a better model");
                self.model.save();
            }
        }
    }
}
