use crate::gym::robot::GymRobot;
use crate::gym::state::State;
use robotics_lib::runner::Runner;
use std::cell::RefCell;
use std::rc::Rc;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

mod robot;
pub mod state;

pub struct Gym {
    pub(crate) state: Rc<RefCell<State>>,
    generator: WorldgeneratorUnwrap,
    runner: Runner,
    coins_destroyed_goal: usize,
    coins_stored_goal: usize,
}

impl Gym {
    pub fn new(
        mut generator: WorldgeneratorUnwrap,
        coins_destroyed_goal: usize,
        coins_stored_goal: usize,
    ) -> Self {
        let state = Rc::new(RefCell::new(State::default()));
        let mut runner = Runner::new(
            Box::new(GymRobot::new(
                state.clone(),
                coins_destroyed_goal,
                coins_stored_goal,
            )),
            &mut generator,
        )
        .unwrap();
        runner.game_tick().unwrap();
        Self {
            state,
            generator,
            runner,
            coins_destroyed_goal,
            coins_stored_goal,
        }
    }
    pub(crate) fn reset(&mut self) {
        *self.state.borrow_mut() = Default::default();
        self.runner = Runner::new(
            Box::new(GymRobot::new(
                self.state.clone(),
                self.coins_destroyed_goal,
                self.coins_stored_goal,
            )),
            &mut self.generator,
        )
        .unwrap();
        self.runner.game_tick().unwrap();
    }
    pub(crate) fn step(&mut self, action: i64) {
        // set the action that the robot needs to take
        self.state.borrow_mut().action = action;
        // perform a game loop
        self.runner.game_tick().unwrap();
    }
}