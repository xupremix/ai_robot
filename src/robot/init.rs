use crate::gym::Gym;
use crate::model::actor::Actor;
use crate::model::critic::Critic;
use crate::model::noise::Noise;
use crate::prelude::{Agent, DEFAULT_MAP_PATH};
use crate::robot::{Eval, FieldNotSet, FieldSet, Init, MlRobot, Train};
use std::marker::PhantomData;
use std::path::PathBuf;
use tch::{CModule, Device};
use worldgen_unwrap::public::WorldgeneratorUnwrap;

impl<M, L> MlRobot<Init, FieldNotSet, M, L, FieldNotSet> {
    pub fn gen_map(
        self,
        coins_destroyed_goal: usize,
        coins_stored_goal: usize,
    ) -> MlRobot<Init, FieldSet<String>, M, L, Gym> {
        let generator = WorldgeneratorUnwrap::init(true, None);
        let gym = Gym::new(generator, coins_destroyed_goal, coins_stored_goal);
        MlRobot {
            _state: PhantomData,
            map: FieldSet {
                data: DEFAULT_MAP_PATH.to_string(),
            },
            model: self.model,
            log: self.log,
            gym,
        }
    }
    pub fn set_map<S>(
        self,
        map: S,
        coins_destroyed_goal: usize,
        coins_stored_goal: usize,
    ) -> MlRobot<Init, FieldSet<S>, M, L, Gym>
    where
        S: Into<PathBuf> + Copy,
    {
        let generator = WorldgeneratorUnwrap::init(false, Some(map.into()));
        let gym = Gym::new(generator, coins_destroyed_goal, coins_stored_goal);
        MlRobot {
            _state: PhantomData,
            map: FieldSet { data: map },
            model: self.model,
            log: self.log,
            gym,
        }
    }
}

impl<S, L, G> MlRobot<Init, S, FieldNotSet, L, G> {
    pub fn set_model<P>(self, save_path: P) -> MlRobot<Init, S, Agent, L, G>
    where
        P: Into<String> + Copy,
    {
        let actor = Actor::new(save_path.into());
        let mut actor_target = Actor::new(save_path.into());
        actor_target.import(&actor);
        let critic = Critic::new();
        let mut critic_target = Critic::new();
        critic_target.import(&critic);
        let noise = Noise::new();
        let model = Agent::new(actor, actor_target, critic, critic_target, noise, true);
        MlRobot {
            _state: PhantomData,
            map: self.map,
            model,
            log: self.log,
            gym: self.gym,
        }
    }
    pub fn load_model<P>(self, model_path: P) -> MlRobot<Init, S, CModule, L, G>
    where
        P: AsRef<std::path::Path>,
    {
        let model = CModule::load_on_device(model_path, Device::cuda_if_available()).unwrap();
        MlRobot {
            _state: PhantomData,
            map: self.map,
            model,
            log: self.log,
            gym: self.gym,
        }
    }
}

impl<S, M, G> MlRobot<Init, S, M, FieldNotSet, G> {
    pub fn set_log<L>(self, log_path: L) -> MlRobot<Init, S, M, FieldSet<L>, G>
    where
        L: Into<String> + Copy,
    {
        MlRobot {
            _state: PhantomData,
            map: self.map,
            model: self.model,
            log: FieldSet { data: log_path },
            gym: self.gym,
        }
    }
}

impl<S, L> MlRobot<Init, FieldSet<S>, Agent, FieldSet<L>, Gym> {
    pub fn build(self) -> MlRobot<Train, FieldSet<S>, Agent, FieldSet<L>, Gym> {
        MlRobot {
            _state: PhantomData,
            map: self.map,
            model: self.model,
            log: self.log,
            gym: self.gym,
        }
    }
}
impl<S, L> MlRobot<Init, FieldSet<S>, CModule, FieldSet<L>, Gym> {
    pub fn build(self) -> MlRobot<Eval, FieldSet<S>, CModule, FieldSet<L>, Gym> {
        MlRobot {
            _state: PhantomData,
            map: self.map,
            model: self.model,
            log: self.log,
            gym: self.gym,
        }
    }
}
