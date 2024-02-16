use crate::gym::Gym;
use crate::prelude::Agent;
use std::marker::PhantomData;
use tch::CModule;

mod eval;
mod init;
mod train;

#[derive(Debug)]
pub struct FieldNotSet;
#[derive(Debug)]
pub struct FieldSet<T> {
    data: T,
}
#[derive(Debug)]
pub struct Init;
#[derive(Debug)]
pub struct Train;
#[derive(Debug)]
pub struct Eval;
#[derive(Debug)]
pub struct MlRobot<T, S, M, L, G> {
    _state: PhantomData<T>,
    map: S,
    model: M,
    log: L,
    gym: G,
}

impl MlRobot<Init, FieldNotSet, FieldNotSet, FieldNotSet, FieldNotSet> {
    pub fn new() -> MlRobot<Init, FieldNotSet, FieldNotSet, FieldNotSet, FieldNotSet> {
        MlRobot {
            _state: PhantomData,
            map: FieldNotSet,
            model: FieldNotSet,
            log: FieldNotSet,
            gym: FieldNotSet,
        }
    }
}

impl<T, S, M, L> MlRobot<T, FieldSet<S>, M, L, Gym> {
    pub fn get_map(&self) -> &S {
        &self.map.data
    }
}
impl<T, S, L, G> MlRobot<T, S, Agent, L, G> {
    pub fn get_model(&self) -> &Agent {
        &self.model
    }
}
impl<T, S, L, G> MlRobot<T, S, CModule, L, G> {
    pub fn get_model(&self) -> &CModule {
        &self.model
    }
}
impl<T, S, M, L, G> MlRobot<T, S, M, FieldSet<L>, G> {
    pub fn get_log(&self) -> &L {
        &self.log.data
    }
}
