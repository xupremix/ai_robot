use std::marker::PhantomData;

use crate::gym::Gym;

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

impl<T, S, M, G> MlRobot<T, S, M, bool, G> {
    pub fn get_log(&self) -> bool {
        self.log
    }
}
