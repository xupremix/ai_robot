use robotics_lib::world::tile::Content;

// Re-Exports
pub use crate::model::agent::Agent;
pub use crate::robot::{Eval, FieldNotSet, FieldSet, Init, MlRobot, Train};

pub const ACTOR_LR: f64 = 0.001;
pub const CRITIC_LR: f64 = 0.002;
pub const OBSERVATION_SPACE: i64 = 4;
pub const ACTION_SPACE: i64 = 4;
pub const DEFAULT_MAP_PATH: &'static str = "src/save/maps/normal_map.bin";
// noise constants
pub const MU: f64 = 0.0;
pub const THETA: f64 = 0.15;
pub const SIGMA: f64 = 0.1;
// The impact of the q value of the next state on the current state's q value.
pub const GAMMA: f64 = 0.99;
// The weight for updating the target networks.
pub const TAU: f64 = 0.005;
pub const MEM_DIM: i64 = 100_000;
pub const REWARD: f64 = -2.0;
pub const REWARD_FOR_ILLEGAL_ACTION: f64 = -10.0;
pub const CONTENT_TARGETS: [Content; 2] = [Content::Bank(0..0), Content::Coin(0)];
pub const PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING: f64 = 0.04;

