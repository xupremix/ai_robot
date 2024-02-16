use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::rc::Rc;

use another_one_bytes_the_dust_resource_scanner_tool::tool::resource_scanner::{
    Pattern, ResourceScanner,
};
use ghost_journey_journal::JourneyJournal;
use rand::{thread_rng, Rng};
use robotics_lib::energy::Energy;
use robotics_lib::event::events::Event;
use robotics_lib::interface::{destroy, go, one_direction_view, put, robot_view, Direction};
use robotics_lib::runner::backpack::BackPack;
use robotics_lib::runner::{Robot, Runnable};
use robotics_lib::world::coordinates::Coordinate;
use robotics_lib::world::tile::Content;
use robotics_lib::world::World;

use crate::gym::state::State;
use crate::prelude::{
    CONTENT_TARGETS, PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING, REWARD, REWARD_FOR_ILLEGAL_ACTION,
};

pub(crate) struct GymRobot {
    state: Rc<RefCell<State>>,
    robot: Robot,
    coins_destroyed: usize,
    coins_stored: usize,
    coins_destroyed_goal: usize,
    coins_stored_goal: usize,
    normal_scan: bool,
    setup: bool,
    turn: bool,
}

impl GymRobot {
    pub fn new(
        state: Rc<RefCell<State>>,
        coins_destroyed_goal: usize,
        coins_stored_goal: usize,
    ) -> Self {
        Self {
            robot: Robot::new(),
            coins_destroyed: 0,
            coins_stored: 0,
            coins_destroyed_goal,
            coins_stored_goal,
            state,
            setup: true,
            turn: false,
            normal_scan: false,
        }
    }
    fn step(&mut self, world: &mut World) {
        let action = self.state.borrow().action;
        println!("Moving: {action}");
        let _ = match action {
            0 => go(self, world, Direction::Up),
            1 => go(self, world, Direction::Right),
            2 => go(self, world, Direction::Down),
            _ => go(self, world, Direction::Left),
        };
        self.update_dir(world);
        let reward = if self.state.borrow().dir[action as usize] == 1.0 {
            REWARD
        } else {
            REWARD_FOR_ILLEGAL_ACTION
        };
        self.state.borrow_mut().reward = reward;
    }
    fn manual_action(&mut self, world: &mut World) {
        let surr = robot_view(self, world);
        for (i, row) in surr.iter().enumerate() {
            for (j, tile) in row.iter().enumerate() {
                // even manhattan distance -> diagonal or under the robot
                if ((i as i32 - 1).abs() + (j as i32 - 1).abs()) % 2 == 0 {
                    continue;
                }
                let direction = match (i, j) {
                    (0, 1) => Direction::Up,
                    (1, 2) => Direction::Right,
                    (2, 1) => Direction::Down,
                    (1, 0) => Direction::Left,
                    _ => panic!("Impossible"),
                };
                match tile {
                    None => {}
                    Some(tile) => match &tile.content {
                        Content::Coin(amt) => {
                            eprintln!("Destroying a coin: {direction:?}");
                            let _ = destroy(self, world, direction);
                            self.coins_destroyed += amt;
                            return;
                        }
                        Content::Bank(_) => {
                            let amt = *self
                                .get_backpack()
                                .get_contents()
                                .get(&Content::Coin(0).to_default())
                                .unwrap();
                            if amt == 0 {
                                continue;
                            }
                            eprintln!("Putting a coin: {direction:?}");
                            if let Ok(amt) = put(self, world, Content::Coin(0), amt, direction) {
                                self.coins_stored += amt;
                            }
                            return;
                        }
                        _ => {}
                    },
                }
            }
        }
        // we did not find any coins / banks adj -> perform a scan
        if thread_rng().gen_bool(0.2) && !self.normal_scan {
            let pattern = match thread_rng().gen_range(0..11) {
                0 => Pattern::DirectionUp(5),
                1 => Pattern::DirectionRight(5),
                2 => Pattern::DirectionDown(5),
                3 => Pattern::DirectionLeft(5),
                4 => Pattern::Area(4),
                5 => Pattern::DiagonalLowerLeft(5),
                6 => Pattern::DiagonalLowerRight(5),
                7 => Pattern::DiagonalUpperLeft(5),
                8 => Pattern::DiagonalUpperRight(5),
                9 => Pattern::StraightStar(5),
                _ => Pattern::DiagonalStar(5),
            };
            let mut scanner = ResourceScanner {};
            if let Err(_) = scanner.scan(world, self, pattern, Content::Coin(0)) {
                self.normal_scan = true;
            }
        } else {
            let distance = (self.get_energy().get_energy_level() as f64 / 3.
                * PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING)
                .floor() as usize;
            if distance > 2 {
                let dir = match thread_rng().gen_range(0..4) {
                    0 => Direction::Up,
                    1 => Direction::Right,
                    2 => Direction::Down,
                    _ => Direction::Left,
                };
                eprintln!("Performing a scan: {dir:?}");
                let _ = one_direction_view(self, world, dir, distance);
            } else {
                self.step(world);
            }
        }
    }
    fn update_dir(&mut self, world: &mut World) {
        let surroundings = robot_view(self, world);
        self.state.borrow_mut().dir = [0.0; 4];

        let mut journal = JourneyJournal::new(&[], &CONTENT_TARGETS, false);
        let robot_i = self.get_coordinate().get_row();
        let robot_j = self.get_coordinate().get_col();

        // get the closest coin and back
        let closest_coin: Option<(usize, usize)> = journal
            .contents_closest_coords(&Content::Coin(0), self, world)
            .unwrap();
        let closest_bank: Option<(usize, usize)> = journal
            .contents_closest_coords(&Content::Bank(0..0), self, world)
            .unwrap();

        // update the dir based on danger and where the closest target is
        let mut available_indexes: HashSet<usize> = HashSet::from([0, 1, 2, 3]);
        let mut found_adj = false;
        for (i, row) in surroundings.iter().enumerate() {
            for (j, tile) in row.iter().enumerate() {
                if ((i as i32 - 1).abs() + (j as i32 - 1).abs()) % 2 == 0 {
                    continue;
                }
                match tile {
                    None => {
                        let _ = available_indexes.remove(&to_idx(i, j));
                    }
                    Some(tile) => match &tile.content {
                        Content::Coin(_) => {
                            self.turn = false;
                            found_adj = true;
                        }
                        Content::Bank(range) => {
                            if *self
                                .get_backpack()
                                .get_contents()
                                .get(&Content::Coin(0))
                                .unwrap()
                                > 0
                                && !range.is_empty()
                            {
                                self.turn = false;
                                found_adj = true;
                            }
                        }
                        _ => {
                            // check if it's walkable
                            if !tile.tile_type.properties().walk() {
                                let _ = available_indexes.remove(&to_idx(i, j));
                            }
                        }
                    },
                }
            }
        }
        // now the available indexes only contains the directions that I can move to
        let update_idx =
            |i: usize, robot_i: usize, j: usize, robot_j: usize| match available_indexes
                .get(&to_dir_idx(i, robot_i, j, robot_j))
            {
                None => {}
                Some(&idx) => {
                    self.state.borrow_mut().dir[idx] = 1.0;
                }
            };
        if !found_adj {
            // get the closest coin which is not adjacent
            if let Some((i, j)) = closest_coin {
                update_idx(i, robot_i, j, robot_j);
            }
            if let Some((i, j)) = closest_bank {
                if *self
                    .get_backpack()
                    .get_contents()
                    .get(&Content::Coin(0))
                    .unwrap()
                    > 0
                {
                    update_idx(i, robot_i, j, robot_j);
                }
            }
        }
    }
}
fn to_dir_idx(i: usize, robot_i: usize, j: usize, robot_j: usize) -> usize {
    match (i.cmp(&robot_i), j.cmp(&robot_j)) {
        (Ordering::Less, _) => 0,
        (Ordering::Greater, _) => 2,
        (_, Ordering::Less) => 3,
        (_, Ordering::Greater) => 1,
        _ => panic!("Unreachable"),
    }
}
fn to_idx(i: usize, j: usize) -> usize {
    match (i, j) {
        (0, 1) => 0,
        (1, 2) => 1,
        (2, 1) => 2,
        (1, 0) => 3,
        _ => panic!("Unreachable"),
    }
}
impl Runnable for GymRobot {
    fn process_tick(&mut self, world: &mut World) {
        // set up the state if it's the first game tick
        if self.setup {
            self.update_dir(world);
            self.setup = false;
        } else if self.turn {
            self.step(world);
            self.turn = false;
        } else {
            // manual action
            self.manual_action(world);
            if self.coins_stored >= self.coins_stored_goal {
                eprintln!("Completed the coins stored task");
                self.state.borrow_mut().done = true;
            } else if self.coins_destroyed >= self.coins_destroyed_goal {
                eprintln!("Completed the coins destroyed task");
                self.state.borrow_mut().done = true;
            }
            self.turn = true;
            self.update_dir(world);
        }
    }

    fn handle_event(&mut self, _event: Event) {
        // TODO
    }

    fn get_energy(&self) -> &Energy {
        &self.robot.energy
    }

    fn get_energy_mut(&mut self) -> &mut Energy {
        &mut self.robot.energy
    }

    fn get_coordinate(&self) -> &Coordinate {
        &self.robot.coordinate
    }

    fn get_coordinate_mut(&mut self) -> &mut Coordinate {
        &mut self.robot.coordinate
    }

    fn get_backpack(&self) -> &BackPack {
        &self.robot.backpack
    }

    fn get_backpack_mut(&mut self) -> &mut BackPack {
        &mut self.robot.backpack
    }
}