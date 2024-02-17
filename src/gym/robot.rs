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
    already_scanned: [bool; 4],
    scan_idx: usize,
    log: bool,
}

impl GymRobot {
    pub fn new(
        log: bool,
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
            already_scanned: [false; 4],
            scan_idx: 0,
            log,
        }
    }
    fn update_scan(&mut self, world: &mut World) {
        let surroundings = robot_view(self, world);
        let robot_i = self.get_coordinate().get_row();
        let robot_j = self.get_coordinate().get_col();
        for (i, row) in surroundings.iter().enumerate() {
            for (j, tile) in row.iter().enumerate() {
                if let Some(tile) = tile {
                    self.state
                        .borrow_mut()
                        .new_tiles
                        .push((tile.clone(), (robot_i + i - 1, robot_j + j - 1)));
                }
            }
        }
    }
    fn step(&mut self, world: &mut World) {
        let action = self.state.borrow().action;
        let reward = if self.state.borrow().dir[action as usize] == 1.0 {
            REWARD
        } else {
            REWARD_FOR_ILLEGAL_ACTION
        };
        if self
            .state
            .borrow()
            .dir
            .iter()
            .position(|e| *e == 1.0)
            .is_none()
            && thread_rng().gen_bool(0.9)
        {
            self.manual_action(world);
            return;
        }
        let dir = match action {
            0 => Direction::Up,
            1 => Direction::Right,
            2 => Direction::Down,
            _ => Direction::Left,
        };
        if self.log {
            eprintln!("Moving: {:?}", dir);
        }
        if let Ok((surr, (robot_i, robot_j))) = go(self, world, dir.clone()) {
            match dir {
                Direction::Up => {
                    for j in 0..3 {
                        if let Some(tile) = &surr[0][j] {
                            self.state
                                .borrow_mut()
                                .new_tiles
                                .push((tile.clone(), (robot_i - 1, robot_j + j - 1)))
                        }
                    }
                }
                Direction::Down => {
                    for j in 0..3 {
                        if let Some(tile) = &surr[2][j] {
                            self.state
                                .borrow_mut()
                                .new_tiles
                                .push((tile.clone(), (robot_i + 1, robot_j + j - 1)))
                        }
                    }
                }
                Direction::Left => {
                    for i in 0..3 {
                        if let Some(tile) = &surr[i][0] {
                            self.state
                                .borrow_mut()
                                .new_tiles
                                .push((tile.clone(), (robot_i + i - 1, robot_j - 1)))
                        }
                    }
                }
                Direction::Right => {
                    for i in 0..3 {
                        if let Some(tile) = &surr[i][2] {
                            self.state
                                .borrow_mut()
                                .new_tiles
                                .push((tile.clone(), (robot_i + i - 1, robot_j + 1)))
                        }
                    }
                }
            }
        }
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
                            if self.log {
                                eprintln!("Destroy: {direction:?}");
                            }
                            let _ = destroy(self, world, direction.clone());
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
                            if self.log {
                                eprintln!("Put: {direction:?}");
                            }
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
        // if we have no info and no adj
        if self
            .state
            .borrow()
            .dir
            .iter()
            .position(|e| *e == 1.0)
            .is_none()
        {
            if thread_rng().gen_bool(0.1) && !self.normal_scan {
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
                if self.log {
                    eprintln!("Special Scan");
                }
                if let Err(_) = scanner.scan(world, self, pattern, Content::Coin(0)) {
                    self.normal_scan = true;
                }
            } else {
                let distance = (self.get_energy().get_energy_level() as f64 / 3.
                    * PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING)
                    .floor() as usize;
                if distance > 2 {
                    let dir = match self.scan_idx {
                        0 => Direction::Up,
                        1 => Direction::Right,
                        2 => Direction::Down,
                        _ => Direction::Left,
                    };
                    if self.scan_idx == 3 {
                        self.already_scanned = [false; 4];
                    }
                    self.scan_idx = (self.scan_idx + 1) % 4;
                    self.already_scanned[self.scan_idx] = true;
                    if self.log {
                        eprintln!("Scanning: {dir:?}, {distance}");
                    }
                    let robot_i = self.get_coordinate().get_row();
                    let robot_j = self.get_coordinate().get_col();
                    if let Ok(surr) = one_direction_view(self, world, dir.clone(), distance) {
                        for (i, row) in surr.iter().enumerate() {
                            for (j, tile) in row.iter().enumerate() {
                                let coord = match dir {
                                    Direction::Up => (robot_i - i - 1, robot_j + j - 1),
                                    Direction::Down => (robot_i + i + 1, robot_j + j - 1),
                                    Direction::Left => (robot_i + i - 1, robot_j - j - 1),
                                    Direction::Right => (robot_i + i - 1, robot_j + j + 1),
                                };
                                self.state
                                    .borrow_mut()
                                    .new_tiles
                                    .push((tile.clone(), coord));
                            }
                        }
                    }
                } else if self.log {
                    eprintln!("Not enough energy for a scan, waiting for a recharge");
                }
            }
        } else if self.log {
            // we have info but no adj
            eprintln!("We already have info, no need to scan");
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
        // get the closest coin which is not adjacent
        if let Some((i, j)) = closest_coin {
            if (i != robot_i || j != robot_j)
                && *self
                    .get_backpack()
                    .get_contents()
                    .get(&Content::Coin(0))
                    .unwrap()
                    < 20
            {
                update_idx(i, robot_i, j, robot_j);
            }
        } else if let Some((i, j)) = closest_bank {
            if *self
                .get_backpack()
                .get_contents()
                .get(&Content::Coin(0))
                .unwrap()
                > 0
            {
                if i != robot_i || j != robot_j {
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
        if self.setup {
            self.update_scan(world);
            self.setup = false;
        } else if self.turn {
            self.step(world);
            self.turn = false;
        } else {
            // manual action
            self.manual_action(world);
            if self.coins_stored >= self.coins_stored_goal {
                self.state.borrow_mut().done = true;
            } else if self.coins_destroyed >= self.coins_destroyed_goal {
                self.state.borrow_mut().done = true;
            }
            self.turn = true;
        }
        self.update_dir(world);
    }

    fn handle_event(&mut self, event: Event) {
        self.state.borrow_mut().events.push(event);
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
