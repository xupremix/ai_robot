use ai_robot::prelude::MlRobot;

fn main() {
    let mut robot = MlRobot::new()
        .set_log("src/save/logs/data.log")
        .set_map("src/save/maps/coin_bank_adj_map.bin", 40, 40)
        .load_model("src/save/models/model.pt")
        .build();
    robot.reset();
    let mut done = false;
    while !done {
        let new_done = robot.step();
        done = new_done;
    }
}
