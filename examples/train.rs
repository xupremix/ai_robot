use ai_robot::prelude::MlRobot;
fn main() {
    let mut robot = MlRobot::new()
        .set_log("src/save/logs/data.log")
        .set_map("src/save/maps/coin_bank_adj_map.bin", 40, 40)
        .set_model("src/save/models/model.pt")
        .build();
    robot.train(1, 200, 30, 10);
    robot.save();
}
