use ai_robot::prelude::MlRobot;
fn main() {
    let mut robot = MlRobot::new()
        .set_log(true)
        .set_map("src/save/maps/testing.bin", 6, 5)
        .set_model("src/save/models/model.pt")
        .build();
    robot.train_while_discovering(10, 100, 30);
}
