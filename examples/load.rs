use ai_robot::prelude::MlRobot;

fn main() {
    let mut robot = MlRobot::new()
        .set_log(true)
        .set_map("src/save/maps/testing.bin", 6, 5)
        .load_model("src/save/models/exam_ready_model.pt")
        .build();
    let mut done = false;
    robot.reset();
    while !done {
        let step = robot.step();
        done = step.0;
    }
}
