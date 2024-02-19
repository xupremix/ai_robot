use ai_robot::prelude::MlRobot;

fn main() {
    let mut robot = MlRobot::new()
        .set_log(false)
        .gen_map("src/save/maps/exam_map.bin", 40, 40)
        .load_model("src/save/models/exam_ready_model.pt")
        .build();
    let mut done = false;
    robot.reset();
    while !done {
        let step = robot.step();
        done = step.0;
    }
}
