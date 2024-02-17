use ai_robot::prelude::MlRobot;
use std::thread::sleep;
use std::time::Duration;

fn main() {
    let mut robot = MlRobot::new()
        .set_log(true)
        .set_map("src/save/maps/DL_D_DD_D_D_DD_L_DLD_D_P_LLLL.bin", 6, 5)
        .load_model("src/save/models/exam_ready_model.pt")
        .build();
    let mut done = false;
    robot.reset();
    while !done {
        let new_done = robot.step();
        sleep(Duration::from_millis(1000));
        done = new_done.0;
    }
}
