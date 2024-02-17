use ai_robot::prelude::MlRobot;
fn main() {
    let mut robot = MlRobot::new()
        .set_log("src/save/logs/data.log")
        .set_map("src/save/maps/DL_D_DD_D_D_DD_L_DLD_D_P_LLLL.bin", 6, 5)
        .set_model("src/save/models/model.pt")
        .build();
    robot.train_while_discovering(10, 100, 30);
}
