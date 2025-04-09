import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
# from deploy.utils.logger import DeployLogger

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    idx2gym = [0, 1, 2, 3, 4, 5, 
                        12,
                        6, 7, 8, 9, 10, 11, 
                        13]
    idx2mjc = [0, 1, 2, 3, 4, 5,
                        7, 8, 9, 10, 11, 12,
                        6, 13]
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)[idx2gym]
        kds = np.array(config["kds"], dtype=np.float32)[idx2gym]

        default_angles = np.array(config["default_angles"], dtype=np.float32)[idx2gym]
        print("default_angles:", default_angles)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        show_log = config["show_log_plot"]
        show_log_time = config["show_plot_time"]
        # save_csv = config["save_plot_as_csv"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    print("Loading model from", xml_path)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    print("DOF names:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])

    # if show_log:
    #     dof_names = [m.joint(i).name for i in range(m.njnt)]
    #     dof_names = dof_names[1:]
    #     dof_pos_limits = np.array([m.jnt_range[i] for i in range(m.njnt)])
    #     dof_pos_limits = dof_pos_limits[1:, :]
    #     logger = DeployLogger(simulation_dt*control_decimation, dof_names)
    #     # logger = DeployLogger() 

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            # print("shape target_dof_pos:", target_dof_pos.shape)
            # print("shape d.qpos[7:]:", d.qpos[7:].shape)
            # print("shape kps:", kps.shape)
            # print("shape np.zeros_like(kds):", np.zeros_like(kds).shape)
            # print("shape d.qvel[6:]:", d.qvel[6:].shape)
            # print("shape kds:", kds.shape)
            tau = pd_control(target_dof_pos, d.qpos[7:][idx2gym], kps, np.zeros_like(kds), d.qvel[6:][idx2gym], kds)
            d.ctrl[:] = tau[idx2mjc]
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:][idx2gym]
                dqj = d.qvel[6:][idx2gym]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                # if show_log and (counter*simulation_dt) < show_log_time: 
                #     logger.log_joint_state(qj, action)
                # elif show_log and (counter*simulation_dt) == show_log_time:
                #     # 将policy_path取到export/之前的
                #     policy_path = policy_path.split("policies")[0]
                #     # 提取policy_path中的g1*到/之间的内容
                #     robot_name = "g1" + policy_path.split("g1")[1].split("/")[0]
                #     logger.plot_states("deploy_mujoco", 0, robot_name, dof_pos_limits, policy_path, action_scale)
                #     if save_csv:
                #         logger.save_csv("deploy_mujoco", 0, robot_name, dof_pos_limits, policy_path, action_scale)
                # if show_log:
                #     logger.record("time", counter*simulation_dt)
                #     for i in range(num_actions):
                #         dof_name = dof_names[i]
                #         # record 前边的字典由measure和action合起来
                #         logger.record(dof_name +"_measure", qj[i])
                #         logger.record(dof_name + "_action", action[i]*action_scale + default_angles[i])
                
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = default_angles
                
                # target_dof_pos = default_angles.copy()
                # print("Current dof pos:", d.qpos[7:])
                # print("Target dof pos:", target_dof_pos)
                # print different between target and current
                # print("Difference:", (target_dof_pos - d.qpos[7:])/np.pi*180)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # print("sim time up!")
            # if show_log:
            #     logger.save_to_csv()
