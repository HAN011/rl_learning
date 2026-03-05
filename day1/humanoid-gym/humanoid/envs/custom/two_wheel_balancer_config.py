# SPDX-License-Identifier: BSD-3-Clause

from humanoid.envs.custom.humanoid_config import XBotLCfg, XBotLCfgPPO


class TwoWheelBalancerCfg(XBotLCfg):
    class env(XBotLCfg.env):
        num_active_dofs = 2
        num_passive_dofs = 0
        num_commands = 3

        frame_stack = 3
        c_frame_stack = 1

        obs_names = ["base_euler_xyz", "base_ang_vel", "dof_pos", "dof_vel", "base_lin_vel", "actions"]
        privileged_obs_names = ["base_euler_xyz", "base_ang_vel", "dof_pos", "dof_vel", "base_lin_vel", "actions"]

        num_envs = 4096
        episode_length_s = 8.0
        use_ref_actions = False

        max_tilt_deg = 45.0
        min_base_height = 0.035
        reset_angle_range = 0.20
        reset_wheel_vel = 4.0

    class safety(XBotLCfg.safety):
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(XBotLCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/two_wheel_balancer/urdf/two_wheel_balancer.urdf"
        name = "two_wheel_balancer"
        foot_names = ["wheel_L", "wheel_R"]
        knee_names = []

        terminate_after_contacts_on = ["base_link"]
        penalize_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(XBotLCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 1.5
        dynamic_friction = 1.4
        restitution = 0.0

    class noise(XBotLCfg.noise):
        add_noise = True
        noise_level = 0.4

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.15
            base_ang_vel = 0.08
            base_lin_vel = 0.08
            base_euler = 0.03

    class init_state(XBotLCfg.init_state):
        pos = [0.0, 0.0, 0.07]

        default_joint_angles = {
            "wheel_L_joint": 0.0,
            "wheel_R_joint": 0.0,
        }

    class control(XBotLCfg.control):
        stiffness = {"wheel": 0.0}
        damping = {"wheel": 0.0}
        action_scale = 25.0
        wheel_damping = 0.2
        decimation = 5

    class sim(XBotLCfg.sim):
        dt = 0.002
        substeps = 1
        up_axis = 1

        class physx(XBotLCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 2
            contact_offset = 0.005
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class domain_rand(XBotLCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.8, 1.6]

        randomize_restitution = False
        restitution_range = [0.0, 0.2]

        randomize_base_mass = True
        added_mass_range = [-0.2, 0.2]

        randomize_com_displacement = True
        com_displacement_range = [-0.01, 0.01]

        randomize_joint_friction = False
        randomize_joint_armature = False
        randomize_joint_pos_bias = False
        randomize_joint_kp = False
        randomize_joint_kd = False
        randomize_base_euler_bias = False

        push_robots = False
        push_interval_s = 4
        update_step = 2000 * 8
        push_duration = [0.0]
        max_push_vel_xy = 0.15
        max_push_ang_vel = 0.2

        action_delay = 0.0
        action_noise = 0.0

    class commands(XBotLCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        heading_command = False
        resampling_time = 8.0

        gait = ["stand"]
        gait_time_range = {"stand": [1.0, 1.0]}
        stand_com_threshold = 0.05

        class ranges(XBotLCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(XBotLCfg.rewards):
        only_positive_rewards = True
        upright_sigma = 8.0
        stability_sigma = 1.5

        class scales:
            upright = 2.0
            stability = 0.6
            energy = -2e-4
            position = -0.5
            action_rate = -0.01
            termination = -5.0

    class normalization(XBotLCfg.normalization):
        clip_observations = 10.0
        clip_actions = 1.0


class TwoWheelBalancerCfgPPO(XBotLCfgPPO):
    class runner(XBotLCfgPPO.runner):
        max_iterations = 4000
        save_interval = 200
        experiment_name = "two_wheel_balancer_ppo"
        run_name = ""
