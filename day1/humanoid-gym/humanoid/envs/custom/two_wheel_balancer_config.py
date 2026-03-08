# SPDX-License-Identifier: BSD-3-Clause

from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class TwoWheelBalancerCfg(LeggedRobotCfg):
    _MAX_CURRICULUM_STAGE = 4

    class env(LeggedRobotCfg.env):
        num_active_dofs = 2
        num_passive_dofs = 0
        num_commands = 3

        frame_stack = 3
        c_frame_stack = 1

        obs_names = ["base_euler_xyz", "base_ang_vel", "dof_pos", "dof_vel", "base_lin_vel", "actions"]
        privileged_obs_names = ["base_euler_xyz", "base_ang_vel", "dof_pos", "dof_vel", "base_lin_vel", "actions"]

        num_envs = 4096
        episode_length_s = 60.0
        use_ref_actions = False
        residual_action_scale = 0.0

        # Reference controller (residual RL): action = ref_action + residual_scale * policy_action
        ref_kp_pitch = 8.0
        ref_kd_pitch = 0.10
        ref_ki_pitch = 1.8
        ref_kp_pos = 0.03
        ref_kd_pos = 2.0
        ref_action_clip = 1.0
        ref_pitch_i_clip = 0.4

        max_tilt_deg = 75.0
        over_tilt_grace_steps = 1
        # base_link origin is below wheel centers in this URDF, so nominal z is near 0.0.
        min_base_height = -0.03
        base_contact_force_threshold = 20.0

        reset_angle_range = 0.03
        reset_wheel_vel = 0.0
        reset_base_lin_vel = 0.0
        reset_base_ang_vel = 0.0
        _curriculum_stage = 0

        @property
        def curriculum_stage(self):
            return self._curriculum_stage

        @curriculum_stage.setter
        def curriculum_stage(self, value):
            stage = int(value)
            if stage < 0 or stage > TwoWheelBalancerCfg._MAX_CURRICULUM_STAGE:
                raise ValueError(
                    f"curriculum_stage must be in [0, {TwoWheelBalancerCfg._MAX_CURRICULUM_STAGE}], got {value}."
                )
            self._curriculum_stage = stage
            parent_cfg = getattr(self, "_parent_cfg", None)
            if parent_cfg is not None:
                parent_cfg._apply_curriculum_stage(stage)

    class safety:
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/two_wheel_balancer/urdf/two_wheel_balancer.urdf"
        name = "two_wheel_balancer"
        foot_names = ["wheel_L", "wheel_R"]
        knee_names = []

        # base_link may continuously graze/receive contact impulses on this URDF;
        # keep termination driven by tilt/height/timeout instead of base contact.
        terminate_after_contacts_on = []
        penalize_contacts_on = ["base_link"]
        self_collisions = 1
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        static_friction = 1.5
        dynamic_friction = 1.4
        restitution = 0.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 0.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.15
            base_ang_vel = 0.08
            base_lin_vel = 0.08
            base_euler = 0.03

    class init_state(LeggedRobotCfg.init_state):
        # Align spawn height with URDF kinematics: wheel centers are at z=0.07 relative to base_link.
        pos = [0.0, 0.0, 0.0]

        default_joint_angles = {
            "wheel_L_joint": 0.0,
            "wheel_R_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {"wheel": 0.0}
        damping = {"wheel": 0.0}
        action_scale = 30.0
        wheel_damping = 0.0
        decimation = 5

    class sim(LeggedRobotCfg.sim):
        dt = 0.002
        substeps = 1
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
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

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.8, 1.6]

        randomize_restitution = False
        restitution_range = [0.0, 0.2]

        randomize_base_mass = False
        added_mass_range = [-0.2, 0.2]

        randomize_com_displacement = False
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

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        heading_command = False
        resampling_time = 8.0

        gait = ["stand"]
        gait_time_range = {"stand": [1.0, 1.0]}
        stand_com_threshold = 0.05

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        upright_sigma = 12.0
        stability_sigma = 2.0

        class scales:
            # upright = 2.0
            # stability = 0.6
            # energy = -2e-4
            # position = -0.5
            # action_rate = -0.01
            # termination = -5.0
            upright = 5.0
            stability = 2.0
            energy = -5e-5
            position = -0.05
            action_rate = -0.005
            alive = 1.0
            termination = -120.0

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 10.0
        clip_actions = 1.0

    def __init__(self):
        super().__init__()
        self.env._parent_cfg = self
        self._apply_curriculum_stage(self.env.curriculum_stage)

    def _apply_curriculum_stage(self, stage=None):
        if stage is None:
            stage = self.env.curriculum_stage
        stage = int(stage)
        if stage < 0 or stage > self._MAX_CURRICULUM_STAGE:
            raise ValueError(f"curriculum_stage must be in [0, {self._MAX_CURRICULUM_STAGE}], got {stage}.")

        self.env._curriculum_stage = stage

        # Reset to the exact easy-mode baseline before applying each curriculum stage.
        self.domain_rand.randomize_friction = False
        self.domain_rand.friction_range = [0.8, 1.6]
        self.domain_rand.randomize_base_mass = False
        self.domain_rand.added_mass_range = [-0.2, 0.2]
        self.domain_rand.randomize_com_displacement = False
        self.domain_rand.com_displacement_range = [-0.01, 0.01]
        self.domain_rand.push_robots = False

        self.noise.add_noise = False
        self.noise.noise_level = 0.0

        if stage >= 1:
            self.domain_rand.randomize_friction = True
            self.domain_rand.friction_range = [0.8, 1.6]
        if stage >= 2:
            self.domain_rand.randomize_base_mass = True
            self.domain_rand.added_mass_range = [-0.5, 0.5]
        if stage >= 3:
            self.domain_rand.randomize_com_displacement = True
            self.domain_rand.com_displacement_range = [-0.02, 0.02]
        if stage >= 4:
            self.noise.add_noise = True
            self.noise.noise_level = 1.0


class TwoWheelBalancerCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.05

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 1e-4
        learning_rate = 3e-4

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 500
        save_interval = 200
        experiment_name = "two_wheel_balancer_ppo"
        run_name = ""
