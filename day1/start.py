import time
from collections import deque
from threading import Lock
from typing import Callable, Dict, Tuple

import numpy as np
import mujoco
import mujoco.viewer


def sensor_slice(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str) -> np.ndarray:
    sid = model.sensor(sensor_name).id
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr: adr + dim]


def root_pitch_from_qpos(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    num = 2.0 * (x * z + w * y)
    den = 1.0 - 2.0 * (x * x + y * y)
    return float(np.arctan2(num, den))


def solve_discrete_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Solve the discrete-time LQR controller gain K for x[k+1]=Ax[k]+Bu[k]."""
    P = Q.copy()
    for _ in range(max_iter):
        bt_p = B.T @ P
        s = R + bt_p @ B
        k = np.linalg.solve(s, bt_p @ A)
        p_next = A.T @ P @ A - A.T @ P @ B @ k + Q
        if np.max(np.abs(p_next - P)) < tol:
            P = p_next
            break
        P = p_next
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def linearize_balance_dynamics(
    model: mujoco.MjModel,
    dt: float,
    wheel_l_qadr: int,
    wheel_r_qadr: int,
    wheel_l_dadr: int,
    wheel_r_dadr: int,
    left_motor: int,
    right_motor: int,
    mit_kp: float,
    mit_kd: float,
    torque_limit: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearize the closed loop (MIT inner loop included) around the upright equilibrium.
    State: [pitch, pitch_rate, x_err, x_vel_err, pitch_integral], input: qd_balance.
    """

    def one_step(state: np.ndarray, u_cmd: float) -> np.ndarray:
        pitch, pitch_rate, x_err, x_vel_err, pitch_i = state

        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Root pose/velocity around upright equilibrium.
        data.qpos[0] = float(x_err)
        data.qpos[3] = float(np.cos(0.5 * pitch))
        data.qpos[4] = 0.0
        data.qpos[5] = float(np.sin(0.5 * pitch))
        data.qpos[6] = 0.0
        data.qvel[:] = 0.0
        data.qvel[0] = float(x_vel_err)
        data.qvel[4] = float(pitch_rate)

        mujoco.mj_forward(model, data)

        q_l = float(data.qpos[wheel_l_qadr])
        q_r = float(data.qpos[wheel_r_qadr])
        qd_l = float(data.qvel[wheel_l_dadr])
        qd_r = float(data.qvel[wheel_r_dadr])

        q_des_l = q_l + u_cmd * dt
        q_des_r = q_r + u_cmd * dt
        tau_l = mit_kp * (q_des_l - q_l) + mit_kd * (u_cmd - qd_l)
        tau_r = mit_kp * (q_des_r - q_r) + mit_kd * (u_cmd - qd_r)

        data.ctrl[left_motor] = float(np.clip(tau_l, -torque_limit, torque_limit))
        data.ctrl[right_motor] = float(np.clip(tau_r, -torque_limit, torque_limit))

        mujoco.mj_step(model, data)

        pitch_next = root_pitch_from_qpos(data.qpos[3:7])
        pitch_rate_next = float(data.qvel[4])
        x_err_next = float(data.qpos[0])
        x_vel_err_next = float(data.qvel[0])
        pitch_i_next = float(pitch_i + pitch * dt)
        return np.array(
            [pitch_next, pitch_rate_next, x_err_next, x_vel_err_next, pitch_i_next],
            dtype=np.float64,
        )

    x0 = np.zeros(5, dtype=np.float64)
    u0 = 0.0
    A = np.zeros((5, 5), dtype=np.float64)
    B = np.zeros((5, 1), dtype=np.float64)
    eps_x = np.array([1e-4, 1e-3, 1e-4, 1e-3, 1e-5], dtype=np.float64)
    eps_u = 1e-4

    for i in range(5):
        dx = np.zeros(5, dtype=np.float64)
        dx[i] = eps_x[i]
        f_plus = one_step(x0 + dx, u0)
        f_minus = one_step(x0 - dx, u0)
        A[:, i] = (f_plus - f_minus) / (2.0 * eps_x[i])

    f_u_plus = one_step(x0, u0 + eps_u)
    f_u_minus = one_step(x0, u0 - eps_u)
    B[:, 0] = (f_u_plus - f_u_minus) / (2.0 * eps_u)
    return A, B


def _build_runtime(xml_path: str) -> Dict[str, object]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    left_motor = model.actuator("left_wheel_motor").id
    right_motor = model.actuator("right_wheel_motor").id

    wheel_l_joint = model.joint("wheel_L_hinge").id
    wheel_r_joint = model.joint("wheel_R_hinge").id
    wheel_l_qadr = model.jnt_qposadr[wheel_l_joint]
    wheel_r_qadr = model.jnt_qposadr[wheel_r_joint]
    wheel_l_dadr = model.jnt_dofadr[wheel_l_joint]
    wheel_r_dadr = model.jnt_dofadr[wheel_r_joint]

    return {
        "model": model,
        "data": data,
        "left_motor": left_motor,
        "right_motor": right_motor,
        "wheel_l_qadr": wheel_l_qadr,
        "wheel_r_qadr": wheel_r_qadr,
        "wheel_l_dadr": wheel_l_dadr,
        "wheel_r_dadr": wheel_r_dadr,
        "dt": model.opt.timestep,
        "wheel_speed_limit": 12.9302,
        "mit_kp": 21.0087,
        "mit_kd": 0.9715,
        "torque_limit": 25.0,
        "cmd_v_limit": 2.5,
        "cmd_yaw_limit": 1.8,
    }


def _run_mit_balance_loop(
    runtime: Dict[str, object],
    run_seconds: float,
    enable_plot: bool,
    controller_name: str,
    outer_controller: Callable[[float, float, float, float, float], float],
) -> None:
    model: mujoco.MjModel = runtime["model"]  # type: ignore[assignment]
    data: mujoco.MjData = runtime["data"]  # type: ignore[assignment]
    left_motor = int(runtime["left_motor"])
    right_motor = int(runtime["right_motor"])
    wheel_l_qadr = int(runtime["wheel_l_qadr"])
    wheel_r_qadr = int(runtime["wheel_r_qadr"])
    wheel_l_dadr = int(runtime["wheel_l_dadr"])
    wheel_r_dadr = int(runtime["wheel_r_dadr"])
    dt = float(runtime["dt"])
    wheel_speed_limit = float(runtime["wheel_speed_limit"])
    mit_kp = float(runtime["mit_kp"])
    mit_kd = float(runtime["mit_kd"])
    torque_limit = float(runtime["torque_limit"])
    cmd_v_limit = float(runtime["cmd_v_limit"])
    cmd_yaw_limit = float(runtime["cmd_yaw_limit"])

    pitch_integral = 0.0
    q_des_l = float(data.qpos[wheel_l_qadr])
    q_des_r = float(data.qpos[wheel_r_qadr])
    x_ref = float(data.qpos[0])

    try:
        from pynput import keyboard
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `pynput` for hold/release keyboard control. "
            "Install with: pip install pynput"
        ) from exc

    key_state = {"w": False, "s": False, "a": False, "d": False}
    key_lock = Lock()

    def key_to_name(key) -> str:
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        if key == keyboard.Key.space:
            return "space"
        return ""

    def on_press(key) -> None:
        nonlocal x_ref
        name = key_to_name(key)
        if not name:
            return
        with key_lock:
            if name in key_state:
                key_state[name] = True
            elif name == "space":
                key_state["w"] = False
                key_state["s"] = False
                key_state["a"] = False
                key_state["d"] = False
                x_ref = float(data.qpos[0])

    def on_release(key) -> None:
        name = key_to_name(key)
        if not name:
            return
        with key_lock:
            if name in key_state:
                key_state[name] = False

    plot_ctx = None
    if enable_plot:
        import matplotlib.pyplot as plt

        history_len = int(10.0 / dt)
        t_buf = deque(maxlen=history_len)
        tau_l_buf = deque(maxlen=history_len)
        tau_r_buf = deque(maxlen=history_len)
        ctrl_raw_buf = deque(maxlen=history_len)
        ctrl_clip_buf = deque(maxlen=history_len)
        pitch_buf = deque(maxlen=history_len)
        gyro_buf = deque(maxlen=history_len)

        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        try:
            fig.canvas.manager.set_window_title("Two-Wheel Balancer Telemetry")
        except Exception:
            pass

        line_tau_l, = axes[0].plot([], [], label="tau_left")
        line_tau_r, = axes[0].plot([], [], label="tau_right")
        axes[0].set_ylabel("Torque (Nm)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        line_ctrl_raw, = axes[1].plot([], [], label="ctrl_raw")
        line_ctrl_clip, = axes[1].plot([], [], label="ctrl_clipped")
        axes[1].set_ylabel("Outer Ctrl (rad/s)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        line_pitch, = axes[2].plot([], [], label="pitch")
        line_gyro, = axes[2].plot([], [], label="gyro_y")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Angle / Rate")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        plot_ctx = (
            plt,
            fig,
            axes,
            line_tau_l,
            line_tau_r,
            line_ctrl_raw,
            line_ctrl_clip,
            line_pitch,
            line_gyro,
            t_buf,
            tau_l_buf,
            tau_r_buf,
            ctrl_raw_buf,
            ctrl_clip_buf,
            pitch_buf,
            gyro_buf,
        )

    print(
        f"Controller mode: {controller_name.upper()} | "
        "Hold W/S: forward/backward, hold A/D: turn, release to stop, Space: emergency stop."
    )
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            end_time = time.time() + run_seconds
            step_count = 0
            while viewer.is_running() and time.time() < end_time:
                tick = time.time()
                step_count += 1

                gyro = sensor_slice(model, data, "imu_gyro")
                pitch = root_pitch_from_qpos(data.qpos[3:7])
                pitch_rate = float(gyro[1])
                base_x = float(data.qpos[0])
                base_x_vel = float(data.qvel[0])

                with key_lock:
                    w_down = key_state["w"]
                    s_down = key_state["s"]
                    a_down = key_state["a"]
                    d_down = key_state["d"]

                cmd_v_target = cmd_v_limit * (float(w_down) - float(s_down))
                cmd_yaw_target = cmd_yaw_limit * (float(a_down) - float(d_down))

                if abs(cmd_v_target) < 1e-9:
                    x_ref = base_x
                else:
                    x_ref += cmd_v_target * dt

                pitch_integral = float(np.clip(pitch_integral + pitch * dt, -0.4, 0.4))
                x_err = base_x - x_ref
                x_vel_err = base_x_vel - cmd_v_target

                qd_balance_raw = float(
                    outer_controller(pitch, pitch_rate, pitch_integral, x_err, x_vel_err)
                )
                qd_balance = float(np.clip(qd_balance_raw, -wheel_speed_limit, wheel_speed_limit))

                qd_des_l = float(np.clip(qd_balance - cmd_yaw_target, -wheel_speed_limit, wheel_speed_limit))
                qd_des_r = float(np.clip(qd_balance + cmd_yaw_target, -wheel_speed_limit, wheel_speed_limit))

                q_des_l += qd_des_l * dt
                q_des_r += qd_des_r * dt

                q_l = float(data.qpos[wheel_l_qadr])
                q_r = float(data.qpos[wheel_r_qadr])
                qd_l = float(data.qvel[wheel_l_dadr])
                qd_r = float(data.qvel[wheel_r_dadr])

                tau_l = mit_kp * (q_des_l - q_l) + mit_kd * (qd_des_l - qd_l)
                tau_r = mit_kp * (q_des_r - q_r) + mit_kd * (qd_des_r - qd_r)

                data.ctrl[left_motor] = float(np.clip(tau_l, -torque_limit, torque_limit))
                data.ctrl[right_motor] = float(np.clip(tau_r, -torque_limit, torque_limit))

                if plot_ctx is not None:
                    (
                        plt,
                        _fig,
                        axes,
                        line_tau_l,
                        line_tau_r,
                        line_ctrl_raw,
                        line_ctrl_clip,
                        line_pitch,
                        line_gyro,
                        t_buf,
                        tau_l_buf,
                        tau_r_buf,
                        ctrl_raw_buf,
                        ctrl_clip_buf,
                        pitch_buf,
                        gyro_buf,
                    ) = plot_ctx

                    t_buf.append(float(data.time))
                    tau_l_buf.append(float(np.clip(tau_l, -torque_limit, torque_limit)))
                    tau_r_buf.append(float(np.clip(tau_r, -torque_limit, torque_limit)))
                    ctrl_raw_buf.append(float(qd_balance_raw))
                    ctrl_clip_buf.append(float(qd_balance))
                    pitch_buf.append(float(pitch))
                    gyro_buf.append(float(pitch_rate))

                mujoco.mj_step(model, data)
                viewer.sync()

                if plot_ctx is not None and step_count % 20 == 0 and len(plot_ctx[9]) > 5:
                    (
                        plt,
                        _fig,
                        axes,
                        line_tau_l,
                        line_tau_r,
                        line_ctrl_raw,
                        line_ctrl_clip,
                        line_pitch,
                        line_gyro,
                        t_buf,
                        tau_l_buf,
                        tau_r_buf,
                        ctrl_raw_buf,
                        ctrl_clip_buf,
                        pitch_buf,
                        gyro_buf,
                    ) = plot_ctx

                    t_arr = np.asarray(t_buf)
                    line_tau_l.set_data(t_arr, np.asarray(tau_l_buf))
                    line_tau_r.set_data(t_arr, np.asarray(tau_r_buf))
                    line_ctrl_raw.set_data(t_arr, np.asarray(ctrl_raw_buf))
                    line_ctrl_clip.set_data(t_arr, np.asarray(ctrl_clip_buf))
                    line_pitch.set_data(t_arr, np.asarray(pitch_buf))
                    line_gyro.set_data(t_arr, np.asarray(gyro_buf))

                    x_min = max(0.0, t_arr[-1] - 10.0)
                    x_max = t_arr[-1] + 1e-6
                    for ax in axes:
                        ax.set_xlim(x_min, x_max)
                        ax.relim()
                        ax.autoscale_view(scalex=False, scaley=True)

                    plt.pause(0.001)

                sleep_time = dt - (time.time() - tick)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        listener.stop()


def run_mit_pid_balance(
    xml_path: str = "two_wheel_balancer.xml",
    run_seconds: float = 60.0,
    enable_plot: bool = True,
) -> None:
    runtime = _build_runtime(xml_path)
    pid_params: Dict[str, float] = {
        "kp_pitch": 158.2541,
        "ki_pitch": 36.7398,
        "kd_pitch": 1.8322,
        "k_pos": 0.5492,
        "k_vel": 40.3713,
    }

    def pid_outer_controller(
        pitch: float, pitch_rate: float, pitch_integral: float, x_err: float, x_vel_err: float
    ) -> float:
        return (
            pid_params["kp_pitch"] * pitch
            + pid_params["ki_pitch"] * pitch_integral
            - pid_params["kd_pitch"] * pitch_rate
            - pid_params["k_pos"] * x_err
            - pid_params["k_vel"] * x_vel_err
        )

    _run_mit_balance_loop(runtime, run_seconds, enable_plot, "pid", pid_outer_controller)


def run_mit_lqr_balance(
    xml_path: str = "two_wheel_balancer.xml",
    run_seconds: float = 60.0,
    enable_plot: bool = True,
) -> None:
    runtime = _build_runtime(xml_path)
    model: mujoco.MjModel = runtime["model"]  # type: ignore[assignment]
    dt = float(runtime["dt"])

    A, B = linearize_balance_dynamics(
        model=model,
        dt=dt,
        wheel_l_qadr=int(runtime["wheel_l_qadr"]),
        wheel_r_qadr=int(runtime["wheel_r_qadr"]),
        wheel_l_dadr=int(runtime["wheel_l_dadr"]),
        wheel_r_dadr=int(runtime["wheel_r_dadr"]),
        left_motor=int(runtime["left_motor"]),
        right_motor=int(runtime["right_motor"]),
        mit_kp=float(runtime["mit_kp"]),
        mit_kd=float(runtime["mit_kd"]),
        torque_limit=float(runtime["torque_limit"]),
    )
    Q = np.diag([1024.0, 1.489, 1.136, 35.47, 50530.0])
    R = np.array([[0.22329523909678076]])
    lqr_gain = solve_discrete_lqr(A, B, Q, R).reshape(-1)
    print(f"[LQR] gain K = {np.array2string(lqr_gain, precision=4)}")

    def lqr_outer_controller(
        pitch: float, pitch_rate: float, pitch_integral: float, x_err: float, x_vel_err: float
    ) -> float:
        lqr_state = np.array([pitch, pitch_rate, x_err, x_vel_err, pitch_integral], dtype=np.float64)
        return float(-(lqr_gain @ lqr_state))

    _run_mit_balance_loop(runtime, run_seconds, enable_plot, "lqr", lqr_outer_controller)


if __name__ == "__main__":
    run_mit_lqr_balance()
