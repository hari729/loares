
#!/usr/bin/env python3
"""
Nao gait evaluator — full Pinocchio implementation with numerical IK.

Requirements:
- pinocchio installed (conda-forge recommended)
- a valid NAO URDF file path (no dummy URDF will be created)

Features:
- Damped least-squares IK on SE(3) for specified frames
- Finite-difference velocity/acceleration estimation
- Uses pinocchio RNEA, centerOfMass, computeCentroidalMomentumTimeVariation
- Returns objectives (avg_power, 1/DBM) and 8 constraints per solution
"""

import os
import numpy as np
import pinocchio as pin


# ------------------ Numerical IK (Damped least squares on SE(3)) ------------------
def numerical_ik(model, data, frame_id, target_SE3, q_init=None,
                 max_iter=100, tol=1e-6, damping=1e-4):
    """
    Solve IK for a single frame to reach target_SE3 using damped least squares.
    - model, data: Pinocchio model and data
    - frame_id: integer frame id
    - target_SE3: pin.SE3 desired pose (target)
    - q_init: initial configuration (size nq) or None -> neutral
    Returns: q (configuration achieving pose approx), success (bool)
    """
    if q_init is None:
        q = pin.neutral(model)
    else:
        q = q_init.copy()

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_SE3 = data.oMf[frame_id]  # current placement of the frame in world
        # Pose error in tangent space (6-vector) using log map
        err_se3 = pin.log6(target_SE3.inverse() * current_SE3)  # 6-vector: [omega, v] or convention used by pin
        err_norm = np.linalg.norm(err_se3)
        if err_norm < tol:
            return q, True
        # Jacobian of frame (6 x nv)
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)  # LOCAL or WORLD can be used; LOCAL widely used
        # Damped least squares solution for dq (nv,)
        # J is 6 x nv; solve J dq = -err_se3
        JT = J.T
        JJ = J @ JT  # 6x6
        # Solve for lambda: (J J^T + damping^2 I) y = -err; dq = J^T y
        reg = damping * np.eye(JJ.shape[0])
        try:
            y = np.linalg.solve(JJ + reg, -err_se3)
        except np.linalg.LinAlgError:
            # fallback to pseudoinverse
            dq = -np.linalg.pinv(J) @ err_se3
            q = pin.integrate(model, q, dq)
            continue
        dq = JT @ y  # nv vector
        # Integrate configuration
        q = pin.integrate(model, q, dq)
    return q, False


# ------------------ Utility: find a frame id in model ------------------
def find_first_frame(model, candidates):
    """
    Return first frame id that exists in model, else raise KeyError.
    candidates: list of frame names.
    """
    for name in candidates:
        try:
            fid = model.getFrameId(name)
            # getFrameId returns an id even if the frame is not present? double-check by name string check:
            # In many URDFs frame names exist; we'll accept the id if name matches model.frames[fid].name
            if model.frames[fid].name == name:
                return fid, name
        except Exception:
            continue
    raise KeyError(f"No frame found among candidates: {candidates}")


# ------------------ Main optimizer class (real Pinocchio usage) ------------------
class NaoGaitOptimizer:
    def __init__(self, urdf_path: str, left_foot_candidates=None, right_foot_candidates=None):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Build model & data (free-flyer for humanoid)
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        # Constants
        self.g = 9.81
        self.K_heat_loss = 0.025
        self.num_time_steps = 100

        # default common foot frame name candidates (try typical URDF names)
        if left_foot_candidates is None:
            left_foot_candidates = ['LFoot', 'LeftFoot', 'l_foot', 'l_ankle', 'LeftAnkle', 'l_sole']
        if right_foot_candidates is None:
            right_foot_candidates = ['RFoot', 'RightFoot', 'r_foot', 'r_ankle', 'RightAnkle', 'r_sole']

        # Find actual frame ids in the model
        try:
            self.l_foot_frame_id, self.l_foot_frame_name = find_first_frame(self.model, left_foot_candidates)
        except KeyError:
            self.l_foot_frame_id, self.l_foot_frame_name = None, None

        try:
            self.r_foot_frame_id, self.r_foot_frame_name = find_first_frame(self.model, right_foot_candidates)
        except KeyError:
            self.r_foot_frame_id, self.r_foot_frame_name = None, None

        # Prepare q_min / q_max (if actuated joint names are standard)
        self.actuated_joint_names = [
            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'
        ]
        q_min_legs = np.array([-1.14, -0.38, -1.53, -0.09, -1.19, -0.39,
                               -1.14, -0.79, -1.53, -0.10, -1.18, -0.76], dtype=float)
        q_max_legs = np.array([0.74, 0.79, 0.48, 2.11, 0.92, 0.76,
                               0.74, 0.38, 0.48, 2.12, 0.93, 0.39], dtype=float)

        self.q_min = pin.neutral(self.model).copy()
        self.q_max = pin.neutral(self.model).copy()
        for i, name in enumerate(self.actuated_joint_names):
            try:
                jid = self.model.getJointId(name)
                idx_q = self.model.joints[jid].idx_q
                if 0 <= idx_q < len(self.q_min):
                    self.q_min[idx_q] = q_min_legs[i]
                    self.q_max[idx_q] = q_max_legs[i]
            except Exception:
                continue  # skip if joint missing

    # --- cubic coefficients (unchanged) ---
    def _solve_cubic_coeffs(self, t0, tf, p0, pf, v0, vf):
        t0, tf = float(t0), float(tf)
        A = np.array([
            [t0**3, t0**2, t0, 1.0],
            [tf**3, tf**2, tf, 1.0],
            [3.0*t0**2, 2.0*t0, 1.0, 0.0],
            [3.0*tf**2, 2.0*tf, 1.0, 0.0]
        ], dtype=float)
        b = np.array([float(p0), float(pf), float(v0), float(vf)], dtype=float)
        return np.linalg.solve(A, b)

    # --- build simple hip/ankle/arm trajectories as earlier but using scalars ---
    def _get_trajectories(self, x, phase):
        x = np.asarray(x).reshape(-1)
        # enforce minimum dimensionality for the expected mapping; adapt as needed
        if x.size < 12:
            raise ValueError("Decision vector x must contain at least 12 elements (see mapping).")

        # mapping (example):
        h = float(x[0])
        T = float(x[1])
        v_iz = float(x[2])
        v_fz = float(x[3])
        v_ix = float(x[4])
        v_fx = float(x[5])
        v_iy = float(x[6])
        v_fy = float(x[7])
        q_sp_i = float(x[8])
        q_sp_f = float(x[9])
        q_er_i = float(x[10])
        s_h_max = float(x[11])

        t0, tf = 0.0, max(1e-6, T)

        if phase == 'SSP':
            x_hi, x_hf = 0.03, 0.09
            y_hi, y_hf = 0.025, 0.025
        else:
            x_hi, x_hf = 0.07, 0.11
            y_hi, y_hf = 0.025, -0.025

        coeffs_xh = self._solve_cubic_coeffs(t0, tf, x_hi, x_hf, v_ix, v_fx)
        coeffs_yh = self._solve_cubic_coeffs(t0, tf, y_hi, y_hf, v_iy, v_fy)
        coeffs_zh = self._solve_cubic_coeffs(t0, tf, h, h, v_iz, v_fz)
        coeffs_sp = self._solve_cubic_coeffs(t0, tf, q_sp_i, q_sp_f, 0.0, 0.0)
        coeffs_er = self._solve_cubic_coeffs(t0, tf, q_er_i, (q_er_i if phase == 'SSP' else -1.5), 0.0, 0.0)

        def poly(coeffs):
            a, b, c, d = coeffs
            return lambda t: (a*t**3 + b*t**2 + c*t + d,
                              3*a*t**2 + 2*b*t + c,
                              6*a*t + 2*b)

        traj = {
            'hip_x': poly(coeffs_xh),
            'hip_y': poly(coeffs_yh),
            'hip_z': poly(coeffs_zh),
            'arm_sp': poly(coeffs_sp),
            'arm_er': poly(coeffs_er)
        }

        # Ankle (SSP)
        if phase == 'SSP':
            xi, xf = 0.0, 0.12
            xm = 0.5*(xi + xf)
            s_l = 0.06
            A_za = np.array([
                [xi**3, xi**2, xi, 1.0],
                [(xi + s_l)**3, (xi + s_l)**2, (xi + s_l), 1.0],
                [xm**3, xm**2, xm, 1.0],
                [3.0*xm**2, 2.0*xm, 1.0, 0.0]
            ], dtype=float)
            b_za = np.array([0.0, 0.0, s_h_max, 0.0], dtype=float)
# Check and regularize if singular
            if np.linalg.matrix_rank(A_za) < 4:
                print("[Warning] Singular A_za encountered. Using least-squares fallback.")
                coeffs_za = np.linalg.lstsq(A_za, b_za, rcond=None)[0]
            else:
                coeffs_za = np.linalg.solve(A_za, b_za)
            coeffs_xa = self._solve_cubic_coeffs(0.0, tf, xi, xf, 0.0, 0.0)

            xa = poly(coeffs_xa)
            def za(t):
                xa_val, xa_dot, xa_ddot = xa(t)
                b3, b2, b1, b0 = coeffs_za
                z = b3*xa_val**3 + b2*xa_val**2 + b1*xa_val + b0
                dz_dx = 3.0*b3*xa_val**2 + 2.0*b2*xa_val + b1
                ddz_dx2 = 6.0*b3*xa_val + 2.0*b2
                z_dot = dz_dx * xa_dot
                z_ddot = ddz_dx2 * xa_dot**2 + dz_dx * xa_ddot
                return z, z_dot, z_ddot

            traj['ankle_x'] = xa
            traj['ankle_z'] = za

        return traj

    # --- Evaluate one gait with full Pinocchio computations ---
    def evaluate_single_gait(self, x, phase='SSP'):
        x = np.asarray(x).reshape(-1)
        traj = self._get_trajectories(x, phase)
        T = float(x[1])
        if T <= 0:
            raise ValueError("Duration T must be positive.")
        t_vec = np.linspace(0.0, T, self.num_time_steps)
        dt = T / (self.num_time_steps - 1)

        # histories
        q_hist = []
        tau_hist = []
        x_dbm_hist = []
        y_dbm_hist = []
        power_hist = []

        # initialize q_prev/v_prev for finite differences
        q_prev = pin.neutral(self.model)
        v_prev = np.zeros(self.model.nv)

        # choose swing foot frame id: prefer right foot if available, fallback to left
        frame_id = None
        if self.r_foot_frame_id is not None:
            frame_id = self.r_foot_frame_id
        elif self.l_foot_frame_id is not None:
            frame_id = self.l_foot_frame_id
        else:
            # If no foot frame is found, we will run IK on a joint that definitely exists (pelvis) or raise
            raise RuntimeError("No foot frame found in URDF (cannot perform IK).")

        # main loop
        for i, t in enumerate(t_vec):
            # desired swing foot pose: build SE3 from trajectories
            # Here as example, put target relative to world; in practice you'd compute target based on hip traj + offsets
            # Use ankle X and Z if present, else keep current
            # We'll create a target that moves horizontally by hip_x and vertically by ankle_z if available.
            hip_x, _, _ = traj['hip_x'](t)
            hip_y, _, _ = traj['hip_y'](t)
            hip_z, _, _ = traj['hip_z'](t)

            # For the foot target pose, place it under hip with some offset (this is application-specific)
            target_trans = np.array([hip_x, hip_y, hip_z - 0.15])  # 15 cm below hip roughly
            R_target = np.eye(3)
            target_SE3 = pin.SE3(R_target, target_trans)

            # Solve IK to get q that places frame near target
            q_init = q_prev if i > 0 else pin.neutral(self.model)
            q_sol, success = numerical_ik(self.model, self.data, frame_id, target_SE3, q_init=q_init,
                                          max_iter=80, tol=1e-6, damping=1e-4)
            if not success:
                # if IK fails, we can still proceed but mark solution as bad by making objectives inf later
                pass

            # finite differences for velocity and acceleration
            v_sol = (pin.difference(self.model, q_prev, q_sol) / dt) if i > 0 else np.zeros(self.model.nv)
            a_sol = (v_sol - v_prev) / dt if i > 0 else np.zeros(self.model.nv)

            # store q for history
            q_hist.append(q_sol.copy())

            # dynamics
            tau = pin.rnea(self.model, self.data, q_sol, v_sol, a_sol)
            tau_hist.append(tau.copy())

            # CoM and centroidal momentum
            pin.centerOfMass(self.model, self.data, q_sol, v_sol, a_sol)  # populates data.com, data.a_com
            com_pos = np.array(self.data.com).reshape(3,)
            com_acc = np.array(self.data.a_com).reshape(3,) if hasattr(self.data, 'a_com') else np.zeros(3)
            try:
                pin.computeCentroidalMomentumTimeVariation(self.model, self.data, q_sol, v_sol, a_sol)
                # The exact field name can vary by pinocchio version; try to read common ones
                L_dot = np.zeros(3)
                if hasattr(self.data, 'hg'):
                    # data.hg often has .angular_momentum_rate
                    if hasattr(self.data.hg, 'angular_momentum_rate'):
                        L_dot = np.array(self.data.hg.angular_momentum_rate).reshape(3,)
                    else:
                        # fallback if hg is a vector
                        try:
                            L_dot = np.array(self.data.hg).reshape(3,)
                        except Exception:
                            L_dot = np.zeros(3)
                else:
                    L_dot = np.zeros(3)
            except Exception:
                L_dot = np.zeros(3)

            # total mass: try data.mass or compute from model inertias
            total_mass = getattr(self.data, 'mass', None)
            if total_mass is None or total_mass == 0:
                # compute total mass from model inertias as fallback
                total_mass = 0.0
                for i_joint in range(self.model.njoints):
                    try:
                        I = self.model.inertias[i_joint]
                        # In some pinocchio versions inertia has m attribute
                        total_mass += getattr(I, 'mass', 0.0)
                    except Exception:
                        continue

            # ZMP-like calculation, protect dividing by zero
            denom = total_mass * (com_acc[2] + self.g) if total_mass != 0 else 0.0
            if abs(denom) < 1e-8:
                x_zmp = np.inf
                y_zmp = np.inf
            else:
                x_zmp = (total_mass * self.g * com_pos[0] - L_dot[1]) / denom
                y_zmp = (total_mass * self.g * com_pos[1] + L_dot[0]) / denom

            x_dbm = min(0.1 - x_zmp, x_zmp - (-0.05))
            y_dbm = min(0.04 - y_zmp, y_zmp - (-0.04))
            x_dbm_hist.append(float(x_dbm))
            y_dbm_hist.append(float(y_dbm))

            # instantaneous power (actuated joints only; skip floating base first 6)
            actuated_tau = tau[6:] if len(tau) > 6 else tau
            actuated_v = v_sol[6:] if len(v_sol) > 6 else v_sol
            power_inst = np.sum(np.abs(actuated_tau * actuated_v) + self.K_heat_loss * actuated_tau**2)
            power_hist.append(float(power_inst))

            # update prevs
            q_prev = q_sol
            v_prev = v_sol

        # Convert histories to arrays
        q_hist = np.array(q_hist)
        tau_hist = np.array(tau_hist)
        power_hist = np.array(power_hist)
        x_dbm_hist = np.array(x_dbm_hist)
        y_dbm_hist = np.array(y_dbm_hist)

        # Objectives
        avg_power = float(np.trapz(power_hist, dx=dt) / T)
        min_x_dbm = np.min(x_dbm_hist) if x_dbm_hist.size > 0 else -np.inf
        stability_obj = np.inf if min_x_dbm <= 1e-6 else 1.0 / float(min_x_dbm)
        objectives = [avg_power, stability_obj]

        # Constraints (g1..g8)
        constraints = []
        # g1/g2 joint limits margins
        if q_hist.size == 0:
            constraints += [-np.inf, -np.inf]
        else:
            try:
                # get minimal distance to limits across time and all joints
                dist_to_max = (self.q_max[np.newaxis, :] - q_hist)
                dist_to_min = (q_hist - self.q_min[np.newaxis, :])
                min_dist_to_max = float(np.min(dist_to_max))
                min_dist_to_min = float(np.min(dist_to_min))
            except Exception:
                min_dist_to_max = -np.inf
                min_dist_to_min = -np.inf
            constraints.append(min_dist_to_max)
            constraints.append(min_dist_to_min)

        # g3 torque fluctuation
        if tau_hist.shape[0] <= 1:
            mean_max_fluct = 0.0
        else:
            torque_fluct = np.abs(np.diff(tau_hist, axis=0))
            mean_max_fluct = float(np.mean(np.max(torque_fluct, axis=0)))
        constraints.append(0.4 - mean_max_fluct)

        # g4/g5 min DBM
        constraints.append(float(np.min(x_dbm_hist) - 0.001) if x_dbm_hist.size > 0 else -np.inf)
        constraints.append(float(np.min(y_dbm_hist) - 0.001) if y_dbm_hist.size > 0 else -np.inf)

        # g6/g7 arm coordination proxies (use decision vector entries if available)
        # We assume x[9] - x[8] and x[2] - x[10] mapping like earlier; adapt if you have a canonical mapping
        g6 = float(x[9] - x[8]) if x.size > 9 else 0.0
        g7 = float(x[2] - x[10]) if (x.size > 10 and phase == 'SSP') else 0.0
        constraints.append(g6)
        constraints.append(g7)

        # g8 average Y DBM above threshold
        avg_y_dbm = float(np.mean(y_dbm_hist)) if y_dbm_hist.size > 0 else -np.inf
        threshold = 0.02 if phase == 'SSP' else 0.05
        constraints.append(avg_y_dbm - threshold)

        # Ensure exactly 8 constraints
        if len(constraints) < 8:
            constraints += [-np.inf] * (8 - len(constraints))
        else:
            constraints = constraints[:8]

        return objectives, constraints


# ------------------ Evaluate a population ------------------
def evaluate_population(population, urdf_path, phase='SSP'):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    opt = NaoGaitOptimizer(urdf_path)
    population = np.asarray(population)
    if population.ndim == 1:
        population = population.reshape(1, -1)

    objectives_list = []
    constraints_list = []
    for i, sol in enumerate(population):
        print(f"Evaluating {i+1}/{population.shape[0]} ...")
        obj, cons = opt.evaluate_single_gait(sol, phase=phase)
        objectives_list.append(obj)
        constraints_list.append(cons)

    return np.array(objectives_list, dtype=float), np.array(constraints_list, dtype=float)


# ------------------ Example (usage) ------------------
if __name__ == '__main__':
    # Put your real URDF path here:
    URDF_PATH = "/home/hari/projects/opti/problems/urdf/nao.urdf"  # <<--- CHANGE this to your actual URDF path

    # Example decision vector bounds and sample population (13 variables)
    ssp_lower = np.array([0.25, 0.3, -0.05, -0.05, 0.001, 0.001, -0.2, -0.2, 0.015, 0.4, 0.0, 1.0, -0.04])
    ssp_upper = np.array([0.31, 0.6, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.030, 4.0, np.pi/2, 2.0, 0.5])

    n = 2
    rng = np.random.default_rng(1)
    pop = ssp_lower + rng.random((n, ssp_lower.size)) * (ssp_upper - ssp_lower)

    objectives, constraints = evaluate_population(pop, URDF_PATH, phase='SSP')
    print("Objectives:\n", objectives)
    print("Constraints:\n", constraints)
