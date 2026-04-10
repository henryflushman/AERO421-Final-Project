# =============================================================
# AERO 421 - Final Project
#   Part 2: Torque-Free Orbital and Attitude Motion
#
# Written by Henry Flushman
# Collaborators:
#   - Jackson Mehiel
#   - Nick Schaeffer
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import satellite.SatelliteObject as sat_mod

SO = sat_mod.SatelliteObject


# =============================================================
# Helper: COEs -> Cartesian state
# =============================================================
def coes_to_rv(a, e, i, raan, arg_periapsis, true_anomaly, mu):
    p = a * (1.0 - e**2)
    r_pf = np.array([
        p * np.cos(true_anomaly) / (1.0 + e * np.cos(true_anomaly)),
        p * np.sin(true_anomaly) / (1.0 + e * np.cos(true_anomaly)),
        0.0,
    ])
    v_pf = np.sqrt(mu / p) * np.array([
        -np.sin(true_anomaly),
        e + np.cos(true_anomaly),
        0.0,
    ])
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(arg_periapsis), np.sin(arg_periapsis)
    Q = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])
    return Q @ r_pf, Q @ v_pf


# =============================================================
# HELPER: Continuous Euler Angle Integration
#   Integrates the 3-2-1 kinematic ODEs using the simulated
#   ω(t) history, avoiding the arctan2 / gimbal-lock jumps
#   that corrupt the per-step extraction approach.
# =============================================================
def integrate_euler_angles(t_hist, omega_hist, euler0):
    """
    Integrate the 3-2-1 (roll-pitch-yaw) kinematic equations:

        [φ̇]   [1  sin φ tan θ   cos φ tan θ] [ωx]
        [θ̇] = [0  cos φ        -sin φ      ] [ωy]
        [ψ̇]   [0  sin φ/cos θ   cos φ/cos θ] [ωz]

    Uses the ω(t) history from the quaternion simulation via
    cubic interpolation — no arctan2 or wrapping involved.

    Parameters
    ----------
    t_hist    : (N,)   time vector [s]
    omega_hist: (3,N)  angular velocity history [rad/s]
    euler0    : (3,)   initial [φ, θ, ψ] [rad]

    Returns
    -------
    euler_cont : (3,N)  continuous Euler angles [rad]
    """
    # Cubic interpolant for ω(t)
    omega_fn = interp1d(t_hist, omega_hist, kind="cubic",
                        bounds_error=False, fill_value="extrapolate")

    def _kin(t, euler):
        phi, theta, psi = euler
        wx, wy, wz = omega_fn(t)

        # Guard against exact gimbal-lock (cos θ → 0)
        cos_th = np.cos(theta)
        if abs(cos_th) < 1e-10:
            cos_th = np.sign(cos_th) * 1e-10 if cos_th != 0 else 1e-10

        sin_ph, cos_ph = np.sin(phi), np.cos(phi)
        tan_th = np.sin(theta) / cos_th

        phi_dot = wx + sin_ph * tan_th * wy + cos_ph * tan_th * wz
        the_dot =      cos_ph             * wy - sin_ph             * wz
        psi_dot =      sin_ph / cos_th    * wy + cos_ph / cos_th    * wz
        return [phi_dot, the_dot, psi_dot]

    sol = solve_ivp(
        _kin,
        t_span=(t_hist[0], t_hist[-1]),
        y0=list(euler0),
        t_eval=t_hist,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Euler angle integration failed: {sol.message}")

    return sol.y   # (3, N)  — naturally continuous, no wrapping


# ══════════════════════════════════════════════════════════════
# 1. Load spacecraft (normal-operations phase)
# ══════════════════════════════════════════════════════════════
sc = SO("MehielSat")
J = sc.J
m = sc.totalMass

print("=" * 60)
print(f"  Spacecraft : {sc.name}")
print(f"  Mass       : {m:.3f} kg")
print(f"  Inertia tensor [kg·m²]:")
for row in J:
    print(f"      {row}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 2. Initial Orbit from spacecraft COEs
# ══════════════════════════════════════════════════════════════
MU = 3.986004418e14
R_EARTH = 6.378137e6

if not sc.orbit:
    raise ValueError("No orbit data found in spacecraft JSON.")

a    = sc.orbit["a"]
e    = sc.orbit["e"]
inc  = sc.orbit["i"]
raan = sc.orbit["raan"]
argp = sc.orbit["arg_periapsis"]
nu0  = sc.orbit["true_anomaly"]

r0, v0 = coes_to_rv(a, e, inc, raan, argp, nu0, MU)

orb      = SO.orbital_elements(r0, v0, MU)
T_orbit  = orb["T_orbit"]
n_mean   = orb["n"]

print(f"\n  Orbital elements")
print(f"      Semi-major axis : {orb['a']/1e3:.2f} km")
print(f"      Eccentricity    : {orb['e']:.2e}")
print(f"      Inclination     : {np.degrees(orb['i']):.4f} deg")
print(f"      Orbital period  : {T_orbit:.2f} s  ({T_orbit/60:.2f} min)")
print(f"      Mean altitude   : {sc.altitude/1e3:.2f} km")

# ══════════════════════════════════════════════════════════════
# 3. Initial Attitude (body ≡ LVLH at t = 0)
# ══════════════════════════════════════════════════════════════
q0, euler0, C_BI = SO.initial_attitude_from_lvlh(r0, v0)

print(f"\n  Initial attitude  (body = LVLH)")
print(f"      C_LVLH-ECI =")
for row in C_BI:
    print(f"          [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")
print(f"      q0  [q1,q2,q3,q4] = {np.round(q0, 6)}")
print(f"      Euler [φ,θ,ψ] deg = {np.round(np.degrees(euler0), 4)}")

# ══════════════════════════════════════════════════════════════
# 4. Initial Angular Velocity
# ══════════════════════════════════════════════════════════════
omega0 = np.array([0.001, -0.001, 0.002])   # [rad/s]
print(f"\n  ω₀ = {omega0} rad/s")

x0_att = np.hstack([omega0, q0])

# ══════════════════════════════════════════════════════════════
# 5. Time vector — one orbital period
# ══════════════════════════════════════════════════════════════
N_PTS  = 4000
t_eval = np.linspace(0.0, T_orbit, N_PTS)

# ══════════════════════════════════════════════════════════════
# 6. Simulate attitude
# ══════════════════════════════════════════════════════════════
print("\n  Simulating attitude …", end=" ", flush=True)
t_att, omega_h, q_h, euler_h = sc.simulate_attitude(
    t_eval, x0_att, T_ext=None,
    integrator_opts={
        "solve_ivp_method": "RK45",
        "solve_ivp_kwargs": {"rtol": 1e-10, "atol": 1e-12},
    },
)
print("done.")

# ══════════════════════════════════════════════════════════════
# 7. Simulate orbit
# ══════════════════════════════════════════════════════════════
print("  Simulating orbit …", end=" ", flush=True)
t_orb, r_h, v_h = sc.simulate_orbit(
    t_eval, r0, v0, mu=MU,
    integrator_opts={
        "solve_ivp_method": "RK45",
        "solve_ivp_kwargs": {"rtol": 1e-11, "atol": 1e-13},
    },
)
print("done.\n")

# ══════════════════════════════════════════════════════════════
# 8. Continuous Euler Angles
# ══════════════════════════════════════════════════════════════
print("  Computing continuous Euler angles …", end=" ", flush=True)
euler_cont = integrate_euler_angles(t_att, omega_h, euler0)
print("done.\n")

# ══════════════════════════════════════════════════════════════
# 9. Conservation diagnostics
# ══════════════════════════════════════════════════════════════
H_mag = SO.angular_momentum(J, omega_h)
q_err = SO.quaternion_norm_error(q_h)
r_mag0 = np.linalg.norm(r0)
r_err  = np.linalg.norm(r_h, axis=0) - r_mag0

print("  Conservation check")
print(f"      |H| drift (max)       : {np.max(np.abs(H_mag - H_mag[0])):.2e} kg·m²/s")
print(f"      Quaternion norm error : {np.max(np.abs(q_err)):.2e}")
print(f"      Orbital radius drift  : {np.max(np.abs(r_err)):.2e} m")

# ══════════════════════════════════════════════════════════════
# 10. Plotting
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "lines.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C1, C2, C3, C4 = "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"

# ── Figure — main deliverable plot ─────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig.suptitle("Body to ECI Dynamics and Kinematics", fontsize=13, fontweight="bold")

# Panel 1 — Angular Velocities
ax = axes[0]
ax.set_title("Angular Velocities")
ax.plot(t_att, omega_h[0], color=C1, label=r"$\omega_x$")
ax.plot(t_att, omega_h[1], color=C2, label=r"$\omega_y$")
ax.plot(t_att, omega_h[2], color=C3, label=r"$\omega_z$")
ax.set_ylabel("angular velocity (rad/sec)")
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{x*1e3:.1f}×10⁻³" if x != 0 else "0")
)
ax.legend(loc="upper right", framealpha=0.8)

# Panel 2 — Quaternions
ax = axes[1]
ax.set_title("Quaternions")
ax.plot(t_att, q_h[0], color=C1, label=r"$q_1$")
ax.plot(t_att, q_h[1], color=C2, label=r"$q_2$")
ax.plot(t_att, q_h[2], color=C3, label=r"$q_3$")
ax.plot(t_att, q_h[3], color=C4, label=r"$q_4$")
ax.set_ylabel("Quaternion Parameter")
ax.set_ylim(-1.05, 1.05)
ax.legend(loc="upper right", framealpha=0.8)

# Panel 3 — Euler Angles  (continuous — from ODE integration)
ax = axes[2]
ax.set_title("Euler Angles  (3-2-1, continuously integrated)")
ax.plot(t_att, np.degrees(euler_cont[0]), color=C1, label=r"$\phi$   (roll)")
ax.plot(t_att, np.degrees(euler_cont[1]), color=C2, label=r"$\theta$ (pitch)")
ax.plot(t_att, np.degrees(euler_cont[2]), color=C3, label=r"$\psi$  (yaw)")
ax.set_ylabel("Angle (deg)")
ax.set_xlabel("time (seconds)")
ax.legend(loc="upper right", framealpha=0.8)

fig.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# 11. Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  INITIAL CONDITIONS SUMMARY")
print("=" * 60)
print(f"  r₀  [km]           : {r0/1e3}")
print(f"  v₀  [km/s]         : {v0/1e3}")
print(f"  ω₀  [rad/s]        : {omega0}")
print(f"  q₀  [q1,q2,q3,q4]  : {np.round(q0, 6)}")
print(f"  φ₀  [deg]  roll    : {np.degrees(euler0[0]):.4f}")
print(f"  θ₀  [deg]  pitch   : {np.degrees(euler0[1]):.4f}")
print(f"  ψ₀  [deg]  yaw     : {np.degrees(euler0[2]):.4f}")
print(f"  T   [s]            : {T_orbit:.2f}  ({T_orbit/60:.2f} min)")
print("=" * 60)