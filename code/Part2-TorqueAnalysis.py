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
import satellite.SatelliteObject as sat_mod

SO = sat_mod.SatelliteObject   # convenience alias


# =============================================================
# Helper: COEs -> Cartesian state
# =============================================================
def coes_to_rv(a, e, i, raan, arg_periapsis, true_anomaly, mu):
    """
    Convert classical orbital elements to ECI position and velocity.

    Parameters
    ----------
    a : float
        Semi-major axis [m]
    e : float
        Eccentricity [-]
    i : float
        Inclination [rad]
    raan : float
        Right ascension of ascending node [rad]
    arg_periapsis : float
        Argument of periapsis [rad]
    true_anomaly : float
        True anomaly [rad]
    mu : float
        Gravitational parameter [m^3/s^2]

    Returns
    -------
    r_eci : (3,) ndarray
        Position in ECI [m]
    v_eci : (3,) ndarray
        Velocity in ECI [m/s]
    """
    p = a * (1.0 - e**2)

    # Perifocal position and velocity
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

    # Rotation from perifocal to ECI: R3(raan) * R1(i) * R3(arg_periapsis)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(arg_periapsis), np.sin(arg_periapsis)

    Q = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])

    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return r_eci, v_eci


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
MU = 3.986004418e14      # Earth μ [m^3/s^2]
R_EARTH = 6.378137e6     # equatorial radius [m]

if not sc.orbit:
    raise ValueError(
        "No orbit data found in spacecraft JSON. Add a 'coes' block."
    )

a = sc.orbit["a"]
e = sc.orbit["e"]
inc = sc.orbit["i"]
raan = sc.orbit["raan"]
argp = sc.orbit["arg_periapsis"]
nu0 = sc.orbit["true_anomaly"]

r0, v0 = coes_to_rv(a, e, inc, raan, argp, nu0, MU)

# Orbital elements from generated Cartesian state
orb = SO.orbital_elements(r0, v0, MU)
T_orbit = orb["T_orbit"]
n_mean = orb["n"]

print(f"\n  Orbital elements")
print(f"      Semi-major axis : {orb['a']/1e3:.2f} km")
print(f"      Eccentricity    : {orb['e']:.2e}")
print(f"      Inclination     : {np.degrees(orb['i']):.4f} deg")
print(f"      Orbital period  : {T_orbit:.2f} s  ({T_orbit/60:.2f} min)")
print(f"      Mean altitude   : {sc.altitude/1e3:.2f} km")
print(f"      Perigee alt     : {sc.perigeeAlt/1e3:.2f} km")
print(f"      Apogee alt      : {sc.apogeeAlt/1e3:.2f} km")

# ══════════════════════════════════════════════════════════════
# 3. Initial Attitude (body ≡ LVLH at t = 0)
#
#   C_LVLH-ECI   : DCM from ECI to body
#   q0           : quaternion [q1, q2, q3, q4] body ← ECI
#   euler0       : 3-2-1 Euler angles [roll, pitch, yaw] rad
# ══════════════════════════════════════════════════════════════
q0, euler0, C_BI = SO.initial_attitude_from_lvlh(r0, v0)

print(f"\n  Initial attitude  (body = LVLH)")
print(f"      C_LVLH-ECI =")
for row in C_BI:
    print(f"          [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")
print(f"      q0  [q1,q2,q3,q4] = {np.round(q0, 6)}")
print(f"      Euler [φ,θ,ψ] deg = {np.round(np.degrees(euler0), 4)}")

# ══════════════════════════════════════════════════════════════
# 4. Initial Angular Velocity (normal-operations phase)
# ══════════════════════════════════════════════════════════════
omega0 = np.array([0.001, -0.001, 0.002])   # [rad/s]

print(f"\n  ω₀ = {omega0} rad/s")

# Full attitude state
x0_att = np.hstack([omega0, q0])

# ══════════════════════════════════════════════════════════════
# 5. Time vector — one orbital period
# ══════════════════════════════════════════════════════════════
N_PTS = 4000
t_eval = np.linspace(0.0, T_orbit, N_PTS)

# ══════════════════════════════════════════════════════════════
# 6. Simulate attitude (control.NonlinearIOSystem)
# ══════════════════════════════════════════════════════════════
print("\n  Simulating attitude …", end=" ", flush=True)
t_att, omega_h, q_h, euler_h = sc.simulate_attitude(
    t_eval,
    x0_att,
    T_ext=None,   # torque-free
    integrator_opts={
        "solve_ivp_method": "RK45",
        "solve_ivp_kwargs": {"rtol": 1e-10, "atol": 1e-12},
    },
)
print("done.")

# ══════════════════════════════════════════════════════════════
# 7. Simulate orbit (control.NonlinearIOSystem)
# ══════════════════════════════════════════════════════════════
print("  Simulating orbit …", end=" ", flush=True)
t_orb, r_h, v_h = sc.simulate_orbit(
    t_eval,
    r0,
    v0,
    mu=MU,
    integrator_opts={
        "solve_ivp_method": "RK45",
        "solve_ivp_kwargs": {"rtol": 1e-11, "atol": 1e-13},
    },
)
print("done.\n")

# ══════════════════════════════════════════════════════════════
# 8. Conservation diagnostics
# ══════════════════════════════════════════════════════════════
H_mag = SO.angular_momentum(J, omega_h)
q_err = SO.quaternion_norm_error(q_h)
r_mag0 = np.linalg.norm(r0)
r_err = np.linalg.norm(r_h, axis=0) - r_mag0

print("  Conservation check")
print(f"      |H| drift (max)       : {np.max(np.abs(H_mag - H_mag[0])):.2e} kg·m²/s")
print(f"      Quaternion norm error : {np.max(np.abs(q_err)):.2e}")
print(f"      Orbital radius drift  : {np.max(np.abs(r_err)):.2f} m")

# ══════════════════════════════════════════════════════════════
# 9. Plots — formatted to match reference deliverable
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

# ── Figure 1 — main deliverable plot ─────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig.suptitle("Body to ECI Dynamics and Kinematics", fontsize=13, fontweight="bold")

# Panel 1 — Angular Velocities [rad/s]
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

# Reorder for plotting: scalar-first display
q_plot = np.vstack([q_h[3], q_h[0], q_h[1], q_h[2]])

ax.plot(t_att, q_plot[0], color=C1, label=r"$q_1$")
ax.plot(t_att, q_plot[1], color=C2, label=r"$q_2$")
ax.plot(t_att, q_plot[2], color=C3, label=r"$q_3$")
ax.plot(t_att, q_plot[3], color=C4, label=r"$q_4$")
ax.set_ylabel("Quaternion Parameter")
ax.set_ylim(-1.05, 1.05)
ax.legend(loc="upper right", framealpha=0.8)

# Panel 3 — Euler Angles [deg]
euler_unwrap = np.unwrap(euler_h, axis=1)
ax = axes[2]
ax.set_title("Euler Angles")
ax.plot(t_att, np.degrees(euler_unwrap[0]), color=C1, label=r"$\phi$   (roll)")
ax.plot(t_att, np.degrees(euler_unwrap[1]), color=C2, label=r"$\theta$ (pitch)")
ax.plot(t_att, np.degrees(euler_unwrap[2]), color=C3, label=r"$\psi$  (yaw)")
ax.set_ylabel("Angle (deg)")
ax.set_xlabel("time (seconds)")
ax.legend(loc="upper right", framealpha=0.8)

fig.tight_layout()

# ── Figure 2 — conservation check ────────────────────────────
fig2, axes2 = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
fig2.suptitle(f"{sc.name}  —  Conservation & Integration Diagnostics", fontsize=12)

axes2[0].plot(t_att, H_mag, color=C1)
axes2[0].set_ylabel(r"|$\mathbf{H}$|  [kg·m²/s]")
axes2[0].set_title("Angular Momentum Magnitude  (constant for torque-free)")

axes2[1].plot(t_att, q_err, color=C2)
axes2[1].set_ylabel(r"|$\mathbf{q}$| − 1")
axes2[1].set_title("Quaternion Norm Error")

axes2[2].plot(t_orb, r_err, color=C3)
axes2[2].set_ylabel("Δr  [m]")
axes2[2].set_title("Orbital Radius Deviation")
axes2[2].set_xlabel("time (seconds)")

fig2.tight_layout()

# ── Figure 3 — 3-D orbit ─────────────────────────────────────
fig3 = plt.figure(figsize=(8, 7))
ax3 = fig3.add_subplot(111, projection="3d")
ax3.plot(r_h[0] / 1e3, r_h[1] / 1e3, r_h[2] / 1e3, color=C1, lw=1.2, label="Orbit track")
ax3.scatter(*r0 / 1e3, color="lime", s=60, zorder=5, label="t = 0")
ax3.set_xlabel("X [km]")
ax3.set_ylabel("Y [km]")
ax3.set_zlabel("Z [km]")
ax3.set_title(f"{sc.name}  —  ECI Orbit (one period)")

u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
ax3.plot_surface(
    (R_EARTH / 1e3) * np.cos(u) * np.sin(v),
    (R_EARTH / 1e3) * np.sin(u) * np.sin(v),
    (R_EARTH / 1e3) * np.cos(v),
    color="steelblue",
    alpha=0.15,
)
ax3.legend()
fig3.tight_layout()

plt.show()

# ══════════════════════════════════════════════════════════════
# 10. Summary print
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