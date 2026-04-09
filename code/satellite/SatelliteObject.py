import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SatelliteObject:
    """
    Spacecraft mass-properties, orbital mechanics, and attitude dynamics object.

    State vector convention:  x = [omega (3), q (4)]
        omega : angular velocity in body frame   [rad/s]
        q     : quaternion [q1, q2, q3, q4]  (scalar LAST, Hamilton)

    LVLH Frame Convention  (orbit-normal / along-track / nadir)
    ─────────────────────────────────────────────────────────────
        x̂_LVLH = ĥ          orbit normal  (h = r × v)
        ŷ_LVLH = (−r̂) × ĥ  along-track  (= v̂ for circular orbit)
        ẑ_LVLH = −r̂          nadir        (toward Earth centre)

    This matches C_LVLH-ECI from the course partial solution:
        C = [[ 0  −sin(i)  cos(i)]
             [ 0   cos(i)  sin(i)]
             [−1   0       0     ]]
    for a spacecraft at the ascending node with inclination i, RAAN = 0.
    """

    components: list = field(default_factory=list)
    name: str = "Unnamed Satellite"

    # =========================================================
    # Construction
    # =========================================================

    def __init__(self, name=None, json_file=None):
        self.components = []
        self.name = "Unnamed Satellite"
        if json_file is not None and name is not None:
            raise ValueError("Provide either 'json_file' or 'name', not both.")
        if name is not None:
            json_file = self._get_satellite_json_path(name)
        if json_file is not None:
            self.load_from_json(json_file)

    # ─── path helpers ────────────────────────────────────────
    @staticmethod
    def _project_root():
        """
        project_root/
            code/satellite/SatelliteObject.py
            data/satellites/
        """
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _satellites_dir(cls):
        return cls._project_root() / "data" / "satellites"

    @classmethod
    def _get_satellite_json_path(cls, name):
        path = cls._satellites_dir() / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Satellite JSON not found: {path}")
        return path

    # ─── load / add components ───────────────────────────────
    def load_from_json(self, json_file):
        json_path = Path(json_file)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.name = data.get("name", "Unnamed Satellite")
        self.components = []
        if "components" not in data:
            raise ValueError("JSON file must contain a 'components' list.")
        for comp in data["components"]:
            if comp["type"] == "rect_prism":
                self.add_rect_prism(
                    mass=comp["mass"],
                    dims=comp["dims"],
                    position=comp["position"],
                )
            else:
                raise ValueError(f"Unsupported component type: {comp['type']}")

    def add_rect_prism(self, mass, dims, position):
        self.components.append({
            "mass": float(mass),
            "dims": np.array(dims,     dtype=float),
            "pos":  np.array(position, dtype=float),
        })

    # =========================================================
    # Mass Properties
    # =========================================================

    def total_mass(self):
        if not self.components:
            raise ValueError("No components found.")
        return sum(c["mass"] for c in self.components)

    def center_of_mass(self):
        if not self.components:
            raise ValueError("No components found.")
        m_tot = self.total_mass()
        if m_tot == 0:
            raise ValueError("Total mass is zero.")
        return sum(c["mass"] * c["pos"] for c in self.components) / m_tot

    @staticmethod
    def inertia_rect_prism(mass, dims):
        x, y, z = dims
        return np.diag([
            (1/12) * mass * (y**2 + z**2),
            (1/12) * mass * (x**2 + z**2),
            (1/12) * mass * (x**2 + y**2),
        ])

    @staticmethod
    def parallel_axis(I_cm, mass, r):
        r = np.asarray(r, dtype=float).reshape(3, 1)
        return I_cm + mass * ((r.T @ r)[0, 0] * np.eye(3) - r @ r.T)

    def inertia_tensor(self):
        if not self.components:
            raise ValueError("No components found.")
        com = self.center_of_mass()
        I_total = np.zeros((3, 3))
        for c in self.components:
            I_cm = self.inertia_rect_prism(c["mass"], c["dims"])
            I_total += self.parallel_axis(I_cm, c["mass"], c["pos"] - com)
        return I_total

    @property
    def com(self):       return self.center_of_mass()
    @property
    def totalMass(self): return self.total_mass()
    @property
    def J(self):         return self.inertia_tensor()

    # =========================================================
    # Mathematical Utilities
    # =========================================================

    @staticmethod
    def skew(v):
        """3×3 skew-symmetric matrix: skew(a) @ b  ≡  a × b."""
        v = np.asarray(v, dtype=float)
        return np.array([
            [ 0.0,  -v[2],  v[1]],
            [ v[2],  0.0,  -v[0]],
            [-v[1],  v[0],  0.0 ],
        ])

    @staticmethod
    def xi_matrix(q):
        """
        4×3 kinematic matrix Ξ(q) for quaternion propagation.

            q̇ = ½ Ξ(q) ω

        q = [q1, q2, q3, q4]  (scalar last, Hamilton convention).

        Reference: Schaub & Junkins, Eq. 3.106.
        """
        q1, q2, q3, q4 = q
        return np.array([
            [ q4, -q3,  q2],
            [ q3,  q4, -q1],
            [-q2,  q1,  q4],
            [-q1, -q2, -q3],
        ])

    # =========================================================
    # Attitude Representation Conversions
    # =========================================================

    @staticmethod
    def dcm_to_quaternion(C):
        """
        DCM → quaternion [q1, q2, q3, q4] (scalar last).

        Uses the symmetric K-matrix / eigenvalue method (Shepperd).
        Numerically stable for all orientations, including near-180°.

        Reference: Markley & Crassidis, §2.5.
        """
        K = np.array([
            [C[0,0]-C[1,1]-C[2,2], C[0,1]+C[1,0], C[0,2]+C[2,0], C[2,1]-C[1,2]],
            [C[0,1]+C[1,0], C[1,1]-C[0,0]-C[2,2], C[1,2]+C[2,1], C[0,2]-C[2,0]],
            [C[0,2]+C[2,0], C[1,2]+C[2,1], C[2,2]-C[0,0]-C[1,1], C[1,0]-C[0,1]],
            [C[2,1]-C[1,2], C[0,2]-C[2,0], C[1,0]-C[0,1], np.trace(C)         ],
        ]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        q = eigvecs[:, np.argmax(eigvals)]   # [q1, q2, q3, q4]
        return q if q[3] >= 0 else -q         # canonical: scalar ≥ 0

    @staticmethod
    def quaternion_to_dcm(q):
        """
        Quaternion [q1, q2, q3, q4] (scalar last) → DCM C.

        v_body = C @ v_inertial  when q describes inertial→body rotation.

        Reference: Schaub & Junkins, Eq. 3.58.
        """
        q1, q2, q3, q4 = q
        return np.array([
            [q4**2+q1**2-q2**2-q3**2, 2*(q1*q2+q3*q4),           2*(q1*q3-q2*q4)         ],
            [2*(q1*q2-q3*q4),          q4**2-q1**2+q2**2-q3**2,   2*(q2*q3+q1*q4)         ],
            [2*(q1*q3+q2*q4),          2*(q2*q3-q1*q4),           q4**2-q1**2-q2**2+q3**2 ],
        ])

    @staticmethod
    def dcm_to_euler321(C):
        """
        DCM → 3-2-1 Euler angles [φ, θ, ψ] = [roll, pitch, yaw] in radians.

        Singularity at θ = ±90°.
        """
        theta = np.arcsin(np.clip(-C[2, 0], -1.0, 1.0))
        psi   = np.arctan2(C[1, 0], C[0, 0])
        phi   = np.arctan2(C[2, 1], C[2, 2])
        return np.array([phi, theta, psi])

    @staticmethod
    def euler321_to_dcm(phi, theta, psi):
        """3-2-1 [roll, pitch, yaw] → DCM."""
        c1, s1 = np.cos(phi),   np.sin(phi)
        c2, s2 = np.cos(theta), np.sin(theta)
        c3, s3 = np.cos(psi),   np.sin(psi)
        return np.array([
            [ c2*c3,           c2*s3,          -s2   ],
            [ s1*s2*c3-c1*s3,  s1*s2*s3+c1*c3, s1*c2],
            [ c1*s2*c3+s1*s3,  c1*s2*s3-s1*c3, c1*c2],
        ])

    # =========================================================
    # Orbital Mechanics
    # =========================================================

    @staticmethod
    def orbital_elements(r_vec, v_vec, mu=3.986004418e14):
        """
        Classical orbital elements from Cartesian state.

        Returns dict: a [m], e, i [rad], T_orbit [s], n [rad/s], h_vec, h.
        """
        r_vec = np.asarray(r_vec, dtype=float)
        v_vec = np.asarray(v_vec, dtype=float)
        r = np.linalg.norm(r_vec);  v = np.linalg.norm(v_vec)
        h_vec = np.cross(r_vec, v_vec);  h = np.linalg.norm(h_vec)
        e_vec = (1/mu) * ((v**2 - mu/r)*r_vec - np.dot(r_vec, v_vec)*v_vec)
        e     = np.linalg.norm(e_vec)
        a     = 1.0 / (2.0/r - v**2/mu)
        T     = 2.0 * np.pi * np.sqrt(a**3 / mu)
        n     = 2.0 * np.pi / T
        i     = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
        return {"a": a, "e": e, "i": i, "T_orbit": T, "n": n,
                "h_vec": h_vec, "h": h}

    @staticmethod
    def lvlh_dcm(r_vec, v_vec):
        """
        DCM from ECI to LVLH:  v_LVLH = C_LVLH-ECI @ v_ECI.

        Axis layout
        ───────────
            x̂ = ĥ = (r×v)/|r×v|     orbit normal
            ŷ = (−r̂) × ĥ             along-track  (= v̂ for circular)
            ẑ = −r̂                    nadir

        For ascending node at +X with inclination i (RAAN = 0):
            C = [[ 0  −sin i   cos i]   row 0: orbit normal in ECI
                 [ 0   cos i   sin i]   row 1: along-track in ECI
                 [−1   0       0    ]]  row 2: nadir in ECI

        This exactly reproduces the course partial solution C_LVLH-ECI.
        """
        r_vec = np.asarray(r_vec, dtype=float)
        v_vec = np.asarray(v_vec, dtype=float)

        r_hat = r_vec / np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_hat = h_vec / np.linalg.norm(h_vec)   # orbit normal → x̂
        nadir = -r_hat                            # nadir        → ẑ
        t_hat = np.cross(nadir, h_hat)           # along-track  → ŷ

        return np.vstack([h_hat, t_hat, nadir])  # rows are LVLH axes in ECI

    @staticmethod
    def initial_attitude_from_lvlh(r_vec, v_vec):
        """
        Initial quaternion and Euler angles when body frame ≡ LVLH.

        Returns
        -------
        q0     : (4,)    [q1,q2,q3,q4]  body ← ECI
        euler0 : (3,)    [roll, pitch, yaw]  rad
        C_BI   : (3,3)   DCM  body ← ECI  (= C_LVLH-ECI)
        """
        C_BI   = SatelliteObject.lvlh_dcm(r_vec, v_vec)
        q0     = SatelliteObject.dcm_to_quaternion(C_BI)
        euler0 = SatelliteObject.dcm_to_euler321(C_BI)
        return q0, euler0, C_BI

    # =========================================================
    # Equations of Motion
    # =========================================================

    @staticmethod
    def attitude_eom(t, x, J, T_ext=None):
        """
        Torque-free (or forced) rigid-body attitude EOM.

        State x = [ω(3), q(4)].

        Euler:       J ω̇ = −ω × (Jω) + T_ext
        Kinematics:  q̇  = ½ Ξ(q) ω
        """
        omega   = x[:3]
        q       = x[3:7] / np.linalg.norm(x[3:7])
        T_ext   = np.zeros(3) if T_ext is None else np.asarray(T_ext)
        omega_dot = np.linalg.solve(J, T_ext - np.cross(omega, J @ omega))
        q_dot     = 0.5 * SatelliteObject.xi_matrix(q) @ omega
        return np.hstack([omega_dot, q_dot])

    @staticmethod
    def two_body_eom(t, x, mu=3.986004418e14):
        """Keplerian EOM.  State x = [r(3), v(3)]."""
        r_vec = x[:3]; v_vec = x[3:6]
        return np.hstack([v_vec, -(mu / np.linalg.norm(r_vec)**3) * r_vec])

    # =========================================================
    # Control-Library Systems  (Simulink analogue)
    # =========================================================

    def build_attitude_system(self, T_ext=None):
        """
        Wrap attitude EOM as control.NonlinearIOSystem.

        States 7: [ω(3), q(4)]   |   Inputs 3: torque [N·m]
        Outputs 7: full state
        """
        import control as ct
        J_mat = self.J.copy()
        _T    = None if T_ext is None else np.asarray(T_ext, dtype=float)

        def _upd(t, x, u, params):
            omega  = x[:3]
            q      = x[3:7] / np.linalg.norm(x[3:7])
            torque = u[:3] if _T is None else _T
            omega_dot = np.linalg.solve(
                params["J"], torque - np.cross(omega, params["J"] @ omega))
            q_dot = 0.5 * SatelliteObject.xi_matrix(q) @ omega
            return np.hstack([omega_dot, q_dot])

        def _out(t, x, u, params):
            return x.copy()

        return ct.NonlinearIOSystem(
            _upd, _out, states=7, inputs=3, outputs=7,
            name=f"{self.name}_attitude", params={"J": J_mat})

    def build_orbit_system(self, mu=3.986004418e14):
        """
        Wrap two-body EOM as control.NonlinearIOSystem.

        States 6: [r(3), v(3)]   |   Inputs 1: dummy (autonomous)
        Outputs 6: full state
        """
        import control as ct
        _mu = float(mu)

        def _upd(t, x, u, params):
            r_vec = x[:3]; v_vec = x[3:6]
            return np.hstack([v_vec,
                              -(params["mu"]/np.linalg.norm(r_vec)**3)*r_vec])

        def _out(t, x, u, params):
            return x.copy()

        return ct.NonlinearIOSystem(
            _upd, _out, states=6, inputs=1, outputs=6,
            name=f"{self.name}_orbit", params={"mu": _mu})

    def simulate_attitude(self, t_eval, x0_attitude, T_ext=None, integrator_opts=None):
        """
        Simulate attitude dynamics via control.input_output_response.

        Returns t (N,), omega (3,N), q (4,N), euler (3,N).
        """
        import control as ct

        t_eval = np.asarray(t_eval, dtype=float)
        x0 = np.asarray(x0_attitude, dtype=float)

        # input_output_response expects solver settings through these names:
        #   solve_ivp_method
        #   solve_ivp_kwargs
        default_opts = {
            "solve_ivp_method": "RK45",
            "solve_ivp_kwargs": {"rtol": 1e-10, "atol": 1e-12},
        }
        opts = default_opts if integrator_opts is None else integrator_opts

        sys = self.build_attitude_system(T_ext=T_ext)

        torq = np.zeros(3) if T_ext is None else np.asarray(T_ext, dtype=float)
        U = np.tile(torq.reshape(3, 1), (1, len(t_eval)))

        # Since your system output is the full state, y is the state history
        t, y = ct.input_output_response(sys, t_eval, U, x0, **opts)

        omega = y[:3, :]
        q = y[3:7, :]

        euler = np.zeros((3, len(t)))
        for i in range(len(t)):
            q[:, i] /= np.linalg.norm(q[:, i])
            euler[:, i] = self.dcm_to_euler321(self.quaternion_to_dcm(q[:, i]))

        return t, omega, q, euler


    def simulate_orbit(self, t_eval, r0, v0, mu=3.986004418e14, integrator_opts=None):
        """
        Simulate Keplerian orbit via control.input_output_response.

        Returns t (N,), r (3,N) [m], v (3,N) [m/s].
        """
        import control as ct

        t_eval = np.asarray(t_eval, dtype=float)
        x0 = np.hstack([r0, v0])

        default_opts = {
            "solve_ivp_method": "RK45",
            "solve_ivp_kwargs": {"rtol": 1e-11, "atol": 1e-13},
        }
        opts = default_opts if integrator_opts is None else integrator_opts

        sys = self.build_orbit_system(mu=mu)

        # dummy input for the autonomous system
        U = 0.0

        t, y = ct.input_output_response(sys, t_eval, U, x0, **opts)
        return t, y[:3, :], y[3:6, :]

    # =========================================================
    # Analysis Helpers
    # =========================================================

    @staticmethod
    def angular_momentum(J, omega_hist):
        """|Jω| over time — conserved for torque-free motion."""
        return np.linalg.norm(J @ omega_hist, axis=0)

    @staticmethod
    def quaternion_norm_error(q_hist):
        """Deviation of |q| from unity — integration quality metric."""
        return np.linalg.norm(q_hist, axis=0) - 1.0