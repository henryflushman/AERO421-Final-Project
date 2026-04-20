"""
ADCS.py — Attitude Determination and Control System Utilities
=============================================================
A self-contained library of helper functions and classes for spacecraft
attitude dynamics, kinematics, and control.

Contents
--------
1.  Rotation Matrices         – elementary rotations R1, R2, R3
2.  DCM                       – Direction Cosine Matrix class (any Euler sequence)
3.  Quaternion                – Hamilton quaternion class (passive/body←inertial convention)
4.  MRP                       – Modified Rodrigues Parameters class
5.  Euler Angle Utilities     – conversions between representations
6.  Kinematics                – q̇, ω, MRP rate equations
7.  RigidBody                 – inertia tensor, torque-free Euler equations
8.  Attitude Controllers      – PD and PID on quaternion error
9.  Attitude Determination    – TRIAD, q-method / QUEST (simplified)
10. Orbital Mechanics Helpers – orbital frame DCM (RTN/LVLH)
11. Utility / Math            – skew-symmetric cross product, angle wrapping, etc.

Conventions
-----------
* Passive (alias) rotations: C rotates the *frame*, so v_B = C_BN · v_N.
* Quaternion: q = [q1, q2, q3, q4] = [vec, scalar] (q4 is the scalar part).
* All angles in **radians** unless explicitly noted.
* SI units throughout (kg, m, s, N, N·m).

Dependencies: numpy, scipy (optional, used only in integrate helpers)
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Elementary Rotation Matrices
# ═══════════════════════════════════════════════════════════════════════════════

def R1(angle: float) -> ndarray:
    """Passive rotation about axis 1 (x) by *angle* radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1,  0,  0],
        [0,  c,  s],
        [0, -s,  c],
    ])


def R2(angle: float) -> ndarray:
    """Passive rotation about axis 2 (y) by *angle* radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c,  0, -s],
        [0,  1,  0],
        [s,  0,  c],
    ])


def R3(angle: float) -> ndarray:
    """Passive rotation about axis 3 (z) by *angle* radians."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [ c,  s,  0],
        [-s,  c,  0],
        [ 0,  0,  1],
    ])


_ELEMENTARY = {1: R1, 2: R2, 3: R3}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Direction Cosine Matrix (DCM)
# ═══════════════════════════════════════════════════════════════════════════════

class DCM:
    """
    Direction Cosine Matrix with support for arbitrary Euler rotation sequences.

    Parameters
    ----------
    angles : sequence of float
        Rotation angles in radians, applied right-to-left (last angle is the
        first physical rotation).
    sequence : sequence of int
        Axis sequence, e.g. (3, 2, 1) for yaw-pitch-roll,
                              (3, 1, 3) for classic Euler angles.
        Must be the same length as *angles*.

    Examples
    --------
    >>> C = DCM([0.1, 0.2, 0.3], sequence=(3, 2, 1))   # 3-2-1 Euler
    >>> C = DCM([np.pi/4, np.pi/6, np.pi/4], sequence=(3, 1, 3))  # symmetric
    >>> C.matrix          # 3×3 numpy array
    >>> C.inv()           # transpose (= inverse for orthogonal matrices)
    """

    def __init__(
        self,
        angles: Sequence[float],
        sequence: Sequence[int] = (3, 2, 1),
    ) -> None:
        angles = list(angles)
        sequence = list(sequence)
        if len(angles) != len(sequence):
            raise ValueError("len(angles) must equal len(sequence).")
        for ax in sequence:
            if ax not in (1, 2, 3):
                raise ValueError(f"Axis {ax} is invalid; must be 1, 2, or 3.")
        self.angles = angles
        self.sequence = sequence
        self._C = self._build()

    # ------------------------------------------------------------------
    def _build(self) -> ndarray:
        C = np.eye(3)
        # Rightmost rotation is applied first → compose left-to-right
        for ax, ang in zip(self.sequence, self.angles):
            C = _ELEMENTARY[ax](ang) @ C
        return C

    @property
    def matrix(self) -> ndarray:
        """Return the underlying 3×3 array."""
        return self._C.copy()

    def inv(self) -> ndarray:
        """Return the inverse DCM (= transpose for proper orthogonal matrix)."""
        return self._C.T.copy()

    def __matmul__(self, other):
        """Allow DCM @ vector or DCM @ ndarray directly."""
        if isinstance(other, DCM):
            return self._C @ other._C
        return self._C @ other

    def __repr__(self) -> str:  # pragma: no cover
        seq = "".join(str(a) for a in self.sequence)
        angs = [f"{np.degrees(a):.2f}°" for a in self.angles]
        return f"DCM(sequence={seq}, angles={angs})\n{np.array2string(self._C, precision=6)}"

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(cls, C: ndarray) -> "DCM":
        """Wrap an existing 3×3 array as a DCM object (sequence stored as None)."""
        obj = object.__new__(cls)
        obj.angles = None
        obj.sequence = None
        obj._C = np.asarray(C, dtype=float)
        return obj

    @classmethod
    def from_quaternion(cls, q: Union[Sequence[float], ndarray]) -> "DCM":
        """
        Build a DCM from a quaternion q = [q1, q2, q3, q4] (scalar last).
        """
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)
        q1, q2, q3, q4 = q
        C = np.array([
            [1 - 2*(q2**2 + q3**2),  2*(q1*q2 + q3*q4),      2*(q1*q3 - q2*q4)],
            [2*(q1*q2 - q3*q4),      1 - 2*(q1**2 + q3**2),  2*(q2*q3 + q1*q4)],
            [2*(q1*q3 + q2*q4),      2*(q2*q3 - q1*q4),      1 - 2*(q1**2 + q2**2)],
        ])
        return cls.from_matrix(C)

    @classmethod
    def from_axis_angle(cls, axis: ndarray, angle: float) -> "DCM":
        """
        Build a DCM from a principal rotation axis (unit vector) and angle.
        Uses Rodrigues' rotation formula.
        """
        e = np.asarray(axis, dtype=float)
        e = e / np.linalg.norm(e)
        c, s = math.cos(angle), math.sin(angle)
        E = skew(e)
        C = c * np.eye(3) + (1 - c) * np.outer(e, e) - s * E
        return cls.from_matrix(C)

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def to_euler(self, sequence: Sequence[int] = (3, 2, 1)) -> ndarray:
        """
        Extract Euler angles for the given axis sequence from this DCM.
        Supports any non-degenerate Tait-Bryan or classic Euler sequence.

        Returns
        -------
        ndarray of shape (3,) with angles in radians.
        """
        sequence = tuple(sequence)
        C = self._C
        # ── Tait-Bryan sequences (all axes different) ──────────────────
        if sequence == (3, 2, 1):
            yaw   = math.atan2(C[0, 1], C[0, 0])
            pitch = math.asin(-C[0, 2])
            roll  = math.atan2(C[1, 2], C[2, 2])
            return np.array([yaw, pitch, roll])
        if sequence == (1, 2, 3):
            roll  = math.atan2(-C[1, 2], C[2, 2])
            pitch = math.asin(C[0, 2])
            yaw   = math.atan2(-C[0, 1], C[0, 0])
            return np.array([roll, pitch, yaw])
        if sequence == (3, 1, 2):
            a1 = math.atan2(-C[1, 0], C[1, 1])
            a2 = math.asin(C[1, 2])
            a3 = math.atan2(-C[0, 2], C[2, 2])
            return np.array([a1, a2, a3])
        if sequence == (2, 1, 3):
            a1 = math.atan2(C[2, 0], C[0, 0])
            a2 = math.asin(-C[1, 0])
            a3 = math.atan2(C[1, 2], C[1, 1])
            return np.array([a1, a2, a3])
        # ── Classic Euler sequences (first and last axis same) ─────────
        if sequence == (3, 1, 3):
            a1 = math.atan2(C[2, 0], -C[2, 1])
            a2 = math.acos(np.clip(C[2, 2], -1.0, 1.0))
            a3 = math.atan2(C[0, 2], C[1, 2])
            return np.array([a1, a2, a3])
        if sequence == (3, 2, 3):
            a1 = math.atan2(C[2, 1], C[2, 0])
            a2 = math.acos(np.clip(C[2, 2], -1.0, 1.0))
            a3 = math.atan2(C[1, 2], -C[0, 2])
            return np.array([a1, a2, a3])
        if sequence == (1, 3, 1):
            a1 = math.atan2(C[0, 2], C[0, 1])
            a2 = math.acos(np.clip(C[0, 0], -1.0, 1.0))
            a3 = math.atan2(C[2, 0], -C[1, 0])
            return np.array([a1, a2, a3])
        raise NotImplementedError(f"Euler sequence {sequence} not yet implemented. "
                                  "Use DCM.from_matrix and manual decomposition.")

    def to_quaternion(self) -> ndarray:
        """Return quaternion q = [q1, q2, q3, q4] (scalar last) from this DCM."""
        return dcm_to_quaternion(self._C)

    def to_axis_angle(self) -> Tuple[ndarray, float]:
        """Return (unit axis, angle) principal rotation representation."""
        C = self._C
        angle = math.acos(np.clip((np.trace(C) - 1) / 2, -1.0, 1.0))
        if abs(angle) < 1e-10:
            return np.array([0.0, 0.0, 1.0]), 0.0
        axis = np.array([C[1, 2] - C[2, 1],
                         C[2, 0] - C[0, 2],
                         C[0, 1] - C[1, 0]]) / (2 * math.sin(angle))
        return axis / np.linalg.norm(axis), angle


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Quaternion
# ═══════════════════════════════════════════════════════════════════════════════

class Quaternion:
    """
    Hamilton unit quaternion  q = [q⃗, q4] = [q1, q2, q3, q4].

    q4 is the **scalar** part (some texts use q0; we use q4 for compatibility
    with Schaub & Junkins notation).

    Convention: passive rotation — q represents the attitude of frame B
    relative to frame N, so v_B = C(q) · v_N.
    """

    def __init__(self, q: Union[Sequence[float], ndarray]) -> None:
        self._q = np.asarray(q, dtype=float).ravel()
        if self._q.shape != (4,):
            raise ValueError("Quaternion requires a 4-element array.")
        norm = np.linalg.norm(self._q)
        if norm < 1e-12:
            raise ValueError("Quaternion has zero norm.")
        self._q /= norm

    # ------------------------------------------------------------------
    @property
    def vec(self) -> ndarray:
        """Return q⃗ = [q1, q2, q3]."""
        return self._q[:3].copy()

    @property
    def scalar(self) -> float:
        """Return q4 (scalar part)."""
        return float(self._q[3])

    @property
    def q(self) -> ndarray:
        """Full quaternion array [q1, q2, q3, q4]."""
        return self._q.copy()

    # ------------------------------------------------------------------
    def conjugate(self) -> "Quaternion":
        """Return q* = [-q⃗, q4] (inverse for unit quaternion)."""
        conj: ndarray = np.append(-self._q[:3], self._q[3])
        return Quaternion(conj)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """
        Quaternion multiplication  q_AB = q_AN * q_NB.
        (Composes rotations in that order.)
        """
        q1, q2 = self._q, other._q
        v1, s1 = q1[:3], q1[3]
        v2, s2 = q2[:3], q2[3]
        vec = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        scalar = s1 * s2 - np.dot(v1, v2)
        result: ndarray = np.append(vec, scalar)
        return Quaternion(result)

    def rotate(self, v: ndarray) -> ndarray:
        """Rotate vector v from N frame to B frame: v_B = C(q) · v_N."""
        return self.dcm @ np.asarray(v, dtype=float)

    @property
    def dcm(self) -> ndarray:
        """Return the 3×3 DCM corresponding to this quaternion."""
        return DCM.from_quaternion(self._q).matrix

    def error(self, q_ref: "Quaternion") -> "Quaternion":
        """
        Compute attitude error quaternion δq = q_ref⁻¹ ⊗ q_current.
        A zero-error attitude gives δq = [0, 0, 0, 1].
        """
        return q_ref.conjugate() * self

    # ------------------------------------------------------------------
    @classmethod
    def from_axis_angle(cls, axis: Union[Sequence[float], ndarray], angle: float) -> "Quaternion":
        """Build quaternion from principal axis (unit vec) and rotation angle."""
        e = np.asarray(axis, dtype=float).ravel()
        e = e / np.linalg.norm(e)
        half = angle / 2
        q_arr: ndarray = np.append(math.sin(half) * e, math.cos(half))
        return cls(q_arr)

    @classmethod
    def from_dcm(cls, C: Union[Sequence[Sequence[float]], ndarray]) -> "Quaternion":
        """Build quaternion from a 3×3 DCM."""
        q_arr: ndarray = dcm_to_quaternion(np.asarray(C, dtype=float))
        return cls(q_arr)

    @classmethod
    def from_euler(cls, angles: Sequence[float],
                   sequence: Sequence[int] = (3, 2, 1)) -> "Quaternion":
        """Build quaternion from Euler angles via DCM."""
        C = DCM(angles, sequence)
        return cls.from_dcm(C.matrix)

    @classmethod
    def identity(cls) -> "Quaternion":
        """Return identity quaternion [0, 0, 0, 1]."""
        return cls([0.0, 0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        q = self._q
        return (f"Quaternion([{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}], "
                f"scalar={q[3]:.6f})")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Modified Rodrigues Parameters (MRP)
# ═══════════════════════════════════════════════════════════════════════════════

class MRP:
    """
    Modified Rodrigues Parameters  σ = e·tan(Φ/4).

    Singular at Φ = 2π; use shadow set σ_s = -σ / |σ|² when |σ| > 1 to
    avoid singularity.
    """

    def __init__(self, sigma: Union[Sequence[float], ndarray], auto_shadow: bool = True) -> None:
        self._s = np.asarray(sigma, dtype=float).ravel()
        if self._s.shape != (3,):
            raise ValueError("MRP must be a 3-element array.")
        if auto_shadow and np.dot(self._s, self._s) > 1.0:
            self._s = self.shadow(self._s)

    @staticmethod
    def shadow(sigma: ndarray) -> ndarray:
        """Return shadow MRP set: σ_s = -σ / |σ|²."""
        norm_sq = np.dot(sigma, sigma)
        if norm_sq < 1e-20:
            return sigma.copy()
        return -sigma / norm_sq

    @property
    def sigma(self) -> ndarray:
        return self._s.copy()

    def to_quaternion(self) -> Quaternion:
        """Convert MRP to unit quaternion."""
        s = self._s
        norm_sq = np.dot(s, s)
        q_vec: ndarray = 2 * s / (1 + norm_sq)
        q_scalar = float((1 - norm_sq) / (1 + norm_sq))
        q_arr: ndarray = np.append(q_vec, q_scalar)
        return Quaternion(q_arr)

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> "MRP":
        """Convert unit quaternion to MRP."""
        qv, qs = q.vec, q.scalar
        if qs + 1 < 1e-10:
            raise ValueError("Quaternion at MRP singularity (Φ ≈ 2π).")
        sigma: ndarray = qv / (1 + qs)
        return cls(sigma)

    def dcm(self) -> ndarray:
        """Return DCM from this MRP."""
        return self.to_quaternion().dcm

    def error(self, sigma_ref: "MRP") -> "MRP":
        """
        Compute MRP error:  δσ = σ ⊗ σ_ref⁻¹  (using MRP addition formula).
        """
        s = self._s
        r = sigma_ref.sigma
        num = (1 - np.dot(r, r)) * s - (1 - np.dot(s, s)) * r + 2 * np.cross(s, r)
        den = 1 + np.dot(r, r) * np.dot(s, s) + 2 * np.dot(s, r)
        if abs(den) < 1e-12:
            warnings.warn("MRP error denominator near zero; possible singularity.")
        err_sigma: ndarray = num / den
        return MRP(err_sigma)

    def B_matrix(self) -> ndarray:
        """
        Kinematic mapping B(σ) such that σ̇ = ¼ B(σ) ω.
        Shape: (3, 3).
        """
        s = self._s
        S = skew(s)
        norm_sq = np.dot(s, s)
        return (1 - norm_sq) * np.eye(3) + 2 * S + 2 * np.outer(s, s)

    def __repr__(self) -> str:
        s = self._s
        return f"MRP([{s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f}])"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Euler Angle Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def euler_to_dcm(angles: Sequence[float],
                 sequence: Sequence[int] = (3, 2, 1)) -> ndarray:
    """Build a DCM from Euler angles. Shortcut for ``DCM(angles, sequence).matrix``."""
    return DCM(angles, sequence).matrix


def dcm_to_euler(C: ndarray,
                 sequence: Sequence[int] = (3, 2, 1)) -> ndarray:
    """Extract Euler angles from a DCM for the given rotation sequence."""
    return DCM.from_matrix(C).to_euler(sequence)


def dcm_to_quaternion(C: ndarray) -> ndarray:
    """
    Shepperd's method: robustly convert a 3×3 DCM to quaternion [q1,q2,q3,q4].
    Avoids division by small numbers by choosing the largest component first.
    """
    C = np.asarray(C, dtype=float)
    trace = np.trace(C)
    candidates = np.array([
        trace,
        C[0, 0],
        C[1, 1],
        C[2, 2],
    ])
    idx = int(np.argmax(candidates))
    q = np.zeros(4)
    if idx == 0:
        q[3] = 0.5 * math.sqrt(1 + trace)
        f = 1 / (4 * q[3])
        q[0] = f * (C[1, 2] - C[2, 1])
        q[1] = f * (C[2, 0] - C[0, 2])
        q[2] = f * (C[0, 1] - C[1, 0])
    elif idx == 1:
        q[0] = 0.5 * math.sqrt(1 + C[0, 0] - C[1, 1] - C[2, 2])
        f = 1 / (4 * q[0])
        q[1] = f * (C[0, 1] + C[1, 0])
        q[2] = f * (C[2, 0] + C[0, 2])
        q[3] = f * (C[1, 2] - C[2, 1])
    elif idx == 2:
        q[1] = 0.5 * math.sqrt(1 - C[0, 0] + C[1, 1] - C[2, 2])
        f = 1 / (4 * q[1])
        q[0] = f * (C[0, 1] + C[1, 0])
        q[2] = f * (C[1, 2] + C[2, 1])
        q[3] = f * (C[2, 0] - C[0, 2])
    else:
        q[2] = 0.5 * math.sqrt(1 - C[0, 0] - C[1, 1] + C[2, 2])
        f = 1 / (4 * q[2])
        q[0] = f * (C[2, 0] + C[0, 2])
        q[1] = f * (C[1, 2] + C[2, 1])
        q[3] = f * (C[0, 1] - C[1, 0])
    return q / np.linalg.norm(q)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def wrap_angle_deg(angle: float) -> float:
    """Wrap angle in degrees to [-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Kinematics
# ═══════════════════════════════════════════════════════════════════════════════

def qdot(q: ndarray, omega: ndarray) -> ndarray:
    """
    Quaternion kinematic equation:  q̇ = ½ Ξ(q) ω

    Parameters
    ----------
    q : array-like, shape (4,)   [q1,q2,q3,q4], scalar last
    omega : array-like, shape (3,)  angular velocity in body frame [rad/s]

    Returns
    -------
    q̇ : ndarray, shape (4,)
    """
    q = np.asarray(q, dtype=float)
    w = np.asarray(omega, dtype=float)
    q1, q2, q3, q4 = q
    Xi = np.array([
        [ q4, -q3,  q2],
        [ q3,  q4, -q1],
        [-q2,  q1,  q4],
        [-q1, -q2, -q3],
    ])
    return 0.5 * (Xi @ w)


def euler_rate_matrix(angles: Sequence[float],
                      sequence: Sequence[int] = (3, 2, 1)) -> ndarray:
    """
    Kinematic mapping [Ṫ₁, Ṫ₂, Ṫ₃]ᵀ = B⁻¹(θ) ω  for Tait-Bryan (3-2-1).

    Returns B(θ) such that ω_body = B(θ) · θ̇.
    Supported sequences: (3,2,1) only (extend as needed).
    """
    if tuple(sequence) != (3, 2, 1):
        raise NotImplementedError("euler_rate_matrix currently supports (3,2,1) only.")
    yaw, pitch, roll = angles
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    # B maps θ̇ → ω_B
    B = np.array([
        [-sp,    0,  1],
        [cp*sr,  cr, 0],
        [cp*cr, -sr, 0],
    ])
    return B


def mrp_dot(sigma: ndarray, omega: ndarray) -> ndarray:
    """
    MRP kinematic equation:  σ̇ = ¼ B(σ) ω

    Parameters
    ----------
    sigma : array-like, shape (3,)
    omega : array-like, shape (3,)  angular velocity in body frame [rad/s]

    Returns
    -------
    σ̇ : ndarray, shape (3,)
    """
    sigma = np.asarray(sigma, dtype=float)
    omega = np.asarray(omega, dtype=float)
    mrp = MRP(sigma, auto_shadow=False)
    return 0.25 * mrp.B_matrix() @ omega


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Rigid Body Dynamics
# ═══════════════════════════════════════════════════════════════════════════════

class RigidBody:
    """
    Rigid spacecraft model with inertia tensor I.

    Parameters
    ----------
    inertia : array-like, shape (3, 3) or (3,) for principal axes
        Inertia tensor in body frame [kg·m²].
    """

    def __init__(self, inertia: Union[ndarray, Sequence[float]]) -> None:
        I = np.asarray(inertia, dtype=float)
        if I.ndim == 1:
            if I.shape != (3,):
                raise ValueError("1-D inertia must have 3 elements (principal moments).")
            I = np.diag(I)
        if I.shape != (3, 3):
            raise ValueError("Inertia must be (3,) or (3,3).")
        self.I = I
        self.I_inv = np.linalg.inv(I)

    # ------------------------------------------------------------------
    def euler_equations(self, omega: ndarray, torque: Optional[ndarray] = None) -> ndarray:
        """
        Euler's rotational equations of motion:
            I ω̇ = τ − ω × (I ω)

        Parameters
        ----------
        omega : array-like, shape (3,)  angular velocity [rad/s]
        torque : array-like, shape (3,) external torque [N·m]; defaults to zero

        Returns
        -------
        ω̇ : ndarray, shape (3,)
        """
        omega = np.asarray(omega, dtype=float)
        if torque is None:
            torque = np.zeros(3)
        torque = np.asarray(torque, dtype=float)
        return self.I_inv @ (torque - np.cross(omega, self.I @ omega))

    def angular_momentum(self, omega: ndarray) -> ndarray:
        """H = I ω  [kg·m²/s]"""
        return self.I @ np.asarray(omega, dtype=float)

    def kinetic_energy(self, omega: ndarray) -> float:
        """T = ½ ωᵀ I ω  [J]"""
        omega = np.asarray(omega, dtype=float)
        return 0.5 * float(omega @ self.I @ omega)

    # ------------------------------------------------------------------
    def torque_free_rhs(self, t: float, state: ndarray) -> ndarray:
        """
        ODE right-hand side for torque-free motion with quaternion kinematics.

        State vector: [q1, q2, q3, q4, ω1, ω2, ω3]   (7 elements)

        Suitable for use with scipy.integrate.solve_ivp.
        """
        q = state[:4]
        omega = state[4:]
        q = q / np.linalg.norm(q)          # re-normalise each call
        dqdt = qdot(q, omega)
        domega_dt = self.euler_equations(omega)
        return np.concatenate([dqdt, domega_dt])

    def simulate(
        self,
        q0: ndarray,
        omega0: ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[ndarray] = None,
        torque_func=None,
        **ivp_kwargs,
    ):
        """
        Integrate the full attitude + angular-velocity equations.

        Parameters
        ----------
        q0       : initial quaternion [q1,q2,q3,q4]
        omega0   : initial angular velocity [rad/s]
        t_span   : (t0, tf) integration interval [s]
        t_eval   : optional time points at which to store output
        torque_func : callable(t, state) → ndarray(3,), default zero torque

        Returns
        -------
        scipy OdeResult with .t, .y
        """
        try:
            from scipy.integrate import solve_ivp
        except ImportError:
            raise ImportError("scipy is required for RigidBody.simulate().")

        if torque_func is None:
            torque_func = lambda t, s: np.zeros(3)

        def rhs(t, state):
            q = state[:4] / np.linalg.norm(state[:4])
            omega = state[4:]
            tau = torque_func(t, state)
            return np.concatenate([qdot(q, omega),
                                   self.euler_equations(omega, tau)])

        state0 = np.concatenate([np.asarray(q0, dtype=float),
                                  np.asarray(omega0, dtype=float)])
        return solve_ivp(rhs, t_span, state0, t_eval=t_eval,
                         dense_output=True, **ivp_kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Attitude Controllers
# ═══════════════════════════════════════════════════════════════════════════════

class PDQuaternionController:
    """
    Proportional–Derivative attitude controller using quaternion error.

    Control law:
        τ = −K_p δq⃗ − K_d ω_error

    where δq⃗ is the vector part of the attitude error quaternion and
    ω_error = ω_body − C(δq) ω_ref.

    Parameters
    ----------
    Kp : float or ndarray(3,3)  proportional gain (scalar → K_p I₃)
    Kd : float or ndarray(3,3)  derivative gain
    """

    def __init__(
        self,
        Kp: Union[float, ndarray],
        Kd: Union[float, ndarray],
    ) -> None:
        self.Kp: ndarray = float(Kp) * np.eye(3) if isinstance(Kp, (int, float)) else np.asarray(Kp, dtype=float)
        self.Kd: ndarray = float(Kd) * np.eye(3) if isinstance(Kd, (int, float)) else np.asarray(Kd, dtype=float)

    def control(
        self,
        q_current: ndarray,
        q_ref: ndarray,
        omega_current: ndarray,
        omega_ref: Optional[ndarray] = None,
    ) -> ndarray:
        """
        Compute control torque.

        Parameters
        ----------
        q_current   : current quaternion [q1,q2,q3,q4]
        q_ref       : reference quaternion [q1,q2,q3,q4]
        omega_current : current body angular velocity [rad/s]
        omega_ref   : reference body angular velocity [rad/s]; default zeros

        Returns
        -------
        torque : ndarray(3,)  [N·m]
        """
        q_c = np.asarray(q_current, dtype=float)
        q_r = np.asarray(q_ref, dtype=float)
        omega_c = np.asarray(omega_current, dtype=float)
        if omega_ref is None:
            omega_ref = np.zeros(3)
        omega_r = np.asarray(omega_ref, dtype=float)

        q_c /= np.linalg.norm(q_c)
        q_r /= np.linalg.norm(q_r)

        # Error quaternion: δq = q_ref* ⊗ q_current
        dq = Quaternion(q_c).error(Quaternion(q_r)).q
        dq_vec = dq[:3]
        dq_scalar = dq[3]

        # Sign correction: ensure short-path rotation
        if dq_scalar < 0:
            dq_vec = -dq_vec

        # Angular rate error
        C_err = DCM.from_quaternion(dq).matrix
        omega_err = omega_c - C_err @ omega_r

        return -self.Kp @ dq_vec - self.Kd @ omega_err


class PIDQuaternionController(PDQuaternionController):
    """
    PID quaternion controller extending PDQuaternionController with an
    integral term on the quaternion vector error.

    Parameters
    ----------
    Kp, Kd, Ki : float or ndarray(3,3)
    windup_limit : float   anti-windup clamp on integral state [rad·s]
    """

    def __init__(
        self,
        Kp: Union[float, ndarray],
        Kd: Union[float, ndarray],
        Ki: Union[float, ndarray],
        windup_limit: float = np.inf,
    ) -> None:
        super().__init__(Kp, Kd)
        self.Ki: ndarray = float(Ki) * np.eye(3) if isinstance(Ki, (int, float)) else np.asarray(Ki, dtype=float)
        self.windup_limit = windup_limit
        self._integral = np.zeros(3)

    def reset(self) -> None:
        """Reset the integral accumulator."""
        self._integral = np.zeros(3)

    def control(
        self,
        q_current: ndarray,
        q_ref: ndarray,
        omega_current: ndarray,
        omega_ref: Optional[ndarray] = None,
        dt: float = 0.01,
    ) -> ndarray:
        """
        Compute PID torque.

        Parameters
        ----------
        dt : float   time step [s] for integration
        (Other parameters identical to PDQuaternionController.control)
        """
        q_c = np.asarray(q_current, dtype=float) / np.linalg.norm(q_current)
        q_r = np.asarray(q_ref, dtype=float) / np.linalg.norm(q_ref)

        dq = Quaternion(q_c).error(Quaternion(q_r)).q
        dq_vec = dq[:3]
        if dq[3] < 0:
            dq_vec = -dq_vec

        self._integral += dq_vec * dt
        # Anti-windup clamp
        int_norm = np.linalg.norm(self._integral)
        if int_norm > self.windup_limit:
            self._integral *= self.windup_limit / int_norm

        pd_torque = super().control(q_current, q_ref, omega_current, omega_ref)
        return pd_torque - self.Ki @ self._integral


def compute_gain_pd(I: ndarray, omega_n: float, zeta: float) -> Tuple[ndarray, ndarray]:
    """
    Analytically compute PD gains for a decoupled linearised system.

    ω_n  – desired closed-loop natural frequency [rad/s]
    ζ    – desired damping ratio

    Returns
    -------
    Kp, Kd : ndarray(3,3)
    """
    I = np.asarray(I, dtype=float)
    if I.ndim == 1:
        I = np.diag(I)
    Kp = omega_n**2 * I
    Kd = 2 * zeta * omega_n * I
    return Kp, Kd


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Attitude Determination
# ═══════════════════════════════════════════════════════════════════════════════

def triad(v1_body: ndarray, v2_body: ndarray,
          v1_ref: ndarray, v2_ref: ndarray) -> ndarray:
    """
    TRIAD algorithm: determine attitude DCM from two vector measurements.

    The first vector pair is treated as the primary (more accurate) observation.

    Parameters
    ----------
    v1_body, v2_body : unit vectors measured in body frame
    v1_ref, v2_ref   : corresponding known vectors in reference frame

    Returns
    -------
    C_BN : 3×3 DCM mapping reference → body frame
    """
    def normalise(v):
        return v / np.linalg.norm(v)

    # Body triad
    t1_b = normalise(np.asarray(v1_body, dtype=float))
    t2_b = normalise(np.cross(v1_body, v2_body))
    t3_b = np.cross(t1_b, t2_b)

    # Reference triad
    t1_r = normalise(np.asarray(v1_ref, dtype=float))
    t2_r = normalise(np.cross(v1_ref, v2_ref))
    t3_r = np.cross(t1_r, t2_r)

    M_b = np.column_stack([t1_b, t2_b, t3_b])
    M_r = np.column_stack([t1_r, t2_r, t3_r])

    return M_b @ M_r.T   # C_BN


def quest(
    body_vectors: List[ndarray],
    ref_vectors: List[ndarray],
    weights: Optional[List[float]] = None,
) -> ndarray:
    """
    QUEST (QUaternion ESTimator) — Wahba's problem solved via the K matrix.

    Parameters
    ----------
    body_vectors : list of N unit vectors in body frame
    ref_vectors  : list of N corresponding unit vectors in reference frame
    weights      : list of N positive weights (default: equal weights)

    Returns
    -------
    q_opt : ndarray(4,)  optimal quaternion [q1,q2,q3,q4], scalar last
    """
    n = len(body_vectors)
    if weights is None:
        weights = [1.0 / n] * n
    weights_arr: ndarray = np.asarray(weights, dtype=float)
    weights_arr /= weights_arr.sum()

    B: ndarray = np.zeros((3, 3))
    for _w, _b, _r in zip(weights_arr, body_vectors, ref_vectors):
        B += _w * np.outer(_b, _r)

    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1, 2] - B[2, 1],
                  B[2, 0] - B[0, 2],
                  B[0, 1] - B[1, 0]])

    K = np.zeros((4, 4))
    K[:3, :3] = S - sigma * np.eye(3)
    K[:3, 3] = Z
    K[3, :3] = Z
    K[3, 3] = sigma

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    # Eigenvector corresponding to largest eigenvalue
    q_opt = eigenvectors[:, np.argmax(eigenvalues)]
    # Rearrange from [q4,q1,q2,q3] to [q1,q2,q3,q4] if needed
    # (np.linalg.eigh returns in ascending order; eigenvector stored as [q_vec; q_scalar])
    return q_opt / np.linalg.norm(q_opt)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Orbital Mechanics Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def orbital_elements_to_state(
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    nu: float,
    mu: float = 3.986004418e14,
) -> Tuple[ndarray, ndarray]:
    """
    Convert classical orbital elements to Cartesian state vectors in ECI.

    Parameters
    ----------
    a    : semi-major axis [m]
    e    : eccentricity
    i    : inclination [rad]
    raan : right ascension of ascending node [rad]
    argp : argument of periapsis [rad]
    nu   : true anomaly [rad]
    mu   : gravitational parameter [m³/s²]

    Returns
    -------
    r_ECI : ndarray(3,)  position [m]
    v_ECI : ndarray(3,)  velocity [m/s]
    """
    p = a * (1 - e**2)
    r_mag = p / (1 + e * math.cos(nu))

    # Perifocal frame
    r_pf = r_mag * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    # Rotation: perifocal → ECI
    C = DCM([raan, i, argp], sequence=(3, 1, 3)).inv()
    return C @ r_pf, C @ v_pf


def lvlh_dcm(r_ECI: ndarray, v_ECI: ndarray) -> ndarray:
    """
    Compute DCM rotating ECI → LVLH (Local Vertical Local Horizontal) frame.

    LVLH axes:
        x̂_LVLH – along radial (toward nadir in some conventions; here radially outward)
        ŷ_LVLH – along-track (roughly velocity direction)
        ẑ_LVLH – orbit-normal (= -h̃)

    Returns
    -------
    C_LVLH_ECI : ndarray(3,3)
    """
    r = np.asarray(r_ECI, dtype=float)
    v = np.asarray(v_ECI, dtype=float)
    h = np.cross(r, v)           # angular momentum vector

    x_hat = r / np.linalg.norm(r)
    z_hat = -h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)

    return np.row_stack([x_hat, y_hat, z_hat])


def mean_to_true_anomaly(M: float, e: float, tol: float = 1e-10,
                          max_iter: int = 100) -> float:
    """
    Solve Kepler's equation M = E − e sin(E) for E, then find true anomaly ν.

    Parameters
    ----------
    M   : mean anomaly [rad]
    e   : eccentricity
    tol : convergence tolerance

    Returns
    -------
    nu : true anomaly [rad]
    """
    E = M  # initial guess
    for _ in range(max_iter):
        dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                         math.sqrt(1 - e) * math.cos(E / 2))
    return nu


def orbital_period(a: float, mu: float = 3.986004418e14) -> float:
    """Orbital period [s] from semi-major axis [m]."""
    return 2 * math.pi * math.sqrt(a**3 / mu)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Utility / Math
# ═══════════════════════════════════════════════════════════════════════════════

def skew(v: ndarray) -> ndarray:
    """
    3×3 skew-symmetric (cross-product) matrix of vector v.
    skew(v) @ u  ≡  v × u
    """
    v = np.asarray(v, dtype=float)
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])


def angle_between(v1: ndarray, v2: ndarray) -> float:
    """Return angle between two vectors in radians."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.acos(np.clip(cos_a, -1.0, 1.0))


def rotation_error_angle(C1: ndarray, C2: ndarray) -> float:
    """Return the principal rotation angle between two DCMs [rad]."""
    C_err = C1 @ C2.T
    return math.acos(np.clip((np.trace(C_err) - 1) / 2, -1.0, 1.0))


def is_valid_dcm(C: ndarray, tol: float = 1e-6) -> bool:
    """Check that C is a proper orthogonal matrix (|det| ≈ 1, Cᵀ C ≈ I)."""
    C = np.asarray(C, dtype=float)
    orth = np.allclose(C.T @ C, np.eye(3), atol=tol)
    det = abs(np.linalg.det(C) - 1) < tol
    return bool(orth and det)


def normalize_quaternion(q: ndarray) -> ndarray:
    """Return unit quaternion."""
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def rodrigues_rotation(v: ndarray, k: ndarray, angle: float) -> ndarray:
    """
    Rotate vector v about unit vector k by angle [rad] (active rotation).
    Uses Rodrigues' formula: v' = v cosθ + (k×v) sinθ + k (k·v)(1−cosθ)
    """
    v = np.asarray(v, dtype=float)
    k = np.asarray(k, dtype=float)
    k = k / np.linalg.norm(k)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)


# ═══════════════════════════════════════════════════════════════════════════════
# Quick-reference demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import textwrap

    np.set_printoptions(precision=6, suppress=True)

    print("=" * 62)
    print("  ADCS.py — Quick Demo")
    print("=" * 62)

    # ── DCM (3-2-1 Euler) ────────────────────────────────────────────
    angles_321 = [np.radians(30), np.radians(20), np.radians(10)]
    C321 = DCM(angles_321, sequence=(3, 2, 1))
    print("\n[1] 3-2-1 DCM (yaw=30°, pitch=20°, roll=10°):")
    print(C321.matrix)

    # Round-trip check
    recovered = C321.to_euler((3, 2, 1))
    print(f"    Recovered angles (deg): {np.degrees(recovered).round(4)}")

    # ── DCM (3-1-3 classic Euler) ─────────────────────────────────────
    angles_313 = [np.radians(45), np.radians(30), np.radians(60)]
    C313 = DCM(angles_313, sequence=(3, 1, 3))
    print("\n[2] 3-1-3 DCM (α=45°, β=30°, γ=60°):")
    print(C313.matrix)
    print(f"    Is valid DCM: {is_valid_dcm(C313.matrix)}")

    # ── Quaternion ────────────────────────────────────────────────────
    q = Quaternion.from_euler(angles_321, sequence=(3, 2, 1))
    print(f"\n[3] Quaternion from 3-2-1 Euler:\n    {q}")

    q_id = Quaternion.identity()
    err  = q.error(q_id)
    print(f"    Error from identity: {err}")

    # ── TRIAD ─────────────────────────────────────────────────────────
    s_body = np.array([0.267, 0.535, 0.802])   # sun vector in body
    m_body = np.array([-0.577, 0.577, -0.577]) # mag field in body
    s_ref  = np.array([0.0, 0.0, 1.0])
    m_ref  = np.array([1.0, 0.0, 0.0])
    C_triad = triad(s_body, m_body, s_ref, m_ref)
    print("\n[4] TRIAD DCM (body ← reference):")
    print(C_triad)

    # ── PD Controller ─────────────────────────────────────────────────
    I_sc = np.diag([10.0, 15.0, 20.0])          # inertia [kg m²]
    Kp, Kd = compute_gain_pd(I_sc, omega_n=0.1, zeta=0.7)
    ctrl = PDQuaternionController(Kp, Kd)
    q_cur = Quaternion.from_axis_angle([0, 0, 1], np.radians(15)).q
    q_ref = Quaternion.identity().q
    tau   = ctrl.control(q_cur, q_ref, omega_current=np.zeros(3))
    print(f"\n[5] PD Controller torque [N·m]: {tau.round(4)}")

    # ── Orbital period ────────────────────────────────────────────────
    a_LEO = 6_778_000.0   # 400 km altitude [m]
    T     = orbital_period(a_LEO)
    print(f"\n[6] Orbital period at 400 km: {T/60:.2f} min")

    print("\n" + "=" * 62)
    print("  All demos completed successfully.")
    print("=" * 62)