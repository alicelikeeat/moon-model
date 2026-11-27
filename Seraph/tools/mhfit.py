"""Fit the CC0 MakeHuman hm08 body onto the Moon Seraph skeleton.

Pure numpy: loads the base mesh, applies shape targets, then poses it with linear blend
skinning (MakeHuman's own default weights) so its joints land on Seraph's rest skeleton
(T-pose arms, high-heel feet with flat toes). The head and hands stay rigid (no stretch).
"""
import json
import numpy as np


def read_obj(path):
    vs, uvs, faces = [], [], []
    g = None
    for line in open(path):
        if line.startswith('v '):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('vt '):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith('g '):
            g = line.split()[1]
        elif line.startswith('f '):
            f = [[int(y) if y else 0 for y in x.split('/')] for x in line.split()[1:]]
            faces.append((g, f))
    return np.array(vs), np.array(uvs), faces


def read_target(path):
    idx, d = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        idx.append(int(p[0]))
        d.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(idx, dtype=int), np.array(d)


def to_blender(v, scale):
    """MakeHuman (y up, +z forward, decimetres) -> Blender (z up, -y forward, metres)."""
    v = np.asarray(v)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1) * scale


def minrot(a, b):
    """Smallest rotation matrix taking direction a onto direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def axis_stretch(u, r):
    u = u / np.linalg.norm(u)
    return np.eye(3) + (r - 1) * np.outer(u, u)


class Polyline:
    """Maps points near a source polyline onto a target polyline, keeping perpendicular offsets."""

    def __init__(self, src, dst):
        self.src = [np.asarray(p, float) for p in src]
        self.dst = [np.asarray(p, float) for p in dst]
        self.rot = [minrot(self.src[i + 1] - self.src[i], self.dst[i + 1] - self.dst[i]) for i in range(len(src) - 1)]

    def __call__(self, p):
        best = None
        for i in range(len(self.src) - 1):
            a, b = self.src[i], self.src[i + 1]
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
            foot = a + t * ab
            d = np.linalg.norm(p - foot)
            if best is None or d < best[0] - 1e-9:
                best = (d, i, t, foot)
        _, i, t, foot = best
        a2, b2 = self.dst[i], self.dst[i + 1]
        return a2 + t * (b2 - a2) + self.rot[i] @ (p - foot)


class MHBody:
    def __init__(self, mh_dir, targets, scale):
        vs, self.uvs, self.faces = read_obj(f'{mh_dir}/base.obj')
        for name, w in targets:
            idx, d = read_target(f'{mh_dir}/{name}')
            vs[idx] += w * d
        self.raw = vs                      # MakeHuman space, shaped
        self.scale = scale
        self.v = to_blender(vs, 0.1 * scale)
        self.skel = json.load(open(f'{mh_dir}/default.mhskel'))
        self.weights = json.load(open(f'{mh_dir}/default_weights.mhw'))['weights']
        self.bones = self.skel['bones']
        self.order = self._order()

    def joint(self, name):
        return self.v[self.skel['joints'][name]].mean(axis=0)

    def group_center(self, group):
        idx = {i[0] - 1 for g, f in self.faces if g == group for i in f}
        return self.v[sorted(idx)].mean(axis=0)

    def head(self, b):
        return self.joint(self.bones[b]['head'])

    def tail(self, b):
        return self.joint(self.bones[b]['tail'])

    def _order(self):
        out, seen = [], set()

        def visit(b):
            if b in seen:
                return
            p = self.bones[b]['parent']
            if p:
                visit(p)
            seen.add(b)
            out.append(b)
        for b in self.bones:
            visit(b)
        return out

    def body_indices(self, groups=('body',)):
        return sorted({i[0] - 1 for g, f in self.faces if g in groups for i in f})


class Fit:
    """Per-bone affine transforms D_b(p) = H'_b + Q_b S_b (p - h_b)."""

    def __init__(self, mh):
        self.mh = mh
        self.Q, self.S, self.H, self.h = {}, {}, {}, {}

    def chain(self, b, H2, T2):
        mh = self.mh
        p = mh.bones[b]['parent']
        Qp = self.Q[p] if p else np.eye(3)
        h, t = mh.head(b), mh.tail(b)
        u = t - h
        Q = minrot(Qp @ u, T2 - H2) @ Qp
        r = np.linalg.norm(T2 - H2) / max(np.linalg.norm(u), 1e-9)
        self.set(b, h, H2, Q, axis_stretch(u, r))

    def rigid(self, b, H2, Q, S=None):
        self.set(b, self.mh.head(b), H2, Q, np.eye(3) if S is None else S)

    def follow(self, b, Q=None):
        mh = self.mh
        p = mh.bones[b]['parent']
        h = mh.head(b)
        H2 = self.apply_bone(p, h)
        Sp = self.S[p]
        # Inherit only a uniform scale (the head's); never a parent's axis stretch.
        k = np.cbrt(abs(np.linalg.det(Sp)))
        S = Sp if np.allclose(Sp, k * np.eye(3), atol=1e-6) else np.eye(3)
        self.set(b, h, H2, self.Q[p] if Q is None else Q, S)

    def set(self, b, h, H2, Q, S):
        self.h[b], self.H[b], self.Q[b], self.S[b] = h, np.asarray(H2, float), Q, S

    def apply_bone(self, b, p):
        return self.H[b] + self.Q[b] @ self.S[b] @ (p - self.h[b])

    def matrix(self, b):
        A = self.Q[b] @ self.S[b]
        return A, self.H[b] - A @ self.h[b]

    def skin(self, idx):
        """Linear blend skinning of MakeHuman vertex indices `idx`."""
        mh = self.mh
        pos = mh.v
        out = np.zeros((len(idx), 3))
        wsum = np.zeros(len(idx))
        where = {v: i for i, v in enumerate(idx)}
        for b, lst in mh.weights.items():
            if b not in self.Q:
                continue
            A, c = self.matrix(b)
            for vi, w in lst:
                i = where.get(vi)
                if i is None:
                    continue
                out[i] += w * (A @ pos[vi] + c)
                wsum[i] += w
        miss = wsum < 1e-6
        out[~miss] /= wsum[~miss, None]
        out[miss] = pos[np.array(idx)[miss]]
        return out, miss
