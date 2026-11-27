"""Rebuild the Moon Seraph body with real modeled anatomy.

Run with Blender's Python (bpy 5.x):
    tools/fetch_makehuman_assets.sh /tmp/mh
    python rebuild_seraph.py -- --src Moon-Seraph.blend --mh /tmp/mh --out Moon-Seraph.blend

Stages: fit body -> rig -> garments/face -> materials. Every stage is deterministic.
"""
import argparse
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mhfit  # noqa: E402

SIDES = (('L', 'l', 1.0), ('R', 'r', -1.0))

# Face sculpt toward the reference: oval head, short fine chin, slim nose, full lips,
# larger lifted eyes, high cheekbones, slender neck.
FACE = [('head-oval', 0.6), ('head-scale-vert-decr', 0.35), ('chin-height-decr', 0.45), ('chin-width-decr', 0.4),
        ('nose-scale-horiz-decr', 0.35), ('nose-scale-vert-decr', 0.25), ('nose-point-width-decr', 0.35),
        ('mouth-upperlip-volume-incr', 0.4), ('mouth-lowerlip-volume-incr', 0.3), ('mouth-scale-horiz-decr', 0.25),
        ('mouth-cupidsbow-incr', 0.4), ('l-eye-scale-incr', 0.35), ('r-eye-scale-incr', 0.35),
        ('l-eye-corner1-up', 0.3), ('r-eye-corner1-up', 0.3), ('l-cheek-bones-incr', 0.35), ('r-cheek-bones-incr', 0.35),
        ('forehead-scale-vert-decr', 0.3), ('neck-scale-horiz-decr', 0.25)]


def args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--mh', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--stage', default='all')
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- old skeleton

def old_skeleton():
    arm = bpy.data.objects['Seraph']
    return {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in arm.data.bones}


def mesh_points(name):
    o = bpy.data.objects[name]
    return np.array([o.matrix_world @ v.co for v in o.data.vertices])


# ----------------------------------------------------------------------------- body fit

def fit_body(mh_dir):
    sk = old_skeleton()
    mh = mhfit.MHBody(mh_dir, [('proportions.target', 1.0), ('slim.target', 0.45), ('bust.target', 0.45)]
                      + [(f'face/{n}.target', w) for n, w in FACE], scale=0.95)
    F = mhfit.Fit(mh)

    # Head: rigid, uniform scale from temple width; eyes aligned to the old eye line.
    old_face = mesh_points('02 • neck and face')
    old_eyes = mesh_points('Eyes • curved sclera iris and lid rims')
    eye_old = np.array([0.0, old_eyes[:, 1].max() + 0.004, old_eyes[:, 2].mean()])
    eye_mh = (mh.group_center('joint-l-eye') + mh.group_center('joint-r-eye')) / 2
    body_idx = mh.body_indices()
    bv = mh.v[body_idx]

    def half_width(pts, z, band=0.006):
        sel = pts[np.abs(pts[:, 2] - z) < band]
        return np.abs(sel[:, 0]).max()
    zt = 0.03
    k_head = half_width(old_face, eye_old[2] + zt) / half_width(bv, eye_mh[2] + zt)
    k_head = float(np.clip(k_head, 0.88, 1.08))
    head_pivot = mh.head('head')
    head_new = eye_old + k_head * (head_pivot - eye_mh)

    # Spine / neck polyline: pelvis -> neck base -> skull pivot.
    pelvis = mh.head('spine05')
    spine = mhfit.Polyline([pelvis, mh.head('neck01'), head_pivot], [sk['Hips'][0], sk['Neck'][0], head_new])
    F.set('root', mh.head('root'), spine(mh.head('root')), np.eye(3), np.eye(3))
    for b in ['spine05', 'spine04', 'spine03', 'spine02', 'spine01', 'neck01', 'neck02', 'neck03']:
        F.chain(b, spine(mh.head(b)), spine(mh.tail(b)))
    F.rigid('head', head_new, np.eye(3), k_head * np.eye(3))

    for S, s, sign in SIDES:
        # Arms: clavicle -> shoulder -> elbow -> wrist onto the old T-pose bones.
        arm = mhfit.Polyline([mh.head(f'clavicle.{S}'), mh.head(f'upperarm01.{S}'), mh.head(f'lowerarm01.{S}'), mh.head(f'wrist.{S}')],
                             [sk[f'Clavicle.{S}'][0], sk[f'Arm.{S}'][0], sk[f'Forearm.{S}'][0], sk[f'Hand.{S}'][0]])
        for b in ['clavicle', 'shoulder01', 'upperarm01', 'upperarm02', 'lowerarm01', 'lowerarm02']:
            F.chain(f'{b}.{S}', arm(mh.head(f'{b}.{S}')), arm(mh.tail(f'{b}.{S}')))
        knuckle = mh.head(f'finger3-1.{S}')
        Qp = F.Q[f'lowerarm02.{S}']
        hand_dir = sk[f'Hand.{S}'][1] - sk[f'Hand.{S}'][0]
        Qh = mhfit.minrot(Qp @ (knuckle - mh.head(f'wrist.{S}')), hand_dir) @ Qp
        F.rigid(f'wrist.{S}', sk[f'Hand.{S}'][0], Qh)

        # Legs: hip -> knee -> ankle; foot stretched so the ball meets the old ball; toes flat.
        # The hip joint stays where the pelvis puts it (the old rig had it above the pelvis
        # root, which stretched the pelvis); the thigh then reaches the old knee.
        hip = mh.head(f'upperleg01.{S}')
        leg = mhfit.Polyline([hip, mh.head(f'lowerleg01.{S}'), mh.head(f'foot.{S}')],
                             [spine(hip), sk[f'Shin.{S}'][0], sk[f'Foot.{S}'][0]])
        F.chain(f'pelvis.{S}', spine(mh.head(f'pelvis.{S}')), leg(mh.tail(f'pelvis.{S}')))
        for b in ['upperleg01', 'upperleg02', 'lowerleg01', 'lowerleg02']:
            F.chain(f'{b}.{S}', leg(mh.head(f'{b}.{S}')), leg(mh.tail(f'{b}.{S}')))
        ankle = mh.head(f'foot.{S}')
        ball = np.mean([mh.head(f'toe{i}-1.{S}') for i in range(1, 6)], axis=0)
        Qp = F.Q[f'lowerleg02.{S}']
        old_a, old_b = sk[f'Foot.{S}']
        Qf = mhfit.minrot(Qp @ (ball - ankle), old_b - old_a) @ Qp
        r = np.linalg.norm(old_b - old_a) / np.linalg.norm(ball - ankle)
        F.rigid(f'foot.{S}', old_a, Qf, mhfit.axis_stretch(ball - ankle, r))

    # Everything else follows its parent rigidly; toe roots are turned flat to the floor.
    for b in mh.order:
        if b in F.Q:
            continue
        if b.startswith('toe') and b.split('.')[0].endswith('-1'):
            p = mh.bones[b]['parent']
            d = F.Q[p] @ (mh.tail(b) - mh.head(b))
            flat = np.array([d[0], d[1], 0.0])
            F.follow(b, mhfit.minrot(d, flat) @ F.Q[p])
        else:
            F.follow(b)

    posed, miss = F.skin(body_idx)
    print(f'[fit] head scale {k_head:.3f}; unweighted verts {int(miss.sum())}')
    return mh, F, body_idx, posed, dict(k_head=k_head, eye_old=eye_old, eye_mh=eye_mh, head_new=head_new, head_pivot=head_pivot)


def build_mesh(name, verts, mh, body_idx, collection=None):
    remap = {o: n for n, o in enumerate(body_idx)}
    faces, uvf = [], []
    for g, f in mh.faces:
        if g != 'body':
            continue
        faces.append([remap[i[0] - 1] for i in f])
        uvf.append([mh.uvs[i[1] - 1] for i in f])
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    uv = me.uv_layers.new(name='UVMap')
    k = 0
    for f in uvf:
        for c in f:
            uv.data[k].uv = c
            k += 1
    for p in me.polygons:
        p.use_smooth = True
    me.update()
    o = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(o)
    return o


# ----------------------------------------------------------------------------- eyes (mhclo proxy)

def fit_proxy(mh, mhclo_path, obj_path):
    """MakeHuman proxy fitting: v = sum(w_i * base[v_i]) + offset * per-axis scale (in MH space)."""
    raw = mh.raw
    lines = open(mhclo_path).read().splitlines()
    scale = {}
    refs = []
    in_verts = False
    for ln in lines:
        p = ln.split()
        if not p or p[0].startswith('#'):
            continue
        if p[0] in ('x_scale', 'y_scale', 'z_scale'):
            ax = 'xyz'.index(p[0][0])
            a_, b_, ref = int(p[1]), int(p[2]), float(p[3])
            scale[ax] = abs(raw[a_][ax] - raw[b_][ax]) / ref
        elif p[0] == 'verts':
            in_verts = True
        elif in_verts and len(p) == 9:
            refs.append(p)
        elif in_verts and len(p) == 1 and p[0].isdigit():
            refs.append(p)
        elif in_verts:
            in_verts = False
    pos = []
    for p in refs:
        if len(p) == 1:
            pos.append(raw[int(p[0])])
            continue
        v = [int(x) for x in p[:3]]
        w = [float(x) for x in p[3:6]]
        d = np.array([float(x) for x in p[6:9]])
        q = sum(w[i] * raw[v[i]] for i in range(3)) + d * np.array([scale.get(i, 1.0) for i in range(3)])
        pos.append(q)
    _, _, faces = mhfit.read_obj(obj_path)
    return mhfit.to_blender(np.array(pos), 0.1 * mh.scale), faces


def build_eyes(mh, F, mh_dir):
    pos, faces = fit_proxy(mh, f'{mh_dir}/eyes/high-poly.mhclo', f'{mh_dir}/eyes/high-poly.obj')
    A, c = F.matrix('head')
    pos = pos @ A.T + c
    me = bpy.data.meshes.new('Eyes')
    me.from_pydata([tuple(v) for v in pos], [], [[i[0] - 1 for i in f] for g, f in faces])
    for p in me.polygons:
        p.use_smooth = True
    # Planar UV per eye, looking down -Y: the iris disc fills UV radius 0.5.
    uv = me.uv_layers.new(name='UVMap')
    centers = {}
    for s in (1, -1):
        sel = pos[pos[:, 0] * s > 0]
        front = sel[sel[:, 1] < np.percentile(sel[:, 1], 8)]
        centers[s] = (front[:, 0].mean(), front[:, 2].mean())
    r_eye = (pos[:, 0].max() - pos[:, 0].min()) / 4.0   # eyeball radius
    iris_r = r_eye * 0.48
    for poly in me.polygons:
        for li in poly.loop_indices:
            v = pos[me.loops[li].vertex_index]
            cx, cz = centers[1 if v[0] > 0 else -1]
            uv.data[li].uv = (0.5 + (v[0] - cx) / (2 * iris_r), 0.5 + (v[2] - cz) / (2 * iris_r))
    me.update()
    o = bpy.data.objects.new('Eyes', me)
    bpy.context.scene.collection.objects.link(o)
    return o


# ----------------------------------------------------------------------------- head accessories

def refit_head_accessories(body, info):
    """Re-seat crown, halo and hair cap on the new (taller, rounder) skull.

    Below the eye line nothing moves. Above it, points are stretched so the old scalp
    profile lands on the new one; the crown and halo move rigidly with the scalp."""
    bv = np.array([v.co for v in body.data.vertices])
    old_face = mesh_points('02 • neck and face')
    e = info['eye_old'][2]
    head = bv[(bv[:, 2] > e - 0.02) & (np.abs(bv[:, 1]) < 0.15) & (np.abs(bv[:, 0]) < 0.11)]

    def profile(pts, z):
        sel = pts[np.abs(pts[:, 2] - z) < 0.006]
        return np.abs(sel[:, 0]).max(), sel[:, 1].min(), sel[:, 1].max()
    top_old, top_new = old_face[:, 2].max(), head[:, 2].max()
    az = (top_new - e) / (top_old - e)
    zf = e + 0.035
    wo, fo, bo = profile(old_face, zf)
    wn, fn, bn = profile(head, e + 0.035 * az)
    ax = wn / wo
    ay = (bn - fn) / (bo - fo)
    yc_old, yc_new = (fo + bo) / 2, (fn + bn) / 2
    print(f'[head] top {top_old:.3f}->{top_new:.3f}  az {az:.3f} ax {ax:.3f} ay {ay:.3f}')

    def G(p):
        q = p.copy()
        q[..., 0] = p[..., 0] * ax
        q[..., 1] = yc_new + (p[..., 1] - yc_old) * ay
        q[..., 2] = e + (p[..., 2] - e) * az
        return q

    def blend(z):
        t = np.clip((z - (e - 0.07)) / 0.07, 0, 1)
        return t * t * (3 - 2 * t)

    for name, mode in [('07 • layered waves', 'warp'), ('09 • filigree crown', 'rigid_xy'), ('08 • solar diadem', 'rigid')]:
        o = bpy.data.objects[name]
        mw = np.array(o.matrix_world)
        co = np.array([v.co for v in o.data.vertices])
        p = co @ mw[:3, :3].T + mw[:3, 3]
        if mode == 'warp':
            w = blend(p[:, 2])[:, None]
            # Slight outward push so hair clears the scalp everywhere it was stretched.
            q = p + w * (G(p) - p)
            n = q - np.array([0, yc_new, e])
            n[:, 2] *= 0.4
            q += w * 0.0025 * n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
        else:
            cen = p.mean(axis=0)
            base = cen.copy()
            base[2] = np.percentile(p[:, 2], 35) if mode == 'rigid_xy' else cen[2]   # crown: its band, not pendants
            d = G(base[None])[0] - base
            if mode == 'rigid_xy':
                q = p.copy()
                q[:, 0] = cen[0] + (p[:, 0] - cen[0]) * ax
                q[:, 1] = cen[1] + (p[:, 1] - cen[1]) * ay
                q += d
            else:
                q = p + np.array([0, d[1], d[2]])
        inv = np.linalg.inv(mw)
        loc = q @ inv[:3, :3].T + inv[:3, 3]
        for v, c in zip(o.data.vertices, loc):
            v.co = Vector(c)
        o.data.update()


# ----------------------------------------------------------------------------- rig

FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']


def seraph_bone(mh_bone):
    """MakeHuman default-rig bone -> Seraph deform bone."""
    n, side = (mh_bone.split('.') + [''])[:2]
    S = f'.{side}' if side else ''
    if n in ('root', 'spine05', 'spine04') or n == 'pelvis':
        return 'Hips'
    if n in ('spine03', 'spine02', 'spine01', 'breast'):
        return 'Spine'
    if n.startswith('neck'):
        return 'Neck'
    if n in ('clavicle', 'shoulder01'):
        return 'Clavicle' + S
    if n.startswith('upperarm'):
        return 'Arm' + S
    if n.startswith('lowerarm'):
        return 'Forearm' + S
    if n in ('wrist',) or n.startswith('metacarpal'):
        return 'Hand' + S
    if n.startswith('finger'):
        f, j = n[6:].split('-')
        return f'{FINGERS[int(f) - 1]}{j}{S}'
    if n.startswith('upperleg'):
        return 'Thigh' + S
    if n.startswith('lowerleg'):
        return 'Shin' + S
    if n == 'foot':
        return 'Foot' + S
    if n.startswith('toe'):
        return 'Toe' + S
    return 'Head'


def extend_armature(mh, F):
    arm = bpy.data.objects['Seraph']
    bpy.context.view_layer.objects.active = arm
    for o in bpy.context.view_layer.objects:
        o.select_set(o == arm)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    P = lambda b, p: Vector(F.apply_bone(b, p))
    # Anatomical lumbar pivot: Hips/Spine joint at the posed spine03 joint.
    pivot = P('spine03', mh.head('spine03'))
    eb['Hips'].tail = pivot
    eb['Spine'].head = pivot
    for S, s, sign in SIDES:
        eb[f'Thigh.{S}'].head = P(f'upperleg01.{S}', mh.head(f'upperleg01.{S}'))
        hand = eb[f'Hand.{S}']
        for fi, fname in enumerate(FINGERS, start=1):
            parent = hand
            for j in (1, 2, 3):
                mb = f'finger{fi}-{j}.{S}'
                b = eb.new(f'{fname}{j}.{S}')
                b.head = P(mb, mh.head(mb))
                b.tail = P(mb, mh.tail(mb))
                b.roll = 0
                b.parent = parent
                b.use_connect = j > 1
                parent = b
        toes = [f'toe{i}-1.{S}' for i in range(1, 6)]
        tips = [f'toe{i}-{2 if i == 1 else 3}.{S}' for i in range(1, 6)]
        t = eb.new(f'Toe.{S}')
        t.head = Vector(np.mean([F.apply_bone(b, mh.head(b)) for b in toes], axis=0))
        t.tail = Vector(np.mean([F.apply_bone(b, mh.tail(b)) for b in tips], axis=0))
        t.parent = eb[f'Foot.{S}']
        eb[f'Foot.{S}'].tail = t.head
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm


def skin_body(body, mh, body_idx, arm):
    n = len(body_idx)
    where = {v: i for i, v in enumerate(body_idx)}
    acc = {}
    for b, lst in mh.weights.items():
        tgt = seraph_bone(b)
        w = acc.setdefault(tgt, np.zeros(n))
        for vi, wt in lst:
            i = where.get(vi)
            if i is not None:
                w[i] += wt
    total = sum(acc.values())
    total[total < 1e-9] = 1
    for tgt, w in acc.items():
        g = body.vertex_groups.new(name=tgt)
        w = w / total
        for i in np.nonzero(w > 1e-4)[0]:
            g.add([int(i)], float(w[i]), 'REPLACE')
    attach(body, arm)


def subdiv(obj):
    m = obj.modifiers.new('Subdivision', 'SUBSURF')
    m.levels = 0            # viewport and glTF export: the base cage (game-ready)
    m.render_levels = 1     # renders: smooth
    return m


def attach(obj, arm):
    obj.parent = arm
    obj.matrix_parent_inverse = arm.matrix_world.inverted()
    m = obj.modifiers.new('Skin', 'ARMATURE')   # added before any subdivision, so it runs first
    m.object = arm


# ----------------------------------------------------------------------------- garments

def body_data(body):
    P = np.array([v.co for v in body.data.vertices])
    N = np.array([v.normal for v in body.data.vertices])
    names = {g.index: g.name for g in body.vertex_groups}
    W = {}
    for v in body.data.vertices:
        for g in v.groups:
            W.setdefault(names[g.group], np.zeros(len(P)))[v.index] = g.weight
    z = np.zeros(len(P))
    w = lambda *ns: sum((W.get(n, z) for n in ns), z)
    return P, N, w


def face_mask(body, vmask, mode='all'):
    f = np.array([[vmask[i] for i in p.vertices] for p in body.data.polygons], dtype=object)
    return np.array([all(x) if mode == 'all' else (sum(x) * 2 >= len(x)) for x in f])


def shell(body, name, fmask, offset, thickness, material, smooth=0, solid_offset=1.0):
    """A garment shell: the body faces in `fmask`, pushed out along the normals, given thickness."""
    o = body.copy()
    o.data = body.data.copy()
    o.name = o.data.name = name
    bpy.context.scene.collection.objects.link(o)
    for m in list(o.modifiers):
        if m.type != 'ARMATURE':
            o.modifiers.remove(m)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not fmask[f.index]], context='FACES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    # Relax the border so the garment edge is a clean line (it starts as a vertex-selection zigzag).
    for _ in range(12):
        new = {}
        for v in bm.verts:
            if not v.is_boundary:
                continue
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new[v] = v.co * 0.4 + (nb[0].co + nb[1].co) * 0.3
        for v, c in new.items():
            v.co = c
    if smooth:
        for _ in range(smooth):
            bmesh.ops.smooth_vert(bm, verts=[v for v in bm.verts if not v.is_boundary], factor=0.5,
                                  use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * offset
    # Smoothing pulls the shell inward; keep every vertex at least `offset` outside the
    # body's *base cage* (what the game renders), not just its subdivided surface.
    from mathutils.bvhtree import BVHTree
    cage = BVHTree.FromPolygons([v.co for v in body.data.vertices], [list(p.vertices) for p in body.data.polygons])
    for v in bm.verts:
        loc, nrm, _, _ = cage.find_nearest(v.co)
        if loc is None:
            continue
        d = (v.co - loc).dot(nrm)
        if d < offset:
            v.co = loc + nrm * offset
    bm.to_mesh(o.data)
    bm.free()
    o.data.materials.clear()
    o.data.materials.append(material)
    if thickness:
        s = o.modifiers.new('Thickness', 'SOLIDIFY')
        s.thickness = thickness
        s.offset = solid_offset
        s.use_even_offset = True
    subdiv(o)
    return o


def edge_loops_to_tubes(src, name, radius, material, body, min_len=8, profile=6):
    """Gold piping along every open border of `src` (a garment shell), as skinned tube meshes."""
    me = src.data
    bm = bmesh.new()
    bm.from_mesh(me)
    border = [e for e in bm.edges if e.is_boundary]
    adj = {}
    for e in border:
        a, b = e.verts
        adj.setdefault(a.index, []).append(b.index)
        adj.setdefault(b.index, []).append(a.index)
    co = {v.index: np.array(v.co) for v in bm.verts}
    nrm = {v.index: np.array(v.normal) for v in bm.verts}
    bm.free()
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= min_len:
            loops.append(loop)
    verts, faces = [], []
    for loop in loops:
        closed = loop[0] in adj[loop[-1]]
        pts = [co[i] + nrm[i] * radius * 0.6 for i in loop]
        # Light smoothing so the piping reads as a clean line, not a polygon border.
        for _ in range(3):
            pts = [pts[i] if (not closed and i in (0, len(pts) - 1)) else
                   (pts[i - 1] + 2 * pts[i] + pts[(i + 1) % len(pts)]) / 4 for i in range(len(pts))]
        n = len(pts)
        base = len(verts)
        for i in range(n):
            t = pts[(i + 1) % n] - pts[i - 1] if closed else pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            t /= max(np.linalg.norm(t), 1e-9)
            up = nrm[loop[i]] - np.dot(nrm[loop[i]], t) * t
            up /= max(np.linalg.norm(up), 1e-9)
            side = np.cross(t, up)
            for k in range(profile):
                a = 2 * math.pi * k / profile
                verts.append(pts[i] + radius * (math.cos(a) * up + math.sin(a) * side))
        segs = n if closed else n - 1
        for i in range(segs):
            j = (i + 1) % n
            for k in range(profile):
                k2 = (k + 1) % profile
                faces.append([base + i * profile + k, base + j * profile + k, base + j * profile + k2, base + i * profile + k2])
    m = bpy.data.meshes.new(name)
    m.from_pydata([tuple(v) for v in verts], [], faces)
    for p in m.polygons:
        p.use_smooth = True
    m.materials.append(material)
    m.update()
    o = bpy.data.objects.new(name, m)
    bpy.context.scene.collection.objects.link(o)
    copy_weights(o, body)
    return o


def copy_weights(obj, body, side=None):
    """Give a new part the skin weights of the nearest body vertex (same armature).
    `side` (+1 left / -1 right) limits the search to that half, so parts of one foot or
    leg can never pick up weights from the other one."""
    from mathutils.kdtree import KDTree
    bv = body.data.vertices
    names = [g.name for g in body.vertex_groups]

    def lean(v):   # >0: driven by left-side bones, <0: right-side bones, 0: centre
        return sum(g.weight * (1 if names[g.group].endswith('.L') else -1 if names[g.group].endswith('.R') else 0)
                   for g in v.groups)
    pool = [v for v in bv if side is None or lean(v) * side >= 0]
    kd = KDTree(len(pool))
    for v in pool:
        kd.insert(v.co, v.index)
    kd.balance()
    groups = {}
    for v in obj.data.vertices:
        _, idx, _ = kd.find(v.co)
        for g in bv[idx].groups:
            name = body.vertex_groups[g.group].name
            if side is not None and name.endswith('.R' if side > 0 else '.L'):
                continue
            if name not in groups:
                groups[name] = obj.vertex_groups.new(name=name)
            groups[name].add([v.index], g.weight, 'REPLACE')
    attach(obj, bpy.data.objects['Seraph'])


def stiletto(name, top, ground_z, material, body):
    """Tapered stiletto heel from the shoe's heel cup straight down to the floor."""
    rings = [(top + np.array([0, 0.004, 0]), 0.011, 0.014), (top + np.array([0, 0.006, -0.035]), 0.006, 0.008),
             (np.array([top[0], top[1] + 0.012, ground_z + 0.008]), 0.0042, 0.0046),
             (np.array([top[0], top[1] + 0.012, ground_z]), 0.0042, 0.0046)]
    sides = 16
    verts, faces = [], []
    for c, rx, ry in rings:
        for k in range(sides):
            a = 2 * math.pi * k / sides
            verts.append(c + np.array([rx * math.cos(a), ry * math.sin(a), 0]))
    for r in range(len(rings) - 1):
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append([r * sides + k, (r + 1) * sides + k, (r + 1) * sides + k2, r * sides + k2])
    faces.append(list(range(len(rings) * sides - 1, (len(rings) - 1) * sides - 1, -1)))
    m = bpy.data.meshes.new(name)
    m.from_pydata([tuple(v) for v in verts], [], faces)
    for p in m.polygons:
        p.use_smooth = True
    m.materials.append(material)
    o = bpy.data.objects.new(name, m)
    bpy.context.scene.collection.objects.link(o)
    copy_weights(o, body)
    for g in list(o.vertex_groups):      # the heel is rigid with the foot
        if not g.name.startswith('Foot'):
            o.vertex_groups.remove(g)
    side = 'L' if top[0] > 0 else 'R'
    g = o.vertex_groups.get(f'Foot.{side}') or o.vertex_groups.new(name=f'Foot.{side}')
    g.add(list(range(len(verts))), 1.0, 'REPLACE')
    return o


def region_bvh(body, vmask):
    """BVH of only the body faces in a vertex region (so a limb ignores its neighbour)."""
    from mathutils.bvhtree import BVHTree
    verts = [v.co for v in body.data.vertices]
    polys = [list(p.vertices) for p in body.data.polygons if sum(vmask[i] for i in p.vertices) * 2 >= len(p.vertices)]
    return BVHTree.FromPolygons(verts, polys)


def grid_object(name, grid, material, body, closed_u=True, thickness=0.0, uv_scale=(1, 1), side=None):
    """Quad mesh from a rows x cols grid of points (cols wrap if closed_u), with grid UVs."""
    R, C = grid.shape[:2]
    verts = grid.reshape(-1, 3)
    faces = []
    cc = C if closed_u else C - 1
    for r in range(R - 1):
        for c in range(cc):
            c2 = (c + 1) % C
            faces.append([r * C + c, r * C + c2, (r + 1) * C + c2, (r + 1) * C + c])
    m = bpy.data.meshes.new(name)
    m.from_pydata([tuple(v) for v in verts], [], faces)
    uv = m.uv_layers.new(name='UVMap')
    for p in m.polygons:
        p.use_smooth = True
        for li in p.loop_indices:
            vi = m.loops[li].vertex_index
            r, c = divmod(vi, C)
            if closed_u and c == 0 and any(m.loops[k].vertex_index % C == C - 1 for k in p.loop_indices):
                c = C
            uv.data[li].uv = (c / cc * uv_scale[0], r / (R - 1) * uv_scale[1])
    m.materials.append(material)
    m.update()
    o = bpy.data.objects.new(name, m)
    bpy.context.scene.collection.objects.link(o)
    if thickness:
        s = o.modifiers.new('Thickness', 'SOLIDIFY')
        s.thickness = thickness
        s.offset = 1.0
    copy_weights(o, body, side)
    subdiv(o)
    return o


def ring_band(name, body, bvh, a, b, s0, s1, material, rows=6, cols=48, offset=0.002, thickness=0.0012,
              flare=0.0, scallop=0.0, uv_scale=(8, 1), side=None):
    """A clean band hugging the body around the axis a->b between fractions s0..s1 (raycast inward)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = (b - a) / np.linalg.norm(b - a)
    ref = np.array([0, -1.0, 0]) if abs(ax[1]) < 0.9 else np.array([1.0, 0, 0])
    u = ref - np.dot(ref, ax) * ax
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    grid = np.zeros((rows, cols, 3))
    for r in range(rows):
        t = r / (rows - 1)
        s = s0 + (s1 - s0) * t
        c = a + (b - a) * s
        for k in range(cols):
            ang = 2 * math.pi * k / cols
            d = math.cos(ang) * u + math.sin(ang) * v
            hit = bvh.ray_cast(Vector(c + d * 0.25), Vector(-d))
            p = np.array(hit[0]) if hit[0] is not None else c + d * 0.05
            extra = flare * t * t + scallop * t * (0.5 + 0.5 * math.cos(ang * 12))
            grid[r, k] = p + d * (offset + extra)
    return grid_object(name, grid, material, body, True, thickness, uv_scale, side)


def pump(name, body, foot_pts, ball, side, material):
    """A pointed-toe pump fitted to the posed foot.

    Sections are fitted around the foot along a heel -> ball -> toe path (the foot is
    bent at the ball in heels), the vamp is cut open over the instep, the toe box is
    closed and drawn out to a point, and the heel counter is closed at the back."""
    rear = foot_pts[np.argmax(foot_pts[:, 1])]
    tip = foot_pts[np.argmin(foot_pts[:, 1])]
    heel_c = rear + np.array([0, -0.012, 0])
    path = [heel_c, ball, tip, tip + np.array([side * 0.004, -0.026, -0.004])]
    seg = np.r_[0, np.cumsum([np.linalg.norm(path[i + 1] - path[i]) for i in range(3)])]
    total = seg[-1]
    L_ball, L_tip = seg[1], seg[2]

    def at(d):
        i = min(np.searchsorted(seg, d, side='right') - 1, 2)
        t = (d - seg[i]) / (seg[i + 1] - seg[i])
        return path[i] + t * (path[i + 1] - path[i])

    def tangent(d):
        e = 0.012
        v = at(min(total, d + e)) - at(max(0, d - e))
        return v / np.linalg.norm(v)
    n = 34
    ds = np.linspace(0, total, n)
    frames, secs = [], []
    for d in ds:
        c, t = at(d), tangent(d)
        lat = np.cross(t, [0, 0, 1.0])
        lat /= np.linalg.norm(lat)
        up = np.cross(lat, t)
        if up[2] < 0:
            up = -up
        rel = foot_pts - c
        sel = np.abs(rel @ t) < 0.006
        if d <= L_tip and sel.sum() >= 4:
            x, y = rel[sel] @ lat, rel[sel] @ up
            secs.append([(x.max() + x.min()) / 2, (y.max() + y.min()) / 2,
                         (x.max() - x.min()) / 2 + 0.0035, (y.max() - y.min()) / 2 + 0.0035])
        else:
            secs.append(None)
        frames.append((c, t, lat, up))
    known = [i for i, q in enumerate(secs) if q is not None]
    for i in range(n):
        if secs[i] is None and ds[i] <= L_tip:
            j = min(known, key=lambda k: abs(k - i))
            secs[i] = list(secs[j])
    last = max(i for i in range(n) if ds[i] <= L_tip)
    for i in range(last + 1, n):                       # pointed toe
        t = (ds[i] - ds[last]) / (total - ds[last])
        cx, cy, rx, ry = secs[last]
        secs[i] = [cx * (1 - t), cy * (1 - t) - ry * 0.45 * t, max(0.0022, rx * (1 - t) ** 0.85), max(0.002, ry * (1 - t) ** 1.2)]
    secs = np.array(secs)
    for _ in range(4):
        secs[1:-1] = (secs[:-2] + 2 * secs[1:-1] + secs[2:]) / 4
    cols = 44
    grid = np.zeros((n, cols, 3))
    for i, d in enumerate(ds):
        c, t, lat, up = frames[i]
        cx, cy, rx, ry = secs[i]
        f = d / L_ball
        # Opening over the instep: closed heel counter, open throat, closed toe box from the ball.
        if f < 0.25:
            oh = math.radians(1)
        elif f < 0.45:
            oh = math.radians(1 + 74 * (f - 0.25) / 0.2)
        elif f < 0.85:
            oh = math.radians(75)
        elif f < 1.05:
            oh = math.radians(75 * (1 - (f - 0.85) / 0.2) + 1)
        else:
            oh = math.radians(1)
        for k in range(cols):
            th = oh + (2 * math.pi - 2 * oh) * k / (cols - 1)        # 0 = top centre
            grid[i, k] = c + lat * (cx + rx * math.sin(th)) + up * (cy + ry * math.cos(th))
    # Close the back of the heel counter.
    back = grid[0].mean(axis=0) + frames[0][1] * -0.004
    grid = np.concatenate([np.repeat(back[None, None], cols, axis=1), grid], axis=0)
    return grid_object(name, grid, material, body, closed_u=False, thickness=0.0016, uv_scale=(1, 3), side=side)


def build_garments(body, mats):
    P, N, w = body_data(body)
    arms = w('Arm.L', 'Arm.R')
    lower_arm = w('Forearm.L', 'Forearm.R', 'Hand.L', 'Hand.R', *[f'{f}{j}.{S}' for f in FINGERS for j in (1, 2, 3) for S in 'LR'])
    legs = w('Thigh.L', 'Thigh.R', 'Shin.L', 'Shin.R', 'Foot.L', 'Foot.R', 'Toe.L', 'Toe.R')
    torso = w('Hips', 'Spine')
    bones = {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in bpy.data.objects['Seraph'].data.bones}
    ax_x = abs(bones['Arm.L'][0][0])
    el_x = abs(bones['Forearm.L'][0][0])
    glove_x = ax_x + 0.42 * (el_x - ax_x)
    lace_top, lace_bot = 0.805, 0.755

    # Material zones painted straight onto the body surface (keeps fingers/toes crisp).
    glove = (lower_arm > 0.5) | ((arms > 0.5) & (np.abs(P[:, 0]) > glove_x))
    stocking = (legs > 0.5) & (P[:, 2] < lace_top - 0.01)
    zone = np.zeros(len(P), dtype=int)       # 0 skin, 1 glove, 2 stocking
    zone[glove] = 1
    zone[stocking] = 2
    body.data.materials.clear()
    for m in (mats['skin'], mats['glove'], mats['stocking']):
        body.data.materials.append(m)
    for p in body.data.polygons:
        z = [zone[i] for i in p.vertices]
        p.material_index = max(set(z), key=z.count)

    parts = []
    # Bodice: torso from the high-cut leg line up to a sweetheart neckline above the bust.
    # Smoothed so the fabric bridges the cleavage and reads as cups, not paint.
    apex = P[(torso > 0.5) & (P[:, 2] > 1.15)]
    apex_z = apex[np.argmin(apex[:, 1]), 2]
    ang = np.arctan2(P[:, 0], -P[:, 1])                       # 0 = front centre
    top = apex_z + 0.07 - 0.012 * np.cos(2 * ang) + 0.025 * np.clip(np.cos(ang), 0, 1) ** 8
    bod = (torso > 0.55) & (P[:, 2] < top) & (legs < 0.45) & (arms < 0.2) & (w('Clavicle.L', 'Clavicle.R') < 0.3)
    bodice = shell(body, 'Bodice • ivory brocade satin', face_mask(body, bod), 0.0026, 0.0012, mats['silk'], smooth=4)
    parts += [bodice, edge_loops_to_tubes(bodice, 'Bodice • gold piping', 0.0016, mats['gold'], body)]

    def frac(a, b, z):
        return (z - a[2]) / (b[2] - a[2])

    # High lace collar around the neck.
    neck_r = (w('Neck', 'Spine', 'Clavicle.L', 'Clavicle.R') > 0.4) & (P[:, 2] < 1.6)
    na, nb = bones['Neck'][0], bones['Head'][0] + np.array([0, 0, 0.02])
    parts.append(ring_band('Collar • champagne lace', body, region_bvh(body, neck_r), na, nb,
                           frac(na, nb, 1.522), frac(na, nb, 1.585), mats['lace'], rows=6, cols=56, offset=0.0016,
                           flare=0.003, uv_scale=(10, 1)))
    for S in 'LR':
        # Stocking lace bands with a scalloped top.
        leg_r = w(f'Thigh.{S}') > 0.3
        ta, tb = bones[f'Thigh.{S}']
        parts.append(ring_band(f'Stockings • lace band {S}', body, region_bvh(body, leg_r), ta, tb,
                               frac(ta, tb, lace_bot), frac(ta, tb, lace_top), mats['lace_white'], rows=7, cols=56,
                               offset=0.0012, scallop=0.0015, uv_scale=(10, 1), side=1 if S == 'L' else -1))
        # Glove cuffs.
        arm_r = w(f'Arm.{S}', f'Forearm.{S}') > 0.3
        aa, ab = bones[f'Arm.{S}']
        L = np.linalg.norm(ab - aa)
        f0 = (glove_x - abs(aa[0])) / abs(ab[0] - aa[0])
        parts.append(ring_band(f'Gloves • lace cuff {S}', body, region_bvh(body, arm_r), aa, ab, f0 - 0.012 / L, f0 + 0.022 / L,
                               mats['glove_lace'], rows=5, cols=40, offset=0.0010, scallop=0.0012, uv_scale=(6, 1),
                               side=1 if S == 'L' else -1))

        # Pointed pumps + stilettos.
        foot = (w(f'Foot.{S}', f'Toe.{S}') > 0.45) & (P[:, 2] < bones[f'Foot.{S}'][0][2] - 0.012)
        fp = P[foot]
        rear = fp[fp[:, 1] > np.percentile(fp[:, 1], 80)]
        heel = rear[np.argmin(rear[:, 2])] + np.array([0, 0.004, 0])
        sign = 1 if S == 'L' else -1
        parts.append(pump(f'Shoes • champagne pump {S}', body, fp, bones[f'Toe.{S}'][0], sign, mats['shoe']))
        parts.append(stiletto(f'Shoes • stiletto {S}', heel + np.array([0, -0.006, -0.002]), 0.0, mats['gold'], body))
    return parts


# ----------------------------------------------------------------------------- face details

def body_bvh(body):
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(body, dg)


def eye_openings(body, head_z):
    """Open border loops of the body around the eyes (the lid rims)."""
    bm = bmesh.new()
    bm.from_mesh(body.data)
    adj = {}
    for e in bm.edges:
        if e.is_boundary:
            a, b = e.verts
            adj.setdefault(a.index, []).append(b.index)
            adj.setdefault(b.index, []).append(a.index)
    co = {v.index: np.array(v.co) for v in bm.verts}
    bm.free()
    seen, loops = set(), []
    for s in adj:
        if s in seen:
            continue
        loop, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        pts = np.array([co[i] for i in loop])
        c = pts.mean(axis=0)
        if abs(c[2] - head_z) < 0.05 and 0.01 < abs(c[0]) < 0.06 and c[1] < -0.03:
            loops.append(pts)
    return loops


def strand_mesh(name, strands, material, width, body, sides=3):
    """Thin tapered tubes along polylines (lashes, brow hairs)."""
    verts, faces = [], []
    for pts, w0 in strands:
        n = len(pts)
        base = len(verts)
        for i, p in enumerate(pts):
            t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            t /= np.linalg.norm(t)
            a = np.cross(t, [0, 0, 1.0]) if abs(t[2]) < 0.9 else np.cross(t, [1.0, 0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(t, a)
            r = width * w0 * (1 - 0.85 * i / (n - 1))
            for k in range(sides):
                ang = 2 * math.pi * k / sides
                verts.append(p + r * (math.cos(ang) * a + math.sin(ang) * b))
        for i in range(n - 1):
            for k in range(sides):
                k2 = (k + 1) % sides
                faces.append([base + i * sides + k, base + (i + 1) * sides + k, base + (i + 1) * sides + k2, base + i * sides + k2])
    m = bpy.data.meshes.new(name)
    m.from_pydata([tuple(v) for v in verts], [], faces)
    for p in m.polygons:
        p.use_smooth = True
    m.materials.append(material)
    o = bpy.data.objects.new(name, m)
    bpy.context.scene.collection.objects.link(o)
    g = o.vertex_groups.new(name='Head')
    g.add(list(range(len(verts))), 1.0, 'REPLACE')
    attach(o, bpy.data.objects['Seraph'])
    return o


def build_lashes_and_brows(body, eye_z, mats):
    rng = np.random.default_rng(7)
    bvh = body_bvh(body)
    lashes = []
    for loop in eye_openings(body, eye_z):
        c = loop.mean(axis=0)
        side = 1 if c[0] > 0 else -1
        upper = loop[loop[:, 2] > c[2] - 0.0005]
        upper = upper[np.argsort(upper[:, 0] * side)]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(upper, axis=0), axis=1))]
        for s in np.linspace(0.04, 0.98, 34) * seg[-1]:
            p = np.array([np.interp(s, seg, upper[:, k]) for k in range(3)])
            t = s / seg[-1]                                # 0 inner corner -> 1 outer corner
            out = np.array([side * (0.15 + 0.7 * t), -0.85, 0.35])
            out /= np.linalg.norm(out)
            L = 0.0055 + 0.0045 * math.sin(math.pi * min(1, t * 1.15)) + 0.002 * t
            pts = [p + out * L * u + np.array([0, 0, 1.0]) * L * 0.9 * u * u + rng.normal(0, 0.00025, 3)
                   for u in np.linspace(0, 1, 5)]
            lashes.append((np.array(pts), 1.0))
    parts = [strand_mesh('Face • upper lashes', lashes, mats['lash'], 0.00036, body)]

    # Brows: short hairs lying on the skin along an elegant arch, tips flowing outward.
    brows = []
    for side in (1, -1):
        for _ in range(70):
            t = rng.uniform(0, 1)
            x = side * (0.012 + 0.046 * t)
            z = eye_z + 0.0185 + 0.007 * math.sin(math.pi * min(1, t * 1.25)) - 0.004 * t + rng.normal(0, 0.0012) * (1 - 0.6 * t)
            hit = bvh.ray_cast(Vector((x, -0.3, z)), Vector((0, 1, 0)))
            if hit[0] is None:
                continue
            p, n = np.array(hit[0]), np.array(hit[1])
            d = np.array([side * (0.55 + 0.4 * t), 0, 0.55 - 0.9 * t])
            d -= np.dot(d, n) * n
            d /= np.linalg.norm(d)
            L = rng.uniform(0.004, 0.0065)
            pts = [p + n * 0.0004 + d * L * u + n * 0.0006 * u * (1 - u) for u in np.linspace(0, 1, 4)]
            brows.append((np.array(pts), rng.uniform(0.8, 1.1)))
    parts.append(strand_mesh('Face • brows', brows, mats['brow'], 0.00022, body))

    # Gold face filigree from the reference: a forehead drop and a tear line under each eye,
    # raised in relief on the skin.
    def on_face(x, z, lift=0.0005):
        hit = bvh.ray_cast(Vector((x, -0.3, z)), Vector((0, 1, 0)))
        return None if hit[0] is None else np.array(hit[0]) + np.array(hit[1]) * lift

    def line(pts2d, w0=1.0):
        pts = [on_face(x, z) for x, z in pts2d]
        if any(p is None for p in pts):
            return None
        return (np.array(pts), w0)
    marks = []
    fz = eye_z + 0.045
    for ln in ([(0, fz + 0.012), (0, fz), (0, fz - 0.016)],
               [(-0.007, fz + 0.004), (0, fz - 0.002), (0.007, fz + 0.004)],
               [(-0.004, fz - 0.008), (0, fz - 0.011), (0.004, fz - 0.008)]):
        marks.append(line(ln))
    for side in (1, -1):
        x0 = side * 0.030
        marks.append(line([(x0, eye_z - 0.012), (x0 + side * 0.002, eye_z - 0.022), (x0 + side * 0.001, eye_z - 0.032),
                           (x0 + side * 0.004, eye_z - 0.040)]))
        marks.append(line([(x0 + side * 0.0015, eye_z - 0.021), (x0 + side * 0.008, eye_z - 0.025)], 0.8))
        marks.append(line([(x0 + side * 0.001, eye_z - 0.030), (x0 - side * 0.005, eye_z - 0.034)], 0.8))
    marks = [m for m in marks if m is not None]
    # densify each mark so the relief follows the face curvature
    dense = []
    for pts, w0 in marks:
        out = [pts[0]]
        for a_, b_ in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, 6)[1:]:
                out.append(a_ + (b_ - a_) * t)
        dense.append((np.array(out), w0))
    parts.append(strand_mesh('Face • gold filigree', dense, mats['gold'], 0.0007, body, sides=5))
    return parts


def paint_makeup(body, eye_z, mouth):
    """Vertex-colour multiplier: lip tint, soft blush, warm lid shadow."""
    P = np.array([v.co for v in body.data.vertices])
    col = np.ones((len(P), 4))
    dx, dz = P[:, 0], P[:, 2]
    lips_box = (np.abs(dx) < 0.028) & (np.abs(dz - mouth[2]) < 0.014) & (P[:, 1] < -0.04)
    front = P[lips_box, 1].min()
    lips = lips_box & (P[:, 1] < front + 0.010)
    falloff = np.clip(1 - (np.abs(dx) / 0.026) ** 4, 0, 1) * np.clip(1 - (np.abs(dz - mouth[2]) / 0.012) ** 2, 0, 1)
    lip = np.where(lips, falloff, 0)[:, None]
    col[:, :3] = col[:, :3] * (1 - lip) + lip * np.array([0.80, 0.50, 0.48])
    face = (P[:, 1] < -0.03)[:, None]
    for side in (1, -1):
        cheek = np.exp(-(((dx - side * 0.042) / 0.018) ** 2 + ((dz - (eye_z - 0.028)) / 0.014) ** 2))[:, None] * face
        col[:, :3] *= 1 - cheek * (1 - np.array([0.97, 0.88, 0.87]))
        lid = np.exp(-(((dx - side * 0.034) / 0.016) ** 2 + ((dz - (eye_z + 0.007)) / 0.006) ** 2))[:, None] * face
        col[:, :3] *= 1 - lid * (1 - np.array([0.84, 0.68, 0.52]))
    attr = body.data.color_attributes.new('Makeup', 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set('color', col.ravel())


# ----------------------------------------------------------------------------- materials

def principled(name, color, rough=0.5, metal=0.0, sss=0.0, coat=0.0, sheen=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*color, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    b.inputs['Subsurface Weight'].default_value = sss
    b.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.2)
    b.inputs['Subsurface Scale'].default_value = 0.008
    b.inputs['Coat Weight'].default_value = coat
    b.inputs['Sheen Weight'].default_value = sheen
    return m, nt, b


SKIN = (0.86, 0.70, 0.62)


def mix_rgb(nt, blend='MIX'):
    m = nt.nodes.new('ShaderNodeMix')
    m.data_type = 'RGBA'
    m.blend_type = blend
    return m


def make_materials():
    mats = {}
    m, nt, b = principled('Skin • porcelain', SKIN, rough=0.42, sss=0.18)
    tint = nt.nodes.new('ShaderNodeVertexColor')
    tint.layer_name = 'Makeup'
    mul = mix_rgb(nt, 'MULTIPLY')
    mul.inputs['Factor'].default_value = 1.0
    mul.inputs[6].default_value = (*SKIN, 1)
    nt.links.new(tint.outputs['Color'], mul.inputs[7])
    nt.links.new(mul.outputs[2], b.inputs['Base Color'])
    mats['skin'] = m

    def sheer(name, dark, amount, edge):
        m, nt, b = principled(name, dark, rough=0.45)
        lw = nt.nodes.new('ShaderNodeLayerWeight')
        lw.inputs['Blend'].default_value = edge
        mr = nt.nodes.new('ShaderNodeMapRange')
        mr.inputs['To Min'].default_value = amount
        mr.inputs['To Max'].default_value = 0.97
        nt.links.new(lw.outputs['Facing'], mr.inputs['Value'])
        mix = mix_rgb(nt)
        mix.inputs[6].default_value = (*SKIN, 1)
        mix.inputs[7].default_value = (*dark, 1)
        nt.links.new(mr.outputs['Result'], mix.inputs['Factor'])
        nt.links.new(mix.outputs[2], b.inputs['Base Color'])
        return m
    mats['stocking'] = sheer('Stockings • smoke sheer', (0.075, 0.065, 0.07), 0.74, 0.35)
    mats['glove'] = sheer('Gloves • espresso lace', (0.055, 0.035, 0.028), 0.86, 0.3)
    mats['glove_lace'] = principled('Gloves • espresso lace cuff', (0.06, 0.045, 0.04), rough=0.6, sheen=0.5)[0]
    # Bodice: ivory satin with the gold-vine brocade box-projected at fabric scale.
    m, nt, b = principled('Bodice • ivory brocade satin', (0.94, 0.90, 0.80), rough=0.32, sheen=0.25)
    b.inputs['Specular IOR Level'].default_value = 0.6
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.inputs['Scale'].default_value = (7, 7, 3.5)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images['brocade.jpg']
    tex.projection = 'BOX'
    tex.projection_blend = 0.3
    nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
    tint = mix_rgb(nt, 'MULTIPLY')
    tint.inputs['Factor'].default_value = 1.0
    tint.inputs[7].default_value = (1.0, 0.97, 0.9, 1)
    nt.links.new(tex.outputs['Color'], tint.inputs[6])
    nt.links.new(tint.outputs[2], b.inputs['Base Color'])
    mats['silk'] = m
    mats['lace'] = bpy.data.materials['Champagne embroidery']
    mats['gold'] = bpy.data.materials['Antique gold']
    m, nt, b = principled('Stockings • ivory lace band', (0.93, 0.90, 0.84), rough=0.6, sheen=0.6)
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images['brocade.jpg']
    nt.links.new(tex.outputs['Color'], b.inputs['Base Color'])
    mats['lace_white'] = m
    mats['shoe'] = principled('Shoes • champagne patent', (0.86, 0.74, 0.52), rough=0.22, metal=0.85, coat=0.6)[0]
    mats['lash'] = principled('Face • sepia lashes', (0.16, 0.11, 0.07), rough=0.5)[0]
    mats['brow'] = principled('Face • ash-blonde brows', (0.30, 0.22, 0.14), rough=0.55)[0]

    # Eye: amber iris (UV radius < 0.5) inside a warm sclera, under a wet clear coat.
    m, nt, b = principled('Eye • honey amber iris', (0.95, 0.92, 0.88), rough=0.08, coat=1.0, sss=0.05)
    b.inputs['Coat Roughness'].default_value = 0.02
    uv = nt.nodes.new('ShaderNodeUVMap')
    dist = nt.nodes.new('ShaderNodeVectorMath')
    dist.operation = 'DISTANCE'
    dist.inputs[1].default_value = (0.5, 0.5, 0)
    nt.links.new(uv.outputs['UV'], dist.inputs[0])
    edge = nt.nodes.new('ShaderNodeMapRange')
    edge.inputs['From Min'].default_value = 0.47
    edge.inputs['From Max'].default_value = 0.52
    nt.links.new(dist.outputs['Value'], edge.inputs['Value'])
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images['iris.jpg']
    nt.links.new(uv.outputs['UV'], tex.inputs['Vector'])
    mix = mix_rgb(nt)
    nt.links.new(edge.outputs['Result'], mix.inputs['Factor'])
    gold = mix_rgb(nt, 'MULTIPLY')      # warm the brown iris toward honey gold
    gold.inputs['Factor'].default_value = 1.0
    gold.inputs[7].default_value = (1.6, 1.15, 0.45, 1)
    nt.links.new(tex.outputs['Color'], gold.inputs[6])
    nt.links.new(gold.outputs[2], mix.inputs[6])
    mix.inputs[7].default_value = (0.95, 0.92, 0.88, 1)
    nt.links.new(mix.outputs[2], b.inputs['Base Color'])
    mats['eye'] = m
    return mats


def main():
    a = args()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.src))
    mh, F, body_idx, posed, info = fit_body(a.mh)
    body = build_mesh('Body', posed, mh, body_idx)
    if a.stage == 'fit':
        subdiv(body)
        for n in ['01 • figure | sculpt and bodice', '02 • neck and face', '03 • limbs L', '03 • limbs R',
                  'Eyes • curved sclera iris and lid rims', 'Mouth • shaped vermilion']:
            bpy.data.objects[n].hide_render = True
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
        return

    eyes = build_eyes(mh, F, a.mh)
    refit_head_accessories(body, info)
    arm = extend_armature(mh, F)
    skin_body(body, mh, body_idx, arm)
    vg = eyes.vertex_groups.new(name='Head')
    vg.add(list(range(len(eyes.data.vertices))), 1.0, 'REPLACE')
    attach(eyes, arm)
    old_eye_mats = bpy.data.objects['Eyes • curved sclera iris and lid rims'].data.materials
    for n in ['01 • figure | sculpt and bodice', '02 • neck and face', '03 • limbs L', '03 • limbs R',
              'Eyes • curved sclera iris and lid rims', 'Mouth • shaped vermilion']:
        bpy.data.objects.remove(bpy.data.objects[n])
    subdiv(body)
    if a.stage == 'rig':
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
        return

    mats = make_materials()
    eyes.data.materials.clear()
    eyes.data.materials.append(mats['eye'])
    mouth = F.apply_bone('head', mh.group_center('joint-mouth'))
    eye_z = float(info['eye_old'][2])
    paint_makeup(body, eye_z, mouth)
    build_garments(body, mats)
    build_lashes_and_brows(body, eye_z, mats)
    body.name = body.data.name = '01 • body | sculpted figure'
    eyes.name = eyes.data.name = '02 • eyes | amber irises'
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out), compress=True)


if __name__ == '__main__':
    main()
