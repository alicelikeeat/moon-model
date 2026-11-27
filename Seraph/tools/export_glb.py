"""Export Moon-Seraph.blend to a game-ready GLB for Moon Action.

glTF cannot carry procedural shading (vertex-colour makeup, sheer layer-weight fabrics,
box-projected brocade, the eye shader), so those are baked to albedo textures first.
Render-only subdivision stays off, so the GLB ships the light cage.

    python export_glb.py -- Moon-Seraph.blend Moon-Seraph.glb
"""
import os
import sys

import bpy

BAKE = {  # object -> texture size
    '01 • body | sculpted figure': 2048,
    'Bodice • ivory brocade satin': 1024,
    '02 • eyes | amber irises': 512,
}


def bake_albedo(obj, size):
    img = bpy.data.images.new(f'{obj.name} • albedo', size, size)
    nodes = []
    for slot in obj.material_slots:
        nt = slot.material.node_tree
        n = nt.nodes.new('ShaderNodeTexImage')
        n.image = img
        nt.nodes.active = n
        nodes.append((nt, n))
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, margin=8, use_clear=True)
    for nt, n in nodes:
        nt.nodes.remove(n)
    img.pack()
    first = obj.material_slots[0].material.node_tree.nodes['Principled BSDF']
    m = bpy.data.materials.new(f'{obj.name} • baked')
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    for k in ('Roughness', 'Metallic'):
        b.inputs[k].default_value = first.inputs[k].default_value
    t = m.node_tree.nodes.new('ShaderNodeTexImage')
    t.image = img
    m.node_tree.links.new(t.outputs['Color'], b.inputs['Base Color'])
    obj.data.materials.clear()
    obj.data.materials.append(m)
    for p in obj.data.polygons:
        p.material_index = 0


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    src, out = argv[0], argv[1]
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(src))
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 4
    arm = bpy.data.objects['Seraph']
    for pb in arm.pose.bones:          # bake and export from the rest pose
        pb.matrix_basis.identity()
    for name, size in BAKE.items():
        bake_albedo(bpy.data.objects[name], size)
        print('[bake]', name)
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(out), export_format='GLB', export_apply=True,
                              export_skins=True, export_animations=True, export_image_format='JPEG',
                              export_jpeg_quality=88, export_yup=True)
    print('[export]', out, os.path.getsize(out) // 1024, 'KiB')


if __name__ == '__main__':
    main()
