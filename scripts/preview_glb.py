"""Render a .glb the way a glTF viewer sees it, before uploading it to Sketchfab.

Imports only the file's contents into an empty Blender, so anything the exporter
dropped or mis-mapped shows up in the render.

    blender -b --factory-startup --python scripts/preview_glb.py -- model.glb out.png [frame ...]

Writes one PNG per frame (out_f0001.png, ...); defaults to frame 1.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
glb, out = args[0], args[1]
frames = [int(f) for f in args[2:]] or [1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath(glb))
scene = bpy.context.scene

meshes = [o for o in scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector(min(p[i] for p in pts) for i in range(3))
hi = Vector(max(p[i] for p in pts) for i in range(3))
centre, radius = (lo + hi) / 2, (hi - lo).length / 2

cam_data = bpy.data.cameras.new("preview")
cam_data.lens = 50
cam = bpy.data.objects.new("preview", cam_data)
scene.collection.objects.link(cam)
direction = Vector((1.0, -0.9, 0.7)).normalized()
cam.location = centre + direction * radius / math.sin(cam_data.angle / 2) * 1.05
cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.clip_start, cam_data.clip_end = radius / 100, radius * 100
scene.camera = cam

world = bpy.data.worlds.new("preview")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.6, 0.6, 0.6, 1)
scene.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3
sun.rotation_euler = (math.radians(40), 0, math.radians(30))
scene.collection.objects.link(sun)

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 1280, 960
scene.render.image_settings.file_format = "PNG"
stem, _ = os.path.splitext(os.path.abspath(out))
for f in frames:
    scene.frame_set(f)
    scene.render.filepath = f"{stem}_f{f:04d}.png"
    bpy.ops.render.render(write_still=True)
    print("wrote", scene.render.filepath)
