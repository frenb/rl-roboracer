"""
Real-to-sim, step 2: layout.json + mat.png -> a 3D model of the track, in
Blender (true scale, meters, Z up).

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b --python scripts/real2sim/build_blender_model.py -- unity/Assets/Real2Sim/home-track
    "... blender.exe" -b --python scripts/real2sim/build_blender_model.py -- unity/Assets/Real2Sim/home-track --layout layout-v2.json --stem track-v2

Writes <stem>.fbx (textures embedded, default track.fbx) into that folder, and into its blender~
subfolder track.blend, track.glb (viewers, the web) and preview.png, a render
to check the model against the video. Unity skips folders ending in ~, so it
does not try to convert the .blend. The mat is a textured plane; each fence
panel is a translucent sheet between two posts, as in layout.json's
fence_panels.
"""
import json
import math
import sys
from pathlib import Path

import bpy

PANEL_THICKNESS = 0.004
POST_RADIUS = 0.009
POST_CAP = 0.012


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv or argv[0].startswith("--"):
        sys.exit("usage: blender -b --python build_blender_model.py -- <folder with mat.png> "
                 "[--layout layout.json] [--stem track] [--no-render]")
    folder = Path(argv[0]).resolve()
    layout, stem, do_render = "layout.json", "track", True
    i = 1
    while i < len(argv):
        if argv[i] == "--layout" and i + 1 < len(argv):
            layout, i = argv[i + 1], i + 2
        elif argv[i] == "--stem" and i + 1 < len(argv):
            stem, i = argv[i + 1], i + 2
        elif argv[i] == "--no-render":
            do_render, i = False, i + 1
        else:
            sys.exit(f"unknown argument {argv[i]!r}")
    return folder, layout, stem, do_render


def material(name, color, alpha=1.0, roughness=0.6, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
        elif hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    if image:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(image))
        m.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    m.diffuse_color = (*color, alpha)
    return m


def hex_rgb(h, default=(0.3, 0.3, 0.3)):
    try:
        h = h.lstrip("#")
        srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)
    except Exception:
        return default


def box(name, size, location, rotation_z, mat, parent):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location, rotation=(0, 0, rotation_z))
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    o.parent = parent
    return o


def build(folder: Path, layout_name: str):
    layout_path = folder / layout_name
    if not layout_path.exists():
        sys.exit(f"no such layout: {layout_path}")
    L = json.loads(layout_path.read_text(encoding="utf-8"))
    w, l = float(L["mat"]["width"]), float(L["mat"]["length"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    root = bpy.data.objects.new(L.get("name", "track"), None)
    scene.collection.objects.link(root)

    floor_mat = material("Floor", (0.55, 0.45, 0.35), roughness=0.5)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(w / 2, l / 2, -0.001))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (w + 1.5, l + 1.5, 1)
    bpy.ops.object.transform_apply(scale=True)
    floor.data.materials.append(floor_mat)
    floor.parent = root

    # The plane's default UVs span 0..1, so mat.png covers the mat exactly.
    mat_png = folder / "mat.png"
    mat_mat = material("Mat", hex_rgb(L["mat"].get("base_color", "#4a4a4f")), roughness=0.8,
                       image=mat_png if mat_png.exists() else None)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(w / 2, l / 2, 0))
    mat = bpy.context.active_object
    mat.name = "Mat"
    mat.scale = (w, l, 1)
    bpy.ops.object.transform_apply(scale=True)
    mat.data.materials.append(mat_mat)
    mat.parent = root

    panel_mat = material("FencePanel", (0.62, 0.62, 0.6), alpha=0.55, roughness=0.3)
    post_mat = material("FencePost", (0.03, 0.03, 0.03), roughness=0.4)
    fence = bpy.data.objects.new("Fence", None)
    scene.collection.objects.link(fence)
    fence.parent = root
    posts = {}
    for i, p in enumerate(L.get("fence_panels", []), 1):
        x1, y1, x2, y2 = (float(p[k]) for k in ("x1", "y1", "x2", "y2"))
        h = float(p.get("height") or 0.35)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-3:
            continue
        yaw = math.atan2(y2 - y1, x2 - x1)
        box(f"Panel-{i}", (length - 2 * POST_RADIUS, PANEL_THICKNESS, h), ((x1 + x2) / 2, (y1 + y2) / 2, h / 2 + 0.01),
            yaw, panel_mat, fence)
        for x, y in ((x1, y1), (x2, y2)):
            key = (round(x, 2), round(y, 2))
            posts[key] = max(posts.get(key, 0), h)
    for i, ((x, y), h) in enumerate(posts.items(), 1):
        bpy.ops.mesh.primitive_cylinder_add(radius=POST_RADIUS, depth=h + 0.02, location=(x, y, (h + 0.02) / 2), vertices=12)
        o = bpy.context.active_object
        o.name = f"Post-{i}"
        o.data.materials.append(post_mat)
        o.parent = fence
        bpy.ops.mesh.primitive_uv_sphere_add(radius=POST_CAP, location=(x, y, h + 0.02), segments=12, ring_count=6)
        c = bpy.context.active_object
        c.name = f"PostCap-{i}"
        c.data.materials.append(post_mat)
        c.parent = fence
    print(f"built: mat {w} x {l} m, {len(L.get('fence_panels', []))} panels, {len(posts)} posts")
    return L, w, l


def save(folder: Path, stem: str):
    bpy.ops.object.select_all(action="SELECT")
    side = folder / "blender~"
    side.mkdir(exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(side / f"{stem}.blend"))
    bpy.ops.export_scene.gltf(filepath=str(side / f"{stem}.glb"), export_format="GLB")
    bpy.ops.export_scene.fbx(filepath=str(folder / f"{stem}.fbx"), path_mode="COPY", embed_textures=True,
                             apply_scale_options="FBX_SCALE_ALL", bake_space_transform=True)
    print(f"saved: {side / f'{stem}.blend'}, {side / f'{stem}.glb'}, {folder / f'{stem}.fbx'}")


def render(folder: Path, w: float, l: float, stem: str):
    scene = bpy.context.scene
    engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if e in engines:
            try:
                scene.render.engine = e
                break
            except TypeError:
                continue
    if scene.render.engine == "BLENDER_WORKBENCH":
        scene.display.shading.color_type = "TEXTURE"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.8, 0.8, 0.82, 1)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = world
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(30))
    scene.collection.objects.link(sun)

    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    d = max(w, l) * 1.25
    cam.location = (w / 2 - 0.55 * d, l / 2 - 0.85 * d, 0.95 * d)
    target = bpy.data.objects.new("Target", None)
    target.location = (w / 2, l / 2, 0)
    scene.collection.objects.link(target)
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    cam.data.lens = 32
    scene.render.resolution_x, scene.render.resolution_y = 1400, 1000
    preview = "preview.png" if stem == "track" else f"{stem}-preview.png"
    scene.render.filepath = str(folder / "blender~" / preview)
    bpy.ops.render.render(write_still=True)
    print(f"rendered: {scene.render.filepath} ({scene.render.engine})")


if __name__ == "__main__":
    folder, layout_name, stem, do_render = args()
    _, w, l = build(folder, layout_name)
    save(folder, stem)
    if do_render:
        render(folder, w, l, stem)
