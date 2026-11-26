"""Object catalog: every object the simulator can spawn.

MuJoCo compiles the model once, so every catalog object is added to the scene at build time and
"spawning" means moving an object from the parking area (far off to the side, on the floor) onto the
table. "Despawning" moves it back. See main.spawn_object / despawn_object / reset_objects.

Shapes are built from several primitive geoms, so they are non-convex (L, T, cross, mug with a
handle, stepped pyramid, dumbbell). To add a new object, add one make_object(...) entry to
OBJECT_SPECS below. Rules of thumb:
  * part "pos" is measured from the object's BOTTOM (z) and from its footprint centre (x, y), so the
    overhead camera's blob centroid lands close to the body origin.
  * keep one hue per object, at least ~30 degrees away from every other object, and away from the
    bin's green (about 120-155 degrees), because vision.py finds objects by hue.
  * keep overall size at roughly 3-6 cm so the gripper can still close on it.
"""
import colorsys

import mujoco

PARK_X = 1.0          # parking row: on the floor, far from the table and the arms
PARK_Y0 = -0.9
PARK_DY = 0.25

# Used when random placement is switched off (RANDOMIZE_OBJECTS=0).
DEFAULT_LAYOUT = {"red_block": (-0.14, 0.03), "blue_ball": (0.14, -0.10)}

_GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}


def hue_rgba(hue, sat=0.85, val=0.85):
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
    return (r, g, b, 1.0)


def part(kind, size, pos, euler=(0.0, 0.0, 0.0)):
    """One primitive geom. size follows MuJoCo: box (hx,hy,hz), sphere (r), cylinder/capsule (r,half_len),
    ellipsoid (a,b,c). euler is in degrees."""
    return {"type": kind, "size": tuple(size), "pos": tuple(pos), "euler": tuple(euler)}


def make_object(parts, height, radius, rgba, desc, density=600.0,
                friction=(0.8, 0.005, 0.0001), damping=0.3, view_frac=1.0):
    """height: full height (m). radius: bounding radius in xy (used for spacing and pushing).
    view_frac: where the camera sees the object, as a fraction of its height (1.0 = top face,
    0.5 = centre, for round things)."""
    return {"parts": parts, "height": height, "radius": radius, "rgba": rgba, "desc": desc,
            "density": density, "friction": friction, "damping": damping, "view_frac": view_frac}


OBJECT_SPECS = {
    "red_block": make_object(
        [part("box", (0.025, 0.025, 0.025), (0, 0, 0.025))],
        height=0.05, radius=0.025, rgba=(0.85, 0.10, 0.10, 1), density=640,
        desc="a small red cube"),

    "blue_ball": make_object(
        [part("sphere", (0.03,), (0, 0, 0.03))],
        height=0.06, radius=0.03, rgba=(0.15, 0.30, 0.90, 1), density=530,
        friction=(0.8, 0.02, 0.02), damping=0.6, view_frac=0.5,
        desc="a blue ball"),

    # Stepped pyramid: three stacked boxes.
    "orange_pyramid": make_object(
        [part("box", (0.035, 0.035, 0.0075), (0, 0, 0.0075)),
         part("box", (0.025, 0.025, 0.0075), (0, 0, 0.0225)),
         part("box", (0.015, 0.015, 0.0075), (0, 0, 0.0375))],
        height=0.045, radius=0.05, rgba=hue_rgba(30),
        desc="an orange stepped pyramid made of three stacked square slabs"),

    # T-shaped block.
    "yellow_tee": make_object(
        [part("box", (0.035, 0.0125, 0.0175), (0, 0.0137, 0.0175)),
         part("box", (0.0125, 0.0225, 0.0175), (0, -0.0213, 0.0175))],
        height=0.035, radius=0.045, rgba=hue_rgba(58),
        desc="a yellow T-shaped block"),

    # L-shaped bracket.
    "lime_bracket": make_object(
        [part("box", (0.03, 0.0125, 0.02), (0.006, -0.011, 0.02)),
         part("box", (0.0125, 0.0175, 0.02), (-0.0115, 0.019, 0.02))],
        height=0.04, radius=0.045, rgba=hue_rgba(90),
        desc="a light-green L-shaped bracket"),

    # Mug: cylinder body with a C-shaped handle (hole in the middle -> non-convex).
    "cyan_mug": make_object(
        [part("cylinder", (0.028, 0.025), (-0.006, 0, 0.025)),
         part("box", (0.012, 0.004, 0.004), (0.034, 0, 0.040)),
         part("box", (0.012, 0.004, 0.004), (0.034, 0, 0.012)),
         part("box", (0.004, 0.004, 0.016), (0.042, 0, 0.026))],
        height=0.05, radius=0.047, rgba=hue_rgba(185),
        desc="a cyan mug with a handle on one side"),

    # Dumbbell: two spheres joined by a capsule bar lying along x.
    "purple_dumbbell": make_object(
        [part("sphere", (0.022,), (-0.032, 0, 0.022)),
         part("sphere", (0.022,), (0.032, 0, 0.022)),
         part("capsule", (0.008, 0.032), (0, 0, 0.022), euler=(0, 90, 0))],
        height=0.044, radius=0.054, rgba=hue_rgba(275), view_frac=0.5,
        friction=(0.9, 0.01, 0.005),
        desc="a purple dumbbell: two balls joined by a bar"),

    # Plus sign: two crossed boxes.
    "magenta_cross": make_object(
        [part("box", (0.035, 0.0125, 0.0175), (0, 0, 0.0175)),
         part("box", (0.0125, 0.035, 0.0175), (0, 0, 0.0175))],
        height=0.035, radius=0.04, rgba=hue_rgba(325),
        desc="a magenta plus-shaped cross"),
}


def add_to_spec(spec, name):
    """Add one catalog object to an MjSpec as a free body, starting in the parking area."""
    cfg = OBJECT_SPECS[name]
    height = cfg["height"]
    body = spec.worldbody.add_body(name=name, pos=list(parking_pos(name)))
    body.add_joint(name=name + "_joint", type=mujoco.mjtJoint.mjJNT_FREE, damping=cfg["damping"])
    for i, p in enumerate(cfg["parts"]):
        size = list(p["size"]) + [0.0] * (3 - len(p["size"]))
        x, y, z = p["pos"]
        body.add_geom(name=f"{name}_geom{i}", type=_GEOM_TYPES[p["type"]], size=size,
                      pos=[x, y, z - height / 2.0],      # body origin is the object's mid-height
                      euler=list(p["euler"]), rgba=list(cfg["rgba"]),
                      density=cfg["density"], friction=list(cfg["friction"]))


def parking_pos(name):
    """Where an object waits when it is not on the table (resting on the floor)."""
    index = list(OBJECT_SPECS).index(name)
    return [PARK_X, PARK_Y0 + PARK_DY * index, OBJECT_SPECS[name]["height"] / 2.0 + 0.002]
