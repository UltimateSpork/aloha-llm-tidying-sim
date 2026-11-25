import queue
import sys
import threading
import time

import mujoco
import mujoco.viewer
import mujoco_menagerie as mm
import numpy as np

import planner_actuators as pa
import llm as op

MAX_AUTO_RETRIES = 2
PROMPT = "> "


# Environment Setup

def add_object(spec, name):
    """Add one free-moving object from pa.OBJECT_SPECS, resting on the table."""
    cfg = pa.OBJECT_SPECS[name]
    body = spec.worldbody.add_body(name=name, pos=[cfg["xy"][0], cfg["xy"][1], pa.object_rest_z(name)])
    body.add_joint(name=name + "_joint", type=mujoco.mjtJoint.mjJNT_FREE, damping=cfg["damping"])
    if cfg["shape"] == "box":
        shape = mujoco.mjtGeom.mjGEOM_BOX
        size = list(cfg["half"])
    else:
        shape = mujoco.mjtGeom.mjGEOM_SPHERE
        size = [cfg["half"][0], 0, 0]
    body.add_geom(name=name + "_geom", type=shape, size=size, rgba=list(cfg["rgba"]),
                  mass=cfg["mass"], friction=list(cfg["friction"]))


def build_scene():
    """Build ALOHA + table + bin + objects. Returns (model, data, home_ctrl)."""
    spec = mm.get("aloha").spec("aloha") # the robot we are using

    # Save the robot's neutral pose, then remove the keyframe 
    home_ctrl = np.array(spec.keys[0].ctrl, dtype=float)
    for key in list(spec.keys):
        spec.delete(key)

    world = spec.worldbody
    world.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05],
                   rgba=[0.25, 0.30, 0.38, 1])

    # Table
    cx = pa.TABLE_CENTER_XY[0]
    cy = pa.TABLE_CENTER_XY[1]
    hx = pa.TABLE_HALF_XY[0]
    hy = pa.TABLE_HALF_XY[1]
    half_thick = pa.TABLE_THICKNESS / 2
    world.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, half_thick],
                   pos=[cx, cy, pa.TABLE_TOP_Z - half_thick], rgba=[0.72, 0.58, 0.42, 1])
    leg_half_height = (pa.TABLE_TOP_Z - pa.TABLE_THICKNESS) / 2
    leg_number = 0
    for sx in [1, -1]:
        for sy in [1, -1]:
            world.add_geom(name="table_leg" + str(leg_number), type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[0.02, 0.02, leg_half_height],
                           pos=[cx + sx * (hx - 0.03), cy + sy * (hy - 0.03), leg_half_height],
                           rgba=[0.45, 0.35, 0.25, 1])
            leg_number = leg_number + 1

    # Bin
    bx = pa.BIN_CENTER_XY[0]
    by = pa.BIN_CENTER_XY[1]
    world.add_geom(name="bin_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[pa.BIN_RADIUS, 0.001, 0],
                   pos=[bx, by, pa.TABLE_TOP_Z + 0.001], rgba=[0.15, 0.55, 0.25, 0.6],
                   contype=0, conaffinity=0)
    for i in range(24):
        angle = 2 * np.pi * i / 24
        world.add_geom(name="bin_rim" + str(i), type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.004, 0.006],
                       pos=[bx + pa.BIN_RADIUS * np.cos(angle), by + pa.BIN_RADIUS * np.sin(angle),
                            pa.TABLE_TOP_Z + 0.006],
                       euler=[0, 0, np.degrees(angle) + 90], rgba=[0.1, 0.4, 0.2, 1],
                       contype=0, conaffinity=0)

    for name in pa.ACTIVE_OBJECTS:
        add_object(spec, name)

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, home_ctrl


def reset_arms(model, data, amap):
    """Put both arms and grippers in the home pose."""
    for side in pa.SIDES:
        arm = amap[side]
        data.qpos[arm["qpos_adr"]] = amap["home"][arm["act_ids"]]
        for finger in ["left_finger", "right_finger"]:
            data.qpos[model.joint(side + "/" + finger).qposadr] = amap["home"][arm["grip_act"]]
    data.ctrl[:] = amap["home"]
    data.qvel[:] = 0


def reset_objects(model, data):
    """Move every object back to its start position."""
    for name in pa.ACTIVE_OBJECTS:
        joint = model.joint(name + "_joint")
        xy = pa.OBJECT_SPECS[name]["xy"]
        start = joint.qposadr[0]
        data.qpos[start:start + 7] = [xy[0], xy[1], pa.object_rest_z(name), 1, 0, 0, 0]
        start = joint.dofadr[0]
        data.qvel[start:start + 6] = 0
    mujoco.mj_forward(model, data)


# Terminal input (runs in a background thread)
def read_terminal(lines):
    """Wait for typed lines and put them in the queue. Runs forever in a daemon thread."""
    for line in sys.stdin:
        lines.put(line.strip())


def start_input_thread():
    lines = queue.Queue()
    thread = threading.Thread(target=read_terminal, args=(lines,), daemon=True)
    thread.start()
    return lines


def show_prompt():
    print(PROMPT, end="", flush=True)


# Running skills
def make_sim(model, data, amap):
    """All the changing state of the program, in one dictionary."""
    return {
        "model": model,
        "data": data,
        "amap": amap,
        "scratch": mujoco.MjData(model),   # separate data object used by the IK solver
        "mode": "idle",                    # "idle", "moving" or "settling"
        "skills": [],                      # skills waiting to run
        "targets": [],                     # objects we will check at the end
        "retries": 0,
        "keyframes": [],                   # keyframes still to play
        "current": None,                   # keyframe being played
        "start_ctrl": None,                # ctrl at the start of the current keyframe
        "t": 0.0,                          # time inside the current keyframe
        "reset_when_done": False,
        "settle_left": 0.0,
        "quit": False,
    }


def submit_prompt(sim, text):
    """Parse a typed prompt (via Ollama first, with rule-based fallback) and start running it."""
    text = text.strip()
    if text == "":
        show_prompt()
        return
    if text.lower() in ["quit", "exit", "q"]:
        sim["quit"] = True
        return

    positions = pa.read_positions(sim["data"], pa.ACTIVE_OBJECTS)
    ollama_out = op.parse_prompt_ollama(text, positions)
    if ollama_out is not None:
        skills, notes = ollama_out
        print("[LLM] " + str(skills))
    else:
        skills, notes = pa.parse_prompt(text, positions)

    for note in notes:
        print("[Note] " + note)
    if len(skills) == 0:
        show_prompt()
        return

    sim["skills"] = skills
    sim["targets"] = []
    for skill in skills:
        if skill["skill"] == "push" and skill["object"] not in sim["targets"]:
            sim["targets"].append(skill["object"])
    sim["retries"] = 0
    print("[Plan] " + str(len(skills)) + " skill(s) queued")
    start_next_skill(sim)

def start_next_skill(sim):
    """Plan the next skill in the queue and start playing it. If none are left, start settling."""
    while len(sim["skills"]) > 0:
        skill = sim["skills"].pop(0)
        positions = pa.read_positions(sim["data"], pa.ACTIVE_OBJECTS)
        plan, messages = pa.plan_skill(sim["model"], sim["scratch"], sim["amap"], skill, positions,
                                       sim["data"].ctrl.copy())
        for message in messages:
            print("[Plan] " + message)
        if plan is None:
            continue
        print("[Exec] " + plan["description"])
        sim["keyframes"] = list(plan["keyframes"])
        sim["current"] = sim["keyframes"].pop(0)
        sim["start_ctrl"] = sim["data"].ctrl.copy()
        sim["t"] = 0.0
        sim["reset_when_done"] = plan["reset_objects"]
        sim["mode"] = "moving"
        return
    sim["mode"] = "settling"
    sim["settle_left"] = pa.SETTLE_TIME


def interpolate(start, keyframe, alpha):
    """Control vector at progress alpha (0..1) of a keyframe, using cosine easing."""
    u = 0.5 - 0.5 * np.cos(np.pi * alpha)# smooth 0 -> 1 with zero speed at both ends
    points = [start] + keyframe["via"] + [keyframe["ctrl"]]
    if len(points) == 2:
        return points[0] + (points[1] - points[0]) * u
    # With via poses: walk through the points in straight pieces.
    s = u * (len(points) - 1)
    i = min(int(s), len(points) - 2)
    return points[i] + (points[i + 1] - points[i]) * (s - i)


def step_motion(sim):
    """Called once per physics step while moving: set data.ctrl for this step."""
    dt = sim["model"].opt.timestep
    keyframe = sim["current"]
    sim["t"] = sim["t"] + dt
    alpha = min(sim["t"] / keyframe["duration"], 1.0)
    ctrl = interpolate(sim["start_ctrl"], keyframe, alpha)
    sim["data"].ctrl[:] = pa.clamp_ctrl(sim["amap"], ctrl)

    if alpha >= 1.0: # keyframe finished
        sim["start_ctrl"] = keyframe["ctrl"].copy()
        sim["t"] = 0.0
        if len(sim["keyframes"]) > 0:
            sim["current"] = sim["keyframes"].pop(0)
        else:
            sim["current"] = None
            if sim["reset_when_done"]:
                reset_objects(sim["model"], sim["data"])
                sim["reset_when_done"] = False
                sim["targets"] = []
            start_next_skill(sim)


def check_and_replan(sim):
    """Check whether the target objects are in the bin; if not, plan again (up to MAX_AUTO_RETRIES)."""
    positions = pa.read_positions(sim["data"], pa.ACTIVE_OBJECTS)
    lines = []
    missing = []
    for name in sim["targets"]:
        distance = pa.dist_to_bin(positions[name])
        if not pa.on_table(positions[name]):
            lines.append(name + ": OFF TABLE")
        elif distance <= pa.BIN_RADIUS:
            lines.append(name + ": IN BIN (" + format(distance, ".3f") + " m)")
        else:
            lines.append(name + ": NOT in bin (" + format(distance, ".3f") + " m, limit " +
                         format(pa.BIN_RADIUS, ".3f") + " m)")
            missing.append(name)
    if len(sim["targets"]) > 0:
        print("[Check] " + " | ".join(lines))

    if len(missing) > 0 and sim["retries"] < MAX_AUTO_RETRIES:
        sim["retries"] = sim["retries"] + 1
        print("[Replan] attempt " + str(sim["retries"]) + "/" + str(MAX_AUTO_RETRIES) +
              ": pushing again: " + ", ".join(missing))
        sim["skills"] = []
        for name in missing:
            sim["skills"].append({"skill": "push", "object": name, "arm": None, "follow": 0.0,
                                  "attempt": sim["retries"]})
        start_next_skill(sim)
        return

    if len(missing) > 0:
        print("[Check] giving up on: " + ", ".join(missing))
    elif len(sim["targets"]) > 0:
        print("[Check] all targets are in the bin.")
    sim["mode"] = "idle"
    sim["targets"] = []
    show_prompt()


def update(sim):
    """Called once before every physics step."""
    if sim["mode"] == "moving":
        step_motion(sim)
    elif sim["mode"] == "settling":
        sim["settle_left"] = sim["settle_left"] - sim["model"].opt.timestep
        if sim["settle_left"] <= 0:
            check_and_replan(sim)


# Main program

def make_simulation():
    model, data, home_ctrl = build_scene()
    amap = pa.build_actuator_map(model, home_ctrl)
    reset_arms(model, data, amap)
    reset_objects(model, data)
    return model, data, make_sim(model, data, amap)


def run_with_viewer():
    model, data, sim = make_simulation()
    lines = start_input_thread()
    dt = model.opt.timestep
    steps_per_frame = max(1, round(1.0 / 60.0 / dt))# about 60 screen updates per second

    print("ALOHA tidy sim ready. Try: push red block | sweep left | sweep right | tidy everything | "
          "reset | quit")
    with mujoco.viewer.launch_passive(model, data,show_left_ui= False, show_right_ui = False) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.02, 0.28]
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        show_prompt()

        while viewer.is_running() and not sim["quit"]:
            frame_start = time.perf_counter()

            if sim["mode"] == "idle": # take a new command only when not busy
                try:
                    text = lines.get_nowait()
                    submit_prompt(sim, text)
                except queue.Empty:
                    pass

            for i in range(steps_per_frame):
                update(sim)
                mujoco.mj_step(model, data)
            viewer.sync()

            # Sleep for the rest of the frame so the simulation runs in real time.
            spent = time.perf_counter() - frame_start
            time.sleep(max(0.0, steps_per_frame * dt - spent))


def run_headless(commands):
    """Run commands without a viewer (handy for testing)."""
    model, data, sim = make_simulation()
    for command in commands:
        print(PROMPT + command)
        submit_prompt(sim, command)
        start_time = data.time
        while sim["mode"] != "idle" and data.time - start_time < 120:
            update(sim)
            mujoco.mj_step(model, data)
        positions = pa.read_positions(data, pa.ACTIVE_OBJECTS)
        for name in positions:
            print("   " + name + " at " + str(np.round(positions[name], 3)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--headless":
        commands = sys.argv[2:]
        if len(commands) == 0:
            commands = ["push red block"]
        run_headless(commands)
    else:
        run_with_viewer()
