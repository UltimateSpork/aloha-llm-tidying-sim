import mujoco
import numpy as np


TABLE_CENTER_XY = np.array([0.0, -0.019])
TABLE_TOP_Z = 0.24
TABLE_HALF_XY = (0.24, 0.30)
TABLE_THICKNESS = 0.04
GRASP_LIFT_DZ = 0.08      # height lifted above object center during carry
GRASP_CLEAR_DZ = 0.02     # small clearance above grasp center on approach
BIN_CENTER_XY = np.array([0.0, -0.019])
BIN_RADIUS = 0.07 # an object is "in the bin" if its centre is within this distance

# Objects
OBJECT_SPECS = {
    "red_block": {"shape": "box", "half": (0.025, 0.025, 0.025), "rgba": (0.85, 0.10, 0.10, 1),
                  "mass": 0.08, "xy": (-0.14, 0.03), "radius": 0.025,
                  "friction": (0.8, 0.005, 0.0001), "damping": 0.3},
    "blue_ball": {"shape": "sphere", "half": (0.03,), "rgba": (0.15, 0.30, 0.90, 1),
                  "mass": 0.06, "xy": (0.14, -0.10), "radius": 0.03,
                  "friction": (0.8, 0.02, 0.02), "damping": 0.6},
    "green_box": {"shape": "box", "half": (0.035, 0.025, 0.025), "rgba": (0.10, 0.70, 0.25, 1),
                  "mass": 0.10, "xy": (-0.11, -0.12), "radius": 0.035,
                  "friction": (0.8, 0.005, 0.0001), "damping": 0.3},
}
ACTIVE_OBJECTS = ["red_block", "blue_ball"]

# Planner settings
PUSH_TIP_Z = TABLE_TOP_Z + 0.04     # fingertip height while pushing
HOVER_DZ = 0.07                     # extra height when approaching / lifting
WINDUP_GAP = 0.06                   # gap between fingertip and object at the windup pose
REACH_TOL = 0.02                    # IK error (m) that still counts as "reached"
MAX_ACCEPT_ERR = 0.05               # worse than this -> arm can't do the skill
LEVEL_WEIGHT = 0.35                 # weight of the "keep gripper level" term
BIN_ENTRY_FRAC = 0.4                # landing spot = this fraction of the bin radius from the centre
SETTLE_TIME = 0.6                   # seconds to wait before checking the bin

GRIPPER_TIP_LOCAL = np.array([0.145, 0.0, 0.0])   # fingertip position in the gripper_link frame
WRIST_TIP_LOCAL = np.array([0.215, 0.0, 0.0])     # same point in the wrist_link frame (fallback)

SIDES = ["left", "right"]
ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

SYNONYMS = {"cube": "block", "box": "block", "sphere": "ball"}


def object_rest_z(name):
    """Centre height of an object sitting on the table."""
    cfg = OBJECT_SPECS[name]
    if cfg["shape"] == "box":
        half_z = cfg["half"][2]
    else:
        half_z = cfg["half"][0]
    return TABLE_TOP_Z + half_z + 0.001


# Actuator map
def build_actuator_map(model, home_ctrl):
    """Look at model.nu / actuator_ctrlrange and build a dict describing both arms.

    Returns a dict with keys: "lo", "hi", "home", "left", "right".
    Each arm entry is a dict with the actuator ids, joint addresses, limits and end-effector body.
    """
    if model.nu != 14:
        raise RuntimeError("Expected 14 actuators for ALOHA, found " + str(model.nu))

    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    home = np.clip(home_ctrl, lo, hi)
    scratch = mujoco.MjData(model)
    mujoco.mj_forward(model, scratch)

    amap = {"lo": lo, "hi": hi, "home": home}
    for side in SIDES:
        act_ids = []
        for joint in ARM_JOINTS:
            name = side + "/" + joint
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index < 0:
                raise RuntimeError("Actuator not found: " + name)
            act_ids.append(index)
        grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, side + "/gripper")
        act_ids = np.array(act_ids)
        joint_ids = model.actuator_trnid[act_ids, 0]

        # End effector: gripper_link, or wrist_link if that does not exist.
        ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "/gripper_link")
        tip_local = GRIPPER_TIP_LOCAL
        if ee_body < 0:
            ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "/wrist_link")
            tip_local = WRIST_TIP_LOCAL

        base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "/base_link")
        amap[side] = {
            "side": side,
            "act_ids": act_ids,
            "grip_act": grip_act,
            "qpos_adr": model.jnt_qposadr[joint_ids],
            "dof_adr": model.jnt_dofadr[joint_ids],
            # joint limits, also limited by the actuator control range
            "q_lo": np.maximum(model.jnt_range[joint_ids, 0], lo[act_ids]),
            "q_hi": np.minimum(model.jnt_range[joint_ids, 1], hi[act_ids]),
            "home_q": home[act_ids].copy(),
            "ee_body": ee_body,
            "tip_local": tip_local,
            "base_xy": scratch.xpos[base_body][:2].copy(),
        }
    return amap


def clamp_ctrl(amap, ctrl):
    """Clamp a control vector to actuator_ctrlrange."""
    return np.clip(ctrl, amap["lo"], amap["hi"])


# Object positions
def read_positions(data, names):
    """Return {object name: xyz position}."""
    positions = {}
    for name in names:
        positions[name] = data.body(name).xpos.copy()
    return positions


def dist_to_bin(pos):
    return float(np.linalg.norm(pos[:2] - BIN_CENTER_XY))


def in_bin(pos):
    return dist_to_bin(pos) <= BIN_RADIUS


def on_table(pos):
    return pos[2] > TABLE_TOP_Z - 0.03


# Prompt parsing
def has_word(words, options):
    """True if any of the options appears in the list of words."""
    for option in options:
        if option in words:
            return True
    return False


def match_objects(words, names):
    """Return the object names that best match the words (e.g. 'red', 'block')."""
    best_score = 0
    scores = {}
    for name in names:
        score = 0
        for token in name.split("_"):
            token = SYNONYMS.get(token, token)      # "green_box" is matched as "green block"
            if token in words:
                score = score + 1
        scores[name] = score
        if score > best_score:
            best_score = score
    matches = []
    for name in names:
        if best_score > 0 and scores[name] == best_score:
            matches.append(name)
    return matches


def make_push(name, arm, follow):
    return {"skill": "push", "object": name, "arm": arm, "follow": follow, "attempt": 0}

def make_grab(name, arm, follow):
    return {"skill": "grab", "object": name, "arm": arm, "follow": follow, "attempt": 0}



def sort_by_bin_distance(names, positions):
    """Nearest-to-bin first."""
    return sorted(names, key=lambda n: dist_to_bin(positions[n]))


def parse_prompt(text, positions):
    """Turn a prompt into (skills, notes).

    Skill dicts look like:
      {"skill": "push", "object": name, "arm": None / "left" / "right", "follow": 0.0, "attempt": 0}
      {"skill": "sweep", "side": "left" / "right"}     (empty sweep stroke)
      {"skill": "reset"}
      {"skill: grab", "object":name, "arm": None / "left" / "right", "follow": 0.0, "attempt": 0}
    "tidy everything" and "sweep <side>" are expanded using the current object positions.
    """
    skills = []
    notes = []
    names = list(positions.keys())

    # Make every separator a comma, then split into clauses ("sweep left and then sweep right").
    text = text.lower().strip()
    for separator in [" and then ", " then ", " and ", ";"]:
        text = text.replace(separator, ",")

    last_verb = None
    for clause in text.split(","):
        clause = clause.strip()
        if clause == "":
            continue
        words = []
        for w in clause.split():
            words.append(SYNONYMS.get(w, w))
        objs = match_objects(words, names)

        if has_word(words, ["reset", "restart", "home"]):
            skills.append({"skill": "reset"})
            last_verb = "reset"
            continue

        # Decide what the clause wants to do.
        verb = None
        if "sweep" in words:
            verb = "sweep"
        elif has_word(words, ["tidy", "clean", "clear", "everything", "all", "both"]) and len(objs) == 0:
            verb = "tidy"
        elif has_word(words, ["push", "move", "shove", "nudge", "slide", "put", "bring", "knock"]):
            verb = "push"
        elif has_word(words, ["tidy", "clean", "clear"]):
            verb = "push"
        elif has_word(words, ["grab","pick","take"]):
            verb = "grab"
        elif len(objs) > 0 and last_verb in ["push", "sweep", "tidy"]:
            verb = "push"       # e.g. "push red block and blue ball"
        if verb is None:
            notes.append("Didn't understand '" + clause + "'. Try: push red block | sweep left | "
                         "sweep right | tidy everything | reset")
            continue
        last_verb = verb

        if verb == "tidy":
            loose = []
            for n in names:
                if on_table(positions[n]) and not in_bin(positions[n]):
                    loose.append(n)
            if len(loose) == 0:
                notes.append("Everything is already in the bin.")
            for n in sort_by_bin_distance(loose, positions):
                skills.append(make_push(n, None, 0.0))

        elif verb == "sweep":
            side = None
            if "left" in words:
                side = "left"
            elif "right" in words:
                side = "right"
            if len(objs) > 0:
                for n in objs:
                    if not in_bin(positions[n]):
                        skills.append(make_push(n, side, 0.03))
            elif side is None:
                notes.append("Sweep which side? For example: sweep left")
            else:
                # The left arm owns the table half with x < centre, the right arm the other half.
                mine = []
                for n in names:
                    left_half = positions[n][0] < TABLE_CENTER_XY[0]
                    if (side == "left") == left_half and on_table(positions[n]) and not in_bin(positions[n]):
                        mine.append(n)
                if len(mine) == 0:
                    notes.append("No loose objects on the " + side + " side; doing an empty sweep.")
                    skills.append({"skill": "sweep", "side": side})
                for n in sort_by_bin_distance(mine, positions):
                    skills.append(make_push(n, side, 0.03))

        elif verb == "push":
            if len(objs) == 0:
                loose = []
                for n in names:
                    if not in_bin(positions[n]):
                        loose.append(n)
                if len(loose) == 1:
                    objs = loose
                else:
                    notes.append("Which object? Known objects: " + ", ".join(names))
                    continue
            for n in objs:
                if in_bin(positions[n]):
                    notes.append(n + " is already in the bin.")
                else:
                    skills.append(make_push(n, None, 0.0))
        else: # Picks up object
            side = None
            if "left" in words:
                side = "left"
            else:
                side = "right"
                            # The left arm owns the table half with x < centre, the right arm the other half.
            mine = []
            for n in names:
                left_half = positions[n][0] < TABLE_CENTER_XY[0]
                if (side == "left") == left_half and on_table(positions[n]):
                    mine.append(n)
            if len(mine) == 0:
                notes.append("No loose objects on the " + side )
                skills.append({"skill": "grab", "side": side})
            for n in sort_by_bin_distance(mine, positions):
                skills.append(make_grab(n, side, 0.03))


    return skills, notes


# Damped least-squares IK
def skew(v):
    """Matrix form of the cross product: skew(a) @ b == cross(a, b)."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def solve_ik(model, scratch, arm, target, q_seed):
    """Move the fingertip of `arm` to `target` (xyz) using damped least squares.

    Works on a scratch MjData so the real simulation is never disturbed.
    Returns (q, error): the 6 arm joint angles and the final fingertip error in metres.
    Results stay inside the joint limits / actuator ctrlrange.
    """
    damping = 0.04
    posture_gain = 0.15
    q = np.clip(q_seed, arm["q_lo"], arm["q_hi"])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    error = 999.0

    for i in range(200):
        scratch.qpos[arm["qpos_adr"]] = q
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        rotation = scratch.xmat[arm["ee_body"]].reshape(3, 3)
        r = rotation @ arm["tip_local"]                       # link origin -> fingertip (world frame)
        tip = scratch.xpos[arm["ee_body"]] + r
        pos_error = target - tip
        error = float(np.linalg.norm(pos_error))

        # "Keep level": rotate the gripper's forward axis toward its horizontal projection.
        forward = rotation[:, 0]
        flat = np.array([forward[0], forward[1], 0.0])
        if np.linalg.norm(flat) > 1e-6:
            level_error = np.cross(forward, flat / np.linalg.norm(flat))
        else:
            level_error = np.zeros(3)

        if error < 0.002 and np.linalg.norm(level_error) < 0.03:
            break

        mujoco.mj_jacBody(model, scratch, jacp, jacr, arm["ee_body"])
        J_rot = jacr[:, arm["dof_adr"]]
        J_pos = jacp[:, arm["dof_adr"]] - skew(r) @ J_rot     # Jacobian of the fingertip point

        J = np.vstack([J_pos, LEVEL_WEIGHT * J_rot])
        e = np.concatenate([pos_error, LEVEL_WEIGHT * level_error])

        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(6))      # damped pseudo-inverse
        dq = J_pinv @ e
        null_space = np.eye(6) - J_pinv @ J
        dq = dq + null_space @ (posture_gain * (arm["home_q"] - q))           # drift toward home pose

        step = np.linalg.norm(dq)
        if step > 0.25:
            dq = dq * (0.25 / step)                           # limit the step size
        q = np.clip(q + dq, arm["q_lo"], arm["q_hi"])
    return q, error


# ----------------------------------------------------------------------------------------
# Keyframe planning
# ----------------------------------------------------------------------------------------
def make_keyframe(ctrl, duration, label, via=None):
    """A keyframe: reach `ctrl` in `duration` seconds, optionally passing through `via` poses."""
    if via is None:
        via = []
    return {"ctrl": ctrl, "duration": duration, "label": label, "via": via}


def duration_for(q_from, q_to, speed, minimum):
    """Time needed to move between two joint vectors at a given max joint speed (rad/s)."""
    return max(minimum, float(np.max(np.abs(q_to - q_from))) / speed)


def make_ctrl(amap, base_ctrl, arm, q, gripper):
    """Copy base_ctrl and set one arm's six joints and gripper."""
    ctrl = base_ctrl.copy()
    ctrl[arm["act_ids"]] = q
    ctrl[arm["grip_act"]] = gripper
    return clamp_ctrl(amap, ctrl)


def bin_slot(p_obj, n_in_bin):
    """Where an object should end up inside the bin.

    Just inside the bin edge on the side it comes from, shifted sideways by the number of objects that
    are already in the bin, so later pushes don't shove earlier objects back out.
    """
    d = BIN_CENTER_XY - p_obj[:2]
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    sideways = np.array([-d[1], d[0]])
    offsets = [0, 1, -1, 2, -2]
    lateral = offsets[n_in_bin % 5] * 0.03
    return BIN_CENTER_XY - d * BIN_ENTRY_FRAC * BIN_RADIUS + sideways * lateral


def plan_push_for_arm(model, scratch, amap, arm, p_obj, r_obj, follow, attempt, start_ctrl, target_xy):
    """Plan approach -> windup -> push -> lift -> retract for ONE arm.

    If the windup pose is hard to reach, retry with a smaller gap behind the object.
    Returns a plan dict (the best one found) or None.
    """
    vec = target_xy - p_obj[:2]
    dist = float(np.linalg.norm(vec))
    if dist < 1e-4:
        return None
    d = vec / dist                                    # direction: object -> bin
    gripper_closed = amap["lo"][arm["grip_act"]]
    best = None

    for gap in [WINDUP_GAP, 0.04, 0.025]:
        windup_xy = p_obj[:2] - d * (r_obj + gap)                       # behind the object
        end_xy = target_xy - d * r_obj + d * (follow + 0.02 * attempt)  # object centre ends at the slot

        hover_tip = np.array([windup_xy[0], windup_xy[1], PUSH_TIP_Z + HOVER_DZ])
        windup_tip = np.array([windup_xy[0], windup_xy[1], PUSH_TIP_Z])
        end_tip = np.array([end_xy[0], end_xy[1], PUSH_TIP_Z])
        lift_tip = np.array([end_xy[0], end_xy[1], PUSH_TIP_Z + HOVER_DZ])

        # Solve IK for each waypoint, using the previous answer as the starting guess.
        q_hover, err1 = solve_ik(model, scratch, arm, hover_tip, arm["home_q"])
        q_windup, err2 = solve_ik(model, scratch, arm, windup_tip, q_hover)
        worst = max(err1, err2)
        q_guess = q_windup
        via_q = []
        for k in range(1, 5):                                           # points along the push line
            point = windup_tip + (end_tip - windup_tip) * (k / 5.0)
            q_point, err = solve_ik(model, scratch, arm, point, q_guess)
            via_q.append(q_point)
            q_guess = q_point
            worst = max(worst, err)
        q_end, err3 = solve_ik(model, scratch, arm, end_tip, q_guess)
        q_lift, err4 = solve_ik(model, scratch, arm, lift_tip, q_end)
        worst = max(worst, err3, err4)

        # Turn joint angles into full 14-value control vectors.
        via_ctrl = []
        for q in via_q:
            via_ctrl.append(make_ctrl(amap, start_ctrl, arm, q, gripper_closed))
        home_ctrl = start_ctrl.copy()
        home_ctrl[arm["act_ids"]] = amap["home"][arm["act_ids"]]
        home_ctrl[arm["grip_act"]] = amap["home"][arm["grip_act"]]
        push_time = max(1.5, float(np.linalg.norm(end_xy - windup_xy)) / 0.07)

        keyframes = [
            make_keyframe(make_ctrl(amap, start_ctrl, arm, q_hover, gripper_closed),
                          duration_for(arm["home_q"], q_hover, 0.8, 0.8), arm["side"] + ": approach"),
            make_keyframe(make_ctrl(amap, start_ctrl, arm, q_windup, gripper_closed), 0.9,
                          arm["side"] + ": windup"),
            make_keyframe(make_ctrl(amap, start_ctrl, arm, q_end, gripper_closed), push_time,
                          arm["side"] + ": push", via_ctrl),
            make_keyframe(make_ctrl(amap, start_ctrl, arm, q_lift, gripper_closed), 0.8,
                          arm["side"] + ": lift"),
            make_keyframe(clamp_ctrl(amap, home_ctrl), duration_for(q_lift, arm["home_q"], 0.9, 1.0),
                          arm["side"] + ": retract"),
        ]
        plan = {"keyframes": keyframes, "arm": arm["side"], "max_err": worst,
                "approximate": worst > REACH_TOL, "reset_objects": False, "description": ""}
        if best is None or worst < best["max_err"]:
            best = plan
        if worst <= REACH_TOL:
            break
    return best


def plan_grab_for_arm(model, scratch, amap, arm, p_obj, r_obj, start_ctrl, target_xy):
    """Plan hover -> descend -> close -> lift -> carry to bin -> release -> retract."""
    grip_open = amap["hi"][arm["grip_act"]]
    grip_closed = amap["lo"][arm["grip_act"]]

    xy = p_obj[:2]
    grasp_z = max(TABLE_TOP_Z + 0.015, float(p_obj[2]))
    hover_tip = np.array([xy[0], xy[1], grasp_z + HOVER_DZ])
    grasp_tip = np.array([xy[0], xy[1], grasp_z])
    lift_tip = np.array([xy[0], xy[1], grasp_z + GRASP_LIFT_DZ])
    drop_hover_tip = np.array([target_xy[0], target_xy[1], grasp_z + GRASP_LIFT_DZ + GRASP_CLEAR_DZ])
    drop_tip = np.array([target_xy[0], target_xy[1], grasp_z + GRASP_LIFT_DZ])

    q_hover, err1 = solve_ik(model, scratch, arm, hover_tip, arm["home_q"])
    q_grasp, err2 = solve_ik(model, scratch, arm, grasp_tip, q_hover)
    q_lift, err3 = solve_ik(model, scratch, arm, lift_tip, q_grasp)
    q_drop_hover, err4 = solve_ik(model, scratch, arm, drop_hover_tip, q_lift)
    q_drop, err5 = solve_ik(model, scratch, arm, drop_tip, q_drop_hover)

    worst = max(err1, err2, err3, err4, err5)

    home_ctrl = start_ctrl.copy()
    home_ctrl[arm["act_ids"]] = amap["home"][arm["act_ids"]]
    home_ctrl[arm["grip_act"]] = grip_open

    carry_time = max(1.2, float(np.linalg.norm(target_xy - xy)) / 0.08)

    keyframes = [
        make_keyframe(make_ctrl(amap, start_ctrl, arm, q_hover, grip_open),
                      duration_for(arm["home_q"], q_hover, 0.8, 0.8), arm["side"] + ": pre-grasp"),
        make_keyframe(make_ctrl(amap, start_ctrl, arm, q_grasp, grip_open), 0.7,
                      arm["side"] + ": descend"),
        make_keyframe(make_ctrl(amap, start_ctrl, arm, q_grasp, grip_closed), 0.5,
                      arm["side"] + ": close gripper"),
        make_keyframe(make_ctrl(amap, start_ctrl, arm, q_lift, grip_closed), 0.7,
                      arm["side"] + ": lift"),
        make_keyframe(make_ctrl(amap, start_ctrl, arm, q_drop_hover, grip_closed), carry_time,
                      arm["side"] + ": carry to bin"),
        make_keyframe(make_ctrl(amap, start_ctrl, arm, q_drop, grip_open), 0.5,
                      arm["side"] + ": release"),
        make_keyframe(clamp_ctrl(amap, home_ctrl), duration_for(q_drop, arm["home_q"], 0.9, 1.0),
                      arm["side"] + ": retract"),
    ]

    return {
        "keyframes": keyframes,
        "arm": arm["side"],
        "max_err": worst,
        "approximate": worst > REACH_TOL,
        "reset_objects": False,
        "description": "",
    }



def plan_skill(model, scratch, amap, skill, positions, start_ctrl):
    """Turn one skill dict into a plan using the CURRENT object positions.

    Returns (plan, messages); plan is None if the skill is impossible.

    Arm selection: if the skill doesn't force an arm, both arms are planned and the one with the smaller
    IK error wins (the arm on the object's side gets a small bonus). If the best arm can only reach the
    windup pose approximately, a message says so; if it is way off, the skill is rejected.
    """
    messages = []
    kind = skill["skill"]

    if kind == "reset":
        keyframes = [make_keyframe(amap["home"].copy(), 1.8, "reset: home")]
        plan = {"keyframes": keyframes, "arm": None, "max_err": 0.0, "approximate": False,
                "reset_objects": True, "description": "reset"}
        return plan, messages

    if kind == "sweep":                               # empty sweep: a push stroke with no object
            if "object" not in skill:
                side = skill.get("side", "left")
                sign = -1.0 if side == "left" else 1.0
                p_obj = np.array([sign * 0.17, TABLE_CENTER_XY[1], 0.0])
                r_obj = 0.0
                follow = 0.0
                attempt = 0
                forced_arm = side
                name = "empty " + side + " sweep"
                n_in_bin = 0
            else:
                name = skill["object"]
                if not on_table(positions[name]):
                    return None, [name + " is not on the table any more."]
                p_obj = positions[name]
                r_obj = OBJECT_SPECS[name]["radius"]
                follow = skill["follow"]
                attempt = skill["attempt"]
                forced_arm = skill["arm"]
                n_in_bin = sum(1 for other in positions if other != name and in_bin(positions[other]))
    else:
        if "object" not in skill:
            return None, ["No target object specified for " + kind + "."]
        name = skill["object"]
        if not on_table(positions[name]):
            return None, [name + " is not on the table any more."]
        p_obj = positions[name]
        r_obj = OBJECT_SPECS[name]["radius"]
        follow = skill.get("follow", 0.0)
        attempt = skill.get("attempt", 0)
        forced_arm = skill.get("arm")
        n_in_bin = sum(1 for other in positions if other != name and in_bin(positions[other]))

    target_xy = bin_slot(p_obj, n_in_bin)
    if p_obj[0] < TABLE_CENTER_XY[0]:
        preferred = "left"
    else:
        preferred = "right"

    if forced_arm is not None:
        candidates = [forced_arm]
    else:
        candidates = SIDES

    best_plan = None
    best_score = 999.0
    for side in candidates:
        if kind == "grab":
            plan = plan_grab_for_arm(model, scratch, amap, amap[side], p_obj, r_obj, follow, attempt,start_ctrl, target_xy)
        else:
            plan = plan_push_for_arm(model, scratch, amap, amap[side], p_obj, r_obj, follow, attempt,start_ctrl, target_xy)
        if plan is None:
            continue
        score = plan["max_err"]
        if side != preferred:
            score = score + 0.015
        if score < best_score:
            best_score = score
            best_plan = plan

    if best_plan is None:
        return None, ["Could not plan a push for " + name + "."]
    if best_plan["max_err"] > MAX_ACCEPT_ERR:
        return None, [name + ": neither arm can reach (IK error " +
                      str(round(best_plan["max_err"] * 100, 1)) + " cm)."]
    if best_plan["approximate"]:
        messages.append(name + ": the " + best_plan["arm"] + " arm only approximately reaches (IK error " +
                        str(round(best_plan["max_err"] * 100, 1)) + " cm).")
    best_plan["description"] = skill["skill"] + " " + name + " with " + best_plan["arm"] + " arm"
    return best_plan, messages
