"""Autonomous agent: the LLM picks tools, sees the result (through the camera), and decides the next step."""
import io
import json
import queue
import threading
from contextlib import redirect_stdout

import ollama

import planner_actuators as pa
import llm
import vision as vis

MODEL = llm.DEFAULT_MODEL      # set OLLAMA_MODEL to something stronger than 3b (e.g. qwen2.5:7b)
MAX_STEPS = 15


# Observation (always runs on the main/sim thread)

def observe(sim):
    """What the agent knows: camera-based object positions plus the robot's own gripper sensors."""
    data, amap = sim["data"], sim["amap"]
    positions = vis.believed_positions(sim)
    if sim["vision_on"]:
        visible = sim["visible"]
    else:
        visible = {name: True for name in positions}
    held = pa.held_objects(data, amap, sim["grasped"])         # {side: object}, from finger sensors
    held_by = {obj: side for side, obj in held.items()}

    objects = {}
    for name, p in positions.items():
        seen = bool(visible.get(name, False))
        entry = {"visible": seen, "held_by": held_by.get(name)}
        if seen:
            entry.update({
                "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
                "on_table": bool(pa.on_table(p)),
                # a carried object is NOT in the bin, even if it is above it
                "in_bin": bool(pa.in_bin(p)) and name not in held_by,
                "dist_to_bin": round(pa.dist_to_bin(p), 3),
                "side": "left" if p[0] < pa.TABLE_CENTER_XY[0] else "right",
            })
        else:
            entry.update({"xy": None, "on_table": False, "in_bin": False})
            if name not in held_by:
                entry["note"] = "not visible to the camera (off the table or hidden)"
        objects[name] = entry

    arms = {}
    attention = []
    for side in pa.SIDES:
        arm = amap[side]
        g = float(data.ctrl[arm["grip_act"]])
        mid = 0.5 * (amap["lo"][arm["grip_act"]] + amap["hi"][arm["grip_act"]])
        arms[side] = {"gripper": "open" if g > mid else "closed", "holding": held.get(side)}
        if side in held:
            attention.append(f"{side} arm is still holding {held[side]} in the air. "
                             f"Call release_gripper(arm='{side}') to drop it, then return_arm_home.")
        elif side in sim["grasped"] and g <= mid:
            attention.append(f"{side} arm tried to grab {sim['grasped'][side]} but its gripper closed on "
                             f"nothing (grasp failed). Call release_gripper and return_arm_home, "
                             f"then look again and retry or use push_to_bin.")
    result = {"objects": objects, "arms": arms, "bin_radius": pa.BIN_RADIUS,
              "perception": "overhead camera, positions are estimates (a centimetre or two of error)"
                            if sim["vision_on"] else "ground truth"}
    if attention:
        result["attention"] = attention
    return result


# Thread bridge

def make_bridge():
    """State shared between the agent thread and the main (sim) thread, in one dictionary."""
    return {"q": queue.Queue(),          # jobs waiting for the main thread
            "active": None,              # job currently running on the main thread
            "agent_running": False}


def bridge_call(bridge, kind, skill=None, timeout=300):
    """Called from the agent thread. Blocks until the main thread has finished the job.

    kind: "state" (latest belief, instant), "look" (take a new picture first) or "skill".
    """
    job = {"kind": kind, "skill": skill, "done": threading.Event(),
           "result": None, "log": []}
    bridge["q"].put(job)
    if not job["done"].wait(timeout):
        return {"error": "timed out"}
    return job["result"]


def pump_bridge(bridge, sim, submit_skills):
    """Called once per frame from the main loop."""
    # 1. A running job is finished once the sim is idle again and the picture is up to date.
    if bridge["active"] is not None and sim["mode"] == "idle":
        if sim["vision_on"] and sim["belief_stale"]:
            vis.start_perception(sim, lambda s: None)       # look before reporting back
            return
        job, bridge["active"] = bridge["active"], None
        job["result"] = {"planner_log": job["log"], **observe(sim)}
        job["done"].set()

    # 2. Start the next queued job.
    if bridge["active"] is None and sim["mode"] == "idle" and not bridge["q"].empty():
        job = bridge["q"].get_nowait()
        if job["kind"] == "state":
            job["result"] = observe(sim)
            job["done"].set()
        elif job["kind"] == "look":
            if sim["vision_on"]:
                vis.start_perception(sim, lambda s: None)
            bridge["active"] = job
        else:
            buf = io.StringIO()
            with redirect_stdout(buf):               # capture planner messages
                submit_skills(sim, [job["skill"]])
            job["log"] = [l for l in buf.getvalue().splitlines() if l.strip()]
            if job["log"]:
                print("\n".join(job["log"]))
            bridge["active"] = job


# Tools

def _tool(name, description, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props or {},
                       "required": required or []}}}


OBJ = {"type": "string", "enum": list(pa.OBJECT_SPECS)}   # whole catalog; run_tool rejects ones not on the table
ARM = {"type": "string", "enum": ["left", "right"]}

TOOLS = [
    _tool("get_scene", "Look at the table with the overhead camera. Returns where each object is "
          "(estimated from the image, so it may be off by a centimetre or two), whether it is visible, "
          "in the bin or held, and the state of each gripper."),
    _tool("push_to_bin", "Push an object across the table into the bin. The arm is chosen automatically "
          "if omitted. The arm returns home by itself afterwards.",
          {"object": OBJ, "arm": ARM}, ["object"]),
    _tool("grab_object", "Pick up an object and carry it above the bin. The gripper stays CLOSED "
          "holding it; you must call release_gripper afterwards to drop it in the bin.",
          {"object": OBJ, "arm": ARM}, ["object"]),
    _tool("release_gripper", "Open the gripper of an arm (drops whatever it holds).",
          {"arm": ARM}, ["arm"]),
    _tool("return_arm_home", "Move an arm (or both if omitted) back to its home pose.", {"arm": ARM}),
    _tool("reset_scene", "Reset arms to home and put the objects in a new random layout."),
    _tool("finish", "Call when the task is complete or impossible.",
          {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ["success", "summary"]),
]

SYSTEM = (
    "You control a two-arm ALOHA robot at a table with a circular bin. "
    "You cannot read object positions directly: they come from an overhead camera. "
    "Achieve the user's task by calling tools, one at a time.\n"
    "Rules:\n"
    "- Call get_scene first. Every action's result also contains a fresh look at the table, so read it "
    "and verify that the action worked.\n"
    "- Object positions are estimates. If an object is not visible, it may have fallen off the table or "
    "be hidden; look again before giving up.\n"
    "- If an action failed (object not in_bin, planner_log shows an error, an 'attention' message), try a "
    "different arm or a different skill (e.g. grab instead of push). Do not repeat the same failing call "
    "more than once.\n"
    "- After grab_object you must release_gripper, then return_arm_home.\n"
    "- An object that is held is NOT in the bin yet. If a result has an 'attention' field or an arm has "
    "'holding' set, follow it: release_gripper, then return_arm_home.\n"
    "- Only claim success if the latest result confirms it (in_bin is true and nothing is held).\n"
    "- Call finish as soon as the task is satisfied, or when it cannot be done."
)


# Executing a tool call

def run_tool(bridge, name, args):
    arm = args.get("arm") if args.get("arm") in pa.SIDES else None

    if name == "get_scene":
        return bridge_call(bridge, "look")

    if name in ("push_to_bin", "grab_object"):
        objs = bridge_call(bridge, "state")["objects"]
        obj = args.get("object")
        if obj not in objs:
            return {"error": f"unknown object {obj!r}. Known: {list(objs)}"}
        if objs[obj].get("held_by"):
            return {"error": f"{obj} is already held by the {objs[obj]['held_by']} arm"}
        if not objs[obj]["visible"]:
            return {"error": f"{obj} is not visible to the camera. Call get_scene to look again."}
        if not objs[obj]["on_table"]:
            return {"error": f"{obj} is not on the table"}
        if objs[obj]["in_bin"]:
            return {"error": f"{obj} is already in the bin"}
        maker = pa.make_push if name == "push_to_bin" else pa.make_grab
        return bridge_call(bridge, "skill", maker(obj, arm, 0.0))

    if name == "release_gripper":
        return bridge_call(bridge, "skill", {"skill": "drop", "arm": arm})
    if name == "return_arm_home":
        return bridge_call(bridge, "skill", {"skill": "return", "arm": arm})
    if name == "reset_scene":
        return bridge_call(bridge, "skill", {"skill": "reset"})
    return {"error": f"unknown tool {name}"}


# Agent loop (runs in its own thread)

def run_agent(task, bridge, model=MODEL):
    episodes = llm.load_recent_episodes(limit=5, prompt_text=task)
    memory_block = llm.format_episodes_for_prompt(episodes)
    system = SYSTEM + ("\n\n" + memory_block if memory_block else "")

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task}]
    trace, success, summary = [], False, "Step limit reached."
    seen = {}
    finish_blocks = 0
    try:
        for _ in range(MAX_STEPS):
            resp = ollama.chat(model=model, messages=messages, tools=TOOLS,
                               options={"temperature": 0})
            msg = resp.message
            messages.append(msg)
            if not msg.tool_calls:
                summary = msg.content or "Stopped without calling finish."
                break

            done = False
            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                print(f"[Agent] {name}({args})")
                if name == "finish":
                    still = [f"{s} arm is holding {a['holding']}"
                             for s, a in bridge_call(bridge, "state")["arms"].items() if a.get("holding")]
                    if still and finish_blocks < 2:
                        finish_blocks += 1
                        messages.append({"role": "tool", "tool_name": name, "content": json.dumps(
                            {"error": "Cannot finish yet: " + "; ".join(still) +
                                      ". Call release_gripper first, then return_arm_home."})})
                        continue
                    success = bool(args.get("success"))
                    summary = args.get("summary", "")
                    done = True
                    break
                key = (name, json.dumps(args, sort_keys=True))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 2 and name != "get_scene":
                    result = {"error": "You already tried this exact call twice. "
                                       "Try something different or finish."}
                else:
                    result = run_tool(bridge, name, args)
                trace.append({"skill": name, "args": args})
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result)})
            if done:
                break
    except Exception as exc:                      # e.g. Ollama not running
        summary = f"Agent error: {exc}"
    finally:
        print(f"[Agent] {'SUCCESS' if success else 'DONE'}: {summary}")
        llm.record_episode(task, trace, summary, success)
        bridge["agent_running"] = False
        print("> ", end="", flush=True)
