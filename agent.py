"""Autonomous agent: the LLM picks tools, sees the simulation result, and decides the next step."""
import io
import json
import queue
import threading
from contextlib import redirect_stdout

import ollama

import planner_actuators as pa
import llm

MODEL = llm.DEFAULT_MODEL # set OLLAMA_MODEL to something stronger than 3b (e.g. qwen2.5:7b)
MAX_STEPS = 15


# Observation

def observe(sim):
    """Compact state the agent sees after every action."""
    data, amap = sim["data"], sim["amap"]
    positions = pa.read_positions(data, pa.ACTIVE_OBJECTS)
    objects = {}
    for name, p in positions.items():
        objects[name] = {
            "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
            "on_table": bool(pa.on_table(p)),
            "in_bin": bool(pa.in_bin(p)),
            "dist_to_bin": round(pa.dist_to_bin(p), 3),
            "side": "left" if p[0] < pa.TABLE_CENTER_XY[0] else "right",
        }
    arms = {}
    for side in pa.SIDES:
        arm = amap[side]
        g = float(data.ctrl[arm["grip_act"]])
        mid = 0.5 * (amap["lo"][arm["grip_act"]] + amap["hi"][arm["grip_act"]])
        arms[side] = {"gripper": "open" if g > mid else "closed"}
    return {"objects": objects, "arms": arms, "bin_radius": pa.BIN_RADIUS}


# Thread bridge

class Bridge:
    """Lets the agent thread request work that must run on the main (sim) thread."""

    def __init__(self):
        self.q = queue.Queue()
        self.active = None
        self.agent_running = False

    def call(self, kind, skill=None, timeout=300):
        """Called from the agent thread. Blocks until the main thread has finished the job."""
        job = {"kind": kind, "skill": skill, "done": threading.Event(),
               "result": None, "log": []}
        self.q.put(job)
        if not job["done"].wait(timeout):
            return {"error": "timed out"}
        return job["result"]

    def pump(self, sim, submit_skills):
        """Called once per frame from the main loop."""
        # 1. A running job has finished once the sim is idle again.
        if self.active is not None and sim["mode"] == "idle":
            job, self.active = self.active, None
            job["result"] = {"planner_log": job["log"], **observe(sim)}
            job["done"].set()

        # 2. Start the next queued job.
        if self.active is None and sim["mode"] == "idle" and not self.q.empty():
            job = self.q.get_nowait()
            if job["kind"] == "scene":
                job["result"] = observe(sim)
                job["done"].set()
            else:
                buf = io.StringIO()
                with redirect_stdout(buf):               # capture planner messages
                    submit_skills(sim, [job["skill"]])
                job["log"] = [l for l in buf.getvalue().splitlines() if l.strip()]
                if job["log"]:
                    print("\n".join(job["log"]))
                self.active = job


# Tools

def _tool(name, description, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props or {},
                       "required": required or []}}}


OBJ = {"type": "string", "enum": list(pa.ACTIVE_OBJECTS)}
ARM = {"type": "string", "enum": ["left", "right"]}

TOOLS = [
    _tool("get_scene", "Get current object positions, whether each is in the bin, and gripper states."),
    _tool("push_to_bin", "Push an object across the table into the bin. The arm is chosen automatically "
          "if omitted. The arm returns home by itself afterwards.",
          {"object": OBJ, "arm": ARM}, ["object"]),
    _tool("grab_object", "Pick up an object and carry it above the bin. The gripper stays CLOSED "
          "holding it; you must call release_gripper afterwards to drop it in the bin.",
          {"object": OBJ, "arm": ARM}, ["object"]),
    _tool("release_gripper", "Open the gripper of an arm (drops whatever it holds).",
          {"arm": ARM}, ["arm"]),
    _tool("return_arm_home", "Move an arm (or both if omitted) back to its home pose.", {"arm": ARM}),
    _tool("reset_scene", "Reset arms and all objects to their starting positions."),
    _tool("finish", "Call when the task is complete or impossible.",
          {"success": {"type": "boolean"}, "summary": {"type": "string"}}, ["success", "summary"]),
]

SYSTEM = (
    "You control a two-arm ALOHA robot at a table with a circular bin. "
    "Achieve the user's task by calling tools, one at a time.\n"
    "Rules:\n"
    "- Call get_scene first. After each action, read the returned state and verify it worked.\n"
    "- If an action failed (object not in_bin, planner_log shows an error), try a different "
    "arm or a different skill (e.g. grab instead of push). Do not repeat the same failing call more than once.\n"
    "- After grab_object you must release_gripper, then return_arm_home.\n"
    "- Only claim success if get_scene or the last result confirms it (e.g. in_bin is true).\n"
    "- Call finish as soon as the task is satisfied, or when it cannot be done."
)


# Executing a tool call

def run_tool(bridge, name, args):
    arm = args.get("arm") if args.get("arm") in pa.SIDES else None

    if name == "get_scene":
        return bridge.call("scene")

    if name in ("push_to_bin", "grab_object"):
        objs = bridge.call("scene")["objects"]
        obj = args.get("object")
        if obj not in objs:
            return {"error": f"unknown object {obj!r}. Known: {list(objs)}"}
        if not objs[obj]["on_table"]:
            return {"error": f"{obj} is not on the table"}
        if objs[obj]["in_bin"]:
            return {"error": f"{obj} is already in the bin"}
        maker = pa.make_push if name == "push_to_bin" else pa.make_grab
        return bridge.call("skill", maker(obj, arm, 0.0))

    if name == "release_gripper":
        return bridge.call("skill", {"skill": "drop", "arm": arm})
    if name == "return_arm_home":
        return bridge.call("skill", {"skill": "return", "arm": arm})
    if name == "reset_scene":
        return bridge.call("skill", {"skill": "reset"})
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
                    success = bool(args.get("success"))
                    summary = args.get("summary", "")
                    done = True
                    break
                key = (name, json.dumps(args, sort_keys=True))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 2:
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
        bridge.agent_running = False
        print("> ", end="", flush=True)