# ALOHA Pick-and-Place & Tidy Simulation (MuJoCo + Ollama)

Bimanual manipulation simulation for the **ALOHA** robot in **MuJoCo** (`mujoco_menagerie`). There are two ways to drive it:

1. **Commands**: short typed commands (`push red block`, `grab blue ball`, `drop`, ...) parsed by a local **Ollama** model, with a deterministic rule-based fallback.
2. **Autonomous agent**: give the robot a task (`task put both objects in the bin`) and an Ollama tool-calling agent decides which skills to run, checks the simulation state after each one, and adapts if something fails.

## Features

- **Bimanual skill planning (`planner_actuators.py`):**
  - `push` / `sweep`: damped least-squares inverse kinematics (IK) with level-gripper weighting, windup-gap retries, and bin-slot spacing so later pushes don't shove earlier objects out.
  - `grab`: pre-grasp hover, descend, close gripper, lift, carry above the bin. The gripper **stays closed**, so a separate `drop` is needed to release the object.
  - `drop`: opens the gripper in place and holds it open, without changing arm joint targets. If no arm is given, the arm furthest from its home pose (the one carrying) is used.
  - `return`: smoothly returns one arm (`left`/`right`) or both to the home pose.
  - `reset`: returns arms to the home pose and puts all objects back at their start positions.
- **Autonomous agent (`agent.py`):**
  - Tool-calling loop (`get_scene`, `push_to_bin`, `grab_object`, `release_gripper`, `return_arm_home`, `reset_scene`, `finish`).
  - After every action the agent receives the planner messages and a fresh observation: object positions, `in_bin`, `lifted`, `held_by`, and each arm's gripper state and `holding` object.
  - Objects that are carried above the bin are **not** counted as in the bin, and `finish` is refused while an arm is still holding something.
  - Repeated identical failing calls are cut off, and the loop stops after `MAX_STEPS` (15).
  - Every run is logged to `memory_log.jsonl`; relevant past episodes are injected into later prompts.
- **LLM + rule-based command parsing (`llm.py`, `planner_actuators.py`):**
  - Ollama with a strict JSON schema and a scene summary, falling back to `parse_prompt` when Ollama is unavailable or returns nothing.
- **Interactive and headless execution (`main.py`):**
  - Passive MuJoCo viewer with a background terminal input thread, plus a `--headless` mode for scripted tests.

## Project Structure

| File | Purpose |
|---|---|
| `main.py` | Scene construction (`build_scene`), simulation state (`make_sim`), skill queue (`submit_skills`), keyframe interpolation (`step_motion`), auto-retry check for pushes (`check_and_replan`), viewer and headless entry points, `task` command handling. |
| `planner_actuators.py` | Scene constants (`OBJECT_SPECS`, table and bin geometry), actuator map, IK (`solve_ik`), skill planners (`plan_push_for_arm`, `plan_grab_for_arm`, `plan_drop_for_arm`, `plan_return_for_arm`), `plan_skill`, held/lifted detection (`held_objects`, `is_lifted`), and the rule-based `parse_prompt`. |
| `agent.py` | Agent loop (`run_agent`), tool definitions, observation builder (`observe`), and the `Bridge` that lets the agent thread run skills on the simulation thread. |
| `llm.py` | One-shot command parser (`parse_prompt_ollama`, JSON schema, `normalize_skills`) and episode memory (`record_episode`, `load_recent_episodes`, `format_episodes_for_prompt`). |
| `memory_log.jsonl` | Created automatically at runtime. One JSON line per completed episode. |

## How the Agent Works

The simulation must keep stepping in the main thread, while the LLM calls are slow and blocking, so they run in separate threads:

```
agent thread                         main thread (viewer loop)
------------                         -------------------------
LLM picks a tool call
bridge.call("skill", ...)  ───────►  Bridge.pump() plans and starts the skill
   (blocks)                          physics steps until the skill is done
                           ◄───────  result = planner log + observe(sim)
LLM reads the result, picks next tool
...
finish(success, summary)   ──────►   episode written to memory_log.jsonl
```

Because the observation is taken after the skill finishes and the objects settle, the agent can verify its own work and recover, for example by switching arms, switching from `push` to `grab`, or releasing a held object.

## Setup

### Prerequisites

- Python 3.10+
- `mujoco`, `mujoco-menagerie`, `numpy`, and `ollama` (Python package)
- [Ollama](https://ollama.com/) running locally

```bash
pip install mujoco mujoco-menagerie numpy ollama
ollama pull qwen2.5:7b
```

### Choosing a model

The model is read from the `OLLAMA_MODEL` environment variable (default `qwen2.5:3b`).

```bash
export OLLAMA_MODEL=qwen2.5:7b      # recommended for agent tasks
```

- `qwen2.5:3b` is fine for one-shot commands but unreliable for multi-step agent tasks (it tends to loop or skip verification).
- Use a model with **tool-calling support** for the agent (for example `qwen2.5:7b`, `llama3.1:8b`, `qwen3:8b`). Models of this size need roughly 8 GB of RAM.
- Ollama is optional for typed commands: without it, the rule-based parser handles them. The agent requires Ollama.

## Usage

```bash
python main.py                                   # viewer + interactive prompt
python main.py --headless "push red block"       # no viewer, typed commands only
python main.py --headless "grab red block" "drop" "return"
```

### Typed commands

Typed at the `>` prompt in the terminal while the viewer is open:

| Command | What it does |
|---|---|
| `push red block` | Push the object into the bin |
| `tidy everything` | Push every loose object, nearest to the bin first |
| `sweep left` / `sweep right` | Push the loose objects on that side (empty sweep if there are none) |
| `grab blue ball` / `pick up the red block with the left arm` | Pick up and carry above the bin (gripper stays closed) |
| `drop` / `drop right` | Open the gripper |
| `return` / `return left` | Send one or both arms home |
| `reset` | Reset arms and objects |
| `quit` | Exit |

Commands can be chained: `grab red block, then drop, then return`.

Pushed objects are checked after settling and are automatically re-pushed up to `MAX_AUTO_RETRIES` (2) times.

### Agent tasks

Prefix the request with `task`:

```
> task put both objects in the bin
> task pick up the blue ball and drop it in the bin
> task push the red block into the bin, and if that fails try grabbing it
```

Progress is printed as `[Agent] tool(args)` lines, followed by the planner's `[Plan]` / `[Exec]` messages, and ends with `[Agent] SUCCESS: ...` or `[Agent] DONE: ...`. New input is ignored until the agent has finished. Agent tasks are not supported in `--headless` mode.

## Memory

Each agent run appends `{prompt, skills, outcome, success}` to `memory_log.jsonl`. Before the next run (agent or one-shot parse), the five most relevant past episodes, ranked by shared keywords with the new prompt, are added to the prompt. Delete the file to clear the robot's memory. Entries written before the agent was added list skill names, while newer ones list tool names, so older entries look inconsistent next to new ones.

## Configuration

Tunable constants live at the top of `planner_actuators.py`:

| Constant | Meaning |
|---|---|
| `ACTIVE_OBJECTS` | Objects added to the scene (`green_box` is defined but off by default) |
| `BIN_CENTER_XY`, `BIN_RADIUS` | Bin location, and how close an object's centre must be to count as "in the bin" |
| `PUSH_TIP_Z`, `HOVER_DZ`, `WINDUP_GAP` | Push heights and approach geometry |
| `REACH_TOL`, `MAX_ACCEPT_ERR` | IK error that counts as reached, and the error beyond which a skill is rejected |
| `SETTLE_TIME` | Seconds to wait before checking the result |

`MAX_AUTO_RETRIES` is in `main.py`; `MAX_STEPS` is in `agent.py`.

## Troubleshooting

- **The arm keeps holding the object.** `grab` intentionally leaves the gripper closed. Use `drop` (typed) or let the agent call `release_gripper`. If the object still doesn't fall, check that the gripper's open value (`amap["hi"]`) is wider than the object.
- **`neither arm can reach` messages.** The target is outside the arm's workspace within `MAX_ACCEPT_ERR`. Move the object or the bin, or raise the tolerance.
- **The agent loops or finishes too early.** Use a larger model (see above). Duplicate failing calls are cut off after two attempts, and `finish` is blocked while an arm is holding an object.
- **Ollama errors.** Typed commands fall back to the rule-based parser. The agent prints `Agent error: ...` and stops. Make sure `ollama serve` is running and the model is pulled.