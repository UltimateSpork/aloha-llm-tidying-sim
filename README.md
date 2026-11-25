# ALOHA Pick-and-Place & Tidy Simulation (MuJoCo + Ollama)

Bimanual manipulation simulation for the **ALOHA** robot in **MuJoCo** (`mujoco_menagerie`), supporting natural-language and rule-based command parsing (`push`, `sweep`, `grab`, `drop`, `return`, and `reset`) via a local **Ollama** model (`qwen2.5:3b`) with a deterministic rule fallback.

## Features

- **Bimanual Skill Planning (`planner_actuators.py`):**
  - `push` & `sweep`: Damped least-squares inverse kinematics (IK) with level-gripper orientation weighting, windup gap retries, and bin-slot spacing.
  - `grab`: Multi-stage prehensile sequence (pre-grasp hover, descend, close gripper, lift, carry to bin slot, release).
  - `drop`: In-place gripper open/hold/close/settle routine for a selected arm without altering arm joint targets.
  - `return`: Smoothly returns one arm (`left`/`right`) or both arms back to `home` joint and gripper controls.
  - `reset`: Returns arms to neutral home pose and resets object poses/velocities on the table.
- **LLM + Rule-Based Prompt Parsing (`llm.py` & `planner_actuators.py`):**
  - Uses local Ollama (`qwen2.5:3b` by default) with a strict JSON schema and scene state summary (`on_table`, `in_bin`, `dist_to_bin`, coordinates), falling back to `parse_prompt` when needed.
- **Interactive & Headless Execution (`main.py`):**
  - Passive MuJoCo viewer with a background terminal command loop, plus a `--headless` mode for automated or CLI test runs.

## Project Structure

- `main.py` — Scene construction (`build_scene`), simulation state (`make_sim`), keyframe interpolation (`step_motion`), auto-retry checking, and viewer/headless entry points.
- `planner_actuators.py` — Scene constants (`OBJECT_SPECS`, table/bin geometry), actuator mapping, damped least-squares IK (`solve_ik`), skill planners (`plan_push_for_arm`, `plan_grab_for_arm`, `plan_drop_for_arm`, `plan_return_for_arm`), `plan_skill`, and fallback `parse_prompt`.
- `llm.py` — Skill JSON schema (`build_skill_schema`), compact scene serializer (`summarize_scene`), skill normalizer (`normalize_skills`), and Ollama parser (`parse_prompt_ollama`).

## Setup & Usage

### Prerequisites

- Python 3.10+ with `mujoco`, `mujoco-menagerie`, `numpy`, and `ollama` installed
- [Ollama](https://ollama.com/) running locally with `qwen2.5:3b` pulled (optional; rule-based fallback still parses commands if Ollama is unavailable):

```bash
ollama pull qwen2.5:3b