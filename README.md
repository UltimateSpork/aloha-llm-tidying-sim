# ALOHA LLM Tabletop Tidying Simulation

A bimanual robotics simulation built in **MuJoCo** and **Python** using the ALOHA robot model, paired with a local **Ollama (`qwen2.5:3b`)** task planner that converts natural language commands and live scene coordinates into structured manipulation skills.

## How It Works
- **LLM Task Planning (`ollama_planner.py`):** Compresses live tabletop object positions into a compact JSON scene summary and constrains `qwen2.5:3b` via JSON Schema to output validated `push`, `sweep`, and `reset` skill dictionaries.
- **Motion Execution & Fallback (`planner_actuators.py`):** Translates normalized skills into smooth bimanual pushing trajectories in MuJoCo, backed by a deterministic rule-based parser fallback and an automatic post-action check that replans if an object misses the bin.

## Current Scope & Limitations
- **Non-prehensile manipulation:** Grippers operate as closed pushers rather than grasping and lifting objects.
- **Synchronous inference:** Prompt submission waits on local Ollama generation before queuing motion, introducing a brief pause on submit.
- **3B model constraints:** Small local model inference can occasionally misinterpret complex multi-step or spatial prompts, and motion planning does not yet include full multi-object collision avoidance.
