import json
import os
import ollama

import planner_actuators as pa

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

from pathlib import Path

MEMORY_PATH = Path(__file__).resolve().parent / "memory_log.jsonl"


def load_recent_episodes(limit=5, prompt_text=""):
    """Load recent episodes from memory_log.jsonl, prioritizing shared keywords."""
    if not MEMORY_PATH.exists():
        return []

    episodes = []
    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    if not episodes:
        return []

    tokens = {w.lower() for w in prompt_text.replace("_", " ").split() if len(w) > 2}
    if not tokens:
        return episodes[-limit:]

    def score_entry(idx_item):
        idx, ep = idx_item
        haystack = f"{ep.get('prompt', '')} {ep.get('outcome', '')} {json.dumps(ep.get('skills', []))}".lower()
        overlap = sum(1 for tok in tokens if tok in haystack)
        return (overlap, idx)

    ranked = sorted(enumerate(episodes), key=score_entry, reverse=True)
    chosen = [ep for _, ep in ranked[:limit]]
    return chosen


def format_episodes_for_prompt(episodes):
    """Format past episodes into a compact block for qwen2.5:3b."""
    if not episodes:
        return ""
    lines = ["Past episodes (learn from outcomes; avoid repeating failed approaches):"]
    for ep in episodes:
        skill_names = [s.get("skill") for s in ep.get("skills", []) if isinstance(s, dict)]
        status = "OK" if ep.get("success") else "CHECK"
        lines.append(
            f"- [{status}] prompt='{ep.get('prompt', '')}' -> skills={skill_names} | outcome={ep.get('outcome', '')}"
        )
    return "\n".join(lines)


def record_episode(prompt_text, skills, outcome, success=True):
    """Append a completed skill execution episode to memory_log.jsonl."""
    entry = {
        "prompt": prompt_text,
        "skills": skills,
        "outcome": outcome,
        "success": bool(success),
    }
    try:
        with MEMORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass

def build_skill_schema(object_names):
    """JSON schema constraining qwen2.5:3b output to valid skill dictionaries."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["skill", "object", "arm", "side", "follow"],
                    "properties": {
                        "skill": {
                            "type": "string",
                            "enum": ["push", "sweep", "reset", "grab", "drop", "return"],
                        },
                        "object": {
                            "type": ["string", "null"],
                            "enum": list(object_names) + [None],
                        },
                        "arm": {"type": ["string", "null"], "enum": ["left", "right", None]},
                        "side": {"type": ["string", "null"], "enum": ["left", "right", None]},
                        "follow": {"type": "number"},
                    },
                },
            },
            "note": {"type": "string"},
        },
        "required": ["skills", "note"],
    }


def summarize_scene(positions):
    """Build a compact JSON summary of current table and bin state."""
    objects = []
    for name, pos in positions.items():
        objects.append({
            "name": name,
            "xy": [round(float(pos[0]), 3), round(float(pos[1]), 3)],
            "on_table": bool(pa.on_table(pos)),
            "in_bin": bool(pa.in_bin(pos)),
            "dist_to_bin": round(pa.dist_to_bin(pos), 3),
        })
    return {
        "table_center_x": round(float(pa.TABLE_CENTER_XY[0]), 3),
        "bin_center_xy": [round(float(pa.BIN_CENTER_XY[0]), 3), round(float(pa.BIN_CENTER_XY[1]), 3)],
        "bin_radius": float(pa.BIN_RADIUS),
        "objects": objects,
    }


def _extract_arm_or_side(item):
    """Pull a valid 'left' / 'right' arm or side from a skill dict, or None."""
    arm = item.get("arm")
    if arm in ("left", "right"):
        return arm
    side = item.get("side")
    if side in ("left", "right"):
        return side
    return None


def normalize_skills(raw_skills, positions):
    """Validate and convert raw LLM skill dicts into planner_actuators skill dicts."""
    skills = []
    notes = []
    for item in raw_skills:
        if not isinstance(item, dict):
            continue
        kind = item.get("skill")

        if kind == "reset":
            skills.append({"skill": "reset"})

        elif kind == "sweep":
            side = _extract_arm_or_side(item)
            if side in ("left", "right"):
                skills.append({"skill": "sweep", "side": side})
            else:
                notes.append("Sweep skill missing valid side ('left' or 'right').")

        elif kind == "push":
            obj = item.get("object")
            if obj not in positions:
                notes.append(f"Unknown object '{obj}' in LLM output.")
                continue
            if not pa.on_table(positions[obj]):
                notes.append(f"{obj} is not on the table.")
                continue
            if pa.in_bin(positions[obj]):
                notes.append(f"{obj} is already in the bin.")
                continue
            arm = item.get("arm") if item.get("arm") in ("left", "right") else None
            follow = max(0.0, min(0.08, float(item.get("follow", 0.0))))
            skills.append({
                "skill": "push",
                "object": obj,
                "arm": arm,
                "follow": follow,
                "attempt": 0,
            })

        elif kind == "grab":
            obj = item.get("object")
            if obj not in positions:
                notes.append(f"Unknown object '{obj}' in LLM output.")
                continue
            if not pa.on_table(positions[obj]):
                notes.append(f"{obj} is not on the table.")
                continue
            arm = _extract_arm_or_side(item)
            follow = max(0.0, min(0.08, float(item.get("follow", 0.0))))
            skills.append({
                "skill": "grab",
                "object": obj,
                "arm": arm,
                "follow": follow,
                "attempt": 0,
            })

        elif kind == "drop":
            arm = _extract_arm_or_side(item)
            skills.append({"skill": "drop", "arm": arm})

        elif kind == "return":
            arm = _extract_arm_or_side(item)
            skills.append({"skill": "return", "arm": arm})

    return skills, notes


def parse_prompt_ollama(text, positions, model=DEFAULT_MODEL):
    """Query local Ollama via the ollama package; return (skills, notes) or None on failure."""
    
    scene_json = json.dumps(summarize_scene(positions))
    schema = build_skill_schema(positions.keys())
    past_episodes = load_recent_episodes(limit=5, prompt_text=text)
    memory_block = format_episodes_for_prompt(past_episodes)

    user_content = f"Scene: {scene_json}\nPrompt: {text}"
    if memory_block:
        user_content = f"{memory_block}\n\n{user_content}"

    system_msg = (
            "Map the user prompt and scene state to skill dicts.\n"
            "Rules:\n"
            "- For pushing/sliding an object into the bin: skill='push', object=<name>, arm=null (or 'left'/'right' if forced), side=null, follow=0.0.\n"
            "- For grabbing/picking up an object: skill='grab', object=<name>, arm=null (or 'left'/'right' if forced), side=null, follow=0.0.\n"
            "- For opening/closing the claw / dropping / releasing: skill='drop', object=null, arm='left'|'right'|null, side=null, follow=0.0.\n"
            "- For returning arm(s) to home pose: skill='return', object=null, arm='left'|'right'|null, side=null, follow=0.0.\n"
            "- For sweeping a side: skill='sweep', side='left'|'right', object=null, arm=null, follow=0.0.\n"
            "- For full scene reset: skill='reset', object=null, arm=null, side=null, follow=0.0.\n"
            "- Skip objects where in_bin is true unless explicitly asked.\n"
            "- For multiple objects ('tidy everything' / 'clear table'), order nearest dist_to_bin first.\n"
            "- Keep 'note' brief or empty.\n\n"
            "Examples:\n"
            "Prompt: 'grab the red block'\n"
            "Output: {\"skills\":[{\"skill\":\"grab\",\"object\":\"red_block\",\"arm\":null,\"side\":null,\"follow\":0.0}],\"note\":\"\"}\n"
            "Prompt: 'drop it and return right arm'\n"
            "Output: {\"skills\":[{\"skill\":\"drop\",\"object\":null,\"arm\":null,\"side\":null,\"follow\":0.0},{\"skill\":\"return\",\"object\":null,\"arm\":\"right\",\"side\":null,\"follow\":0.0}],\"note\":\"\"}\n"
            "Prompt: 'pick up blue ball with left arm'\n"
            "Output: {\"skills\":[{\"skill\":\"grab\",\"object\":\"blue_ball\",\"arm\":\"left\",\"side\":null,\"follow\":0.0}],\"note\":\"\"}\n"
            "Prompt: 'reset everything'\n"
            "Output: {\"skills\":[{\"skill\":\"reset\",\"object\":null,\"arm\":null,\"side\":null,\"follow\":0.0}],\"note\":\"\"}"
        )

    try:
        response = ollama.chat(
            model=model,
            stream=False,
            format=schema,
            options={"temperature": 0},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Scene: {scene_json}\nPrompt: {text}"},
            ],
        )
    except Exception:
        return None

    content = ""
    if isinstance(response, dict):
        content = (response.get("message") or {}).get("content") or ""
    elif hasattr(response, "message"):
        content = getattr(response.message, "content", "") or ""
    if not content:
        return None

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    skills, notes = normalize_skills(parsed.get("skills", []), positions)
    llm_note = (parsed.get("note") or "").strip()
    if llm_note:
        notes.insert(0, llm_note)
    return skills, notes