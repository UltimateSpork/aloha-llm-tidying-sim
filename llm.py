import json
import os
import ollama

import planner_actuators as pa

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


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
        "- Keep 'note' brief or empty."
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