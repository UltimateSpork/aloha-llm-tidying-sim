import colorsys
import json
import os
import struct
import threading
import zlib

import mujoco
import numpy as np

import planner_actuators as pa

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

BACKEND = os.environ.get("VISION_BACKEND", "color").lower()
VLM_MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:7b")
REFINE = os.environ.get("VISION_REFINE", "1") != "0"

CAMERA_NAME = "vision_cam"
CAM_HEIGHT = 0.9
CAM_FOVY = 60.0
IMG_W, IMG_H = 448, 336

HUE_TOL = 15.0
MIN_SAT = 0.5
MIN_VAL = 0.15
MIN_PIXELS = 25
NOT_VISIBLE = np.array([9.0, 9.0, -1.0])

DESCRIPTIONS = {
    "red_block": "a small red cube",
    "blue_ball": "a blue ball",
    "green_box": "a green rectangular box",
}


# Camera & rendering helpers

def add_camera(world):
    """Add a downward-looking overhead camera centered over the table."""
    world.add_camera(
        name=CAMERA_NAME,
        pos=[float(pa.TABLE_CENTER_XY[0]), float(pa.TABLE_CENTER_XY[1]), CAM_HEIGHT],
        fovy=CAM_FOVY,
    )


def make_camera_info(model):
    """Return a plain dict of camera intrinsics and extrinsics from the compiled model."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"camera '{CAMERA_NAME}' not found in the model")

    scratch = mujoco.MjData(model)
    mujoco.mj_forward(model, scratch)

    fovy = float(model.cam_fovy[cam_id])
    focal = (IMG_H / 2.0) / np.tan(np.radians(fovy) / 2.0)
    return {
        "cam_id": cam_id,
        "pos": scratch.cam_xpos[cam_id].copy(),
        "rot": scratch.cam_xmat[cam_id].reshape(3, 3).copy(),
        "f": focal,
        "u0": (IMG_W - 1) / 2.0,
        "v0": (IMG_H - 1) / 2.0,
    }


def pixel_to_world(cam_info, u, v, z):
    """Project pixel (u, v) onto horizontal plane z in world coordinates."""
    ray = cam_info["rot"] @ np.array([
        (u - cam_info["u0"]) / cam_info["f"],
        -(v - cam_info["v0"]) / cam_info["f"],
        -1.0,
    ])
    t = (z - cam_info["pos"][2]) / ray[2]
    return cam_info["pos"] + t * ray


def world_to_pixel(cam_info, point):
    """Project a 3D world point into pixel coordinates (u, v)."""
    local = cam_info["rot"].T @ (np.asarray(point, dtype=float) - cam_info["pos"])
    u = cam_info["u0"] + cam_info["f"] * local[0] / -local[2]
    v = cam_info["v0"] - cam_info["f"] * local[1] / -local[2]
    return float(u), float(v)


def make_renderer(model):
    """Create a MuJoCo offscreen renderer (call capture_frame on the main thread)."""
    return mujoco.Renderer(model, height=IMG_H, width=IMG_W)


def capture_frame(renderer, data):
    """Render an RGB image from the overhead camera."""
    renderer.update_scene(data, camera=CAMERA_NAME)
    return renderer.render().copy()


# Image & color helpers

def png_bytes(rgb):
    """Encode an HxWx3 uint8 array as PNG bytes."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, body):
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def save_png(path, rgb):
    with open(path, "wb") as f:
        f.write(png_bytes(rgb))


def _rgb_to_hsv(rgb):
    """Vectorized RGB uint8 -> hue (0..360), saturation (0..1), value (0..1)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(axis=2)
    diff = mx - x.min(axis=2)

    h = np.zeros_like(mx)
    nz = diff > 1e-6
    is_r = (mx == r) & nz
    is_g = (mx == g) & nz & ~is_r
    is_b = nz & ~is_r & ~is_g

    h[is_r] = ((g - b)[is_r] / diff[is_r]) % 6.0
    h[is_g] = (b - r)[is_g] / diff[is_g] + 2.0
    h[is_b] = (r - g)[is_b] / diff[is_b] + 4.0

    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return h * 60.0, s, mx


def plane_z(name):
    """Top/center height of an object used for pixel-to-world ray intersection."""
    cfg = pa.OBJECT_SPECS[name]
    extra = cfg["half"][2] if cfg["shape"] == "box" else 0.0
    return pa.object_rest_z(name) + extra


def table_mask(cam_info):
    """Boolean mask covering the table surface plus a small margin."""
    cx, cy = float(pa.TABLE_CENTER_XY[0]), float(pa.TABLE_CENTER_XY[1])
    hx, hy = pa.TABLE_HALF_XY
    margin = 0.02
    us, vs = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            u, v = world_to_pixel(cam_info, [cx + sx * (hx + margin), cy + sy * (hy + margin), pa.TABLE_TOP_Z])
            us.append(u)
            vs.append(v)

    u_lo = max(0, int(np.floor(min(us))))
    u_hi = min(IMG_W, int(np.ceil(max(us))) + 1)
    v_lo = max(0, int(np.floor(min(vs))))
    v_hi = min(IMG_H, int(np.ceil(max(vs))) + 1)

    mask = np.zeros((IMG_H, IMG_W), dtype=bool)
    mask[v_lo:v_hi, u_lo:u_hi] = True
    return mask


def _object_hue(name):
    r, g, b = pa.OBJECT_SPECS[name]["rgba"][:3]
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360.0


def _largest_blob_center(mask):
    """Return (u, v) of the largest connected blob above MIN_PIXELS, or None."""
    if ndimage is not None:
        labels, count = ndimage.label(mask)
        if count == 0:
            return None
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)

    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PIXELS:
        return None
    return float(xs.mean()), float(ys.mean())


def detect_by_color(image, names, cam_info, boxes=None):
    """Return {name: (u, v) or None} using HSV color segmentation."""
    h, s, v = _rgb_to_hsv(image)
    region = table_mask(cam_info)
    pixels = {}

    for name in names:
        dh = np.abs((h - _object_hue(name) + 180.0) % 360.0 - 180.0)
        mask = (dh < HUE_TOL) & (s > MIN_SAT) & (v > MIN_VAL) & region
        if boxes is not None:
            if name not in boxes:
                pixels[name] = None
                continue
            x1, y1, x2, y2 = boxes[name]
            pad = 10
            box_mask = np.zeros_like(mask)
            box_mask[max(0, int(y1) - pad):int(y2) + pad + 1, max(0, int(x1) - pad):int(x2) + pad + 1] = True
            mask = mask & box_mask
        pixels[name] = _largest_blob_center(mask)

    return pixels


# VLM detection

def detect_boxes_vlm(image, names, vlm_model=VLM_MODEL):
    """Ask an Ollama VLM for [x1, y1, x2, y2] bounding boxes per object."""
    import ollama

    schema = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(names)},
                        "bbox": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["name", "bbox"],
                },
            }
        },
        "required": ["objects"],
    }

    listing = "\n".join(f"- {n}: {DESCRIPTIONS.get(n, n.replace('_', ' '))}" for n in names)
    prompt = (
        f"This is a top-down camera image ({IMG_W}x{IMG_H} pixels) of a table. "
        "The flat green disc is a bin; it is NOT an object. Find each of these objects:\n"
        f"{listing}\n"
        "For each one give its bounding box [x1, y1, x2, y2] in pixel coordinates "
        "(x right, y down), or null if not visible. Answer with JSON only."
    )

    response = ollama.chat(
        model=vlm_model,
        messages=[{"role": "user", "content": prompt, "images": [png_bytes(image)]}],
        format=schema,
        options={"temperature": 0},
    )
    content = response["message"]["content"] if isinstance(response, dict) else response.message.content
    items = json.loads(content).get("objects", [])

    boxes = {}
    for item in items:
        name = item.get("name")
        box = item.get("bbox")
        if name in names and isinstance(box, list) and len(box) == 4:
            boxes[name] = [float(c) for c in box]

    biggest = max([max(b) for b in boxes.values()] + [0.0])
    if biggest > max(IMG_W, IMG_H) * 1.05:
        for name, (x1, y1, x2, y2) in list(boxes.items()):
            boxes[name] = [
                x1 * IMG_W / 1000.0,
                y1 * IMG_H / 1000.0,
                x2 * IMG_W / 1000.0,
                y2 * IMG_H / 1000.0,
            ]

    clean = {}
    for name, (x1, y1, x2, y2) in boxes.items():
        x_lo, x_hi = sorted((min(max(x1, 0.0), IMG_W - 1.0), min(max(x2, 0.0), IMG_W - 1.0)))
        y_lo, y_hi = sorted((min(max(y1, 0.0), IMG_H - 1.0), min(max(y2, 0.0), IMG_H - 1.0)))
        clean[name] = [x_lo, y_lo, x_hi, y_hi]
    return clean


def _build_percept(pixels, cam_info, names, source):
    positions, visible = {}, {}
    for name in names:
        px = pixels.get(name)
        if px is None:
            positions[name] = NOT_VISIBLE.copy()
            visible[name] = False
        else:
            world_pt = pixel_to_world(cam_info, px[0], px[1], plane_z(name))
            positions[name] = np.array([world_pt[0], world_pt[1], pa.object_rest_z(name)])
            visible[name] = True
    return {
        "positions": positions,
        "visible": visible,
        "pixels": pixels,
        "source": source,
    }


def locate_objects(image, names, cam_info, backend=BACKEND, vlm_model=VLM_MODEL, refine=REFINE):
    """Detect objects in `image` using `backend` ('color', 'vlm', or 'off' fallback)."""
    source = "color"
    if backend == "vlm":
        try:
            boxes = detect_boxes_vlm(image, names, vlm_model=vlm_model)
            refined = detect_by_color(image, names, cam_info, boxes=boxes) if refine else {}
            pixels = {}
            for name in names:
                if name not in boxes:
                    pixels[name] = None
                    continue
                x1, y1, x2, y2 = boxes[name]
                pixels[name] = refined.get(name) or ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            tag = f"vlm:{vlm_model}" + ("+refine" if refine else "")
            return _build_percept(pixels, cam_info, names, tag)
        except Exception as exc:
            print(f"[Vision] VLM failed ({type(exc).__name__}: {str(exc)[:80]}); falling back to color")
            source = "color (vlm failed)"

    return _build_percept(detect_by_color(image, names, cam_info), cam_info, names, source)


# Sim integration helpers

def believed_positions(sim):
    """Return believed object positions from vision or ground truth when vision is off."""
    if not sim.get("vision_on"):
        return pa.read_positions(sim["data"], pa.ACTIVE_OBJECTS)
    belief = sim.get("belief") or {}
    return {name: pos.copy() for name, pos in belief.items()}


def start_perception(sim, then_fn=None, backend=BACKEND):
    """Capture a frame on the main thread and run detection in a background thread."""
    image = capture_frame(sim["renderer"], sim["data"])
    cam_info = sim["cam_info"]
    done_event = threading.Event()
    holder = {"result": None}

    def run_worker():
        try:
            holder["result"] = locate_objects(image, pa.ACTIVE_OBJECTS, cam_info, backend=backend)
        except Exception as exc:
            holder["result"] = {"error": str(exc)}
        finally:
            done_event.set()

    threading.Thread(target=run_worker, daemon=True).start()
    sim["last_image"] = image
    sim["perception_done"] = done_event
    sim["perception_result"] = holder
    sim["after_perception"] = then_fn
    sim["mode"] = "perceiving"


def finish_perception(sim):
    """Consume completed background perception result and invoke callback."""
    holder = sim.get("perception_result") or {}
    result = holder.get("result") or {"error": "missing result"}

    if "error" in result:
        print("[Vision] error: " + str(result["error"]))
    else:
        sim["belief"] = result["positions"]
        sim["visible"] = result["visible"]
        parts = []
        for name, pos in result["positions"].items():
            if result["visible"][name]:
                parts.append(f"{name} ({pos[0]:.3f}, {pos[1]:.3f})")
            else:
                parts.append(f"{name} not visible")
        print("[Vision] " + result["source"] + ": " + " | ".join(parts))

    sim["belief_stale"] = False
    sim["perception_done"] = None
    sim["perception_result"] = None
    then_fn = sim.get("after_perception")
    sim["after_perception"] = None
    sim["mode"] = "idle"
    if callable(then_fn):
        then_fn(sim)


# Debug drawing

def annotate_frame(image, percept, cam_info, truth=None):
    """Draw green crosses on detected pixels and white crosses on ground-truth projections."""
    out = image.copy()

    def draw_cross(u, v, color):
        cx, cy = int(round(u)), int(round(v))
        for d in range(-6, 7):
            for x, y in ((cx + d, cy), (cx, cy + d)):
                if 0 <= x < IMG_W and 0 <= y < IMG_H:
                    out[y, x] = color

    for px in percept.get("pixels", {}).values():
        if px is not None:
            draw_cross(px[0], px[1], (0, 255, 0))

    if truth is not None:
        for name, pos in truth.items():
            u, v = world_to_pixel(cam_info, [pos[0], pos[1], plane_z(name)])
            draw_cross(u, v, (255, 255, 255))

    return out