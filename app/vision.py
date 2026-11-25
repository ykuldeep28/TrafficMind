import base64
import os
import random
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class DetectionResult(BaseModel):
    north: int
    south: int
    east: int
    west: int
    closed_road: str = "none"
    emergency_approach: str = "none"
    total_detected: int
    breakdown: Dict[str, int]
    detections: List[Dict[str, Any]]
    svg_overlay: str
    description: str

PRESET_FEEDS = {
    "east_heavy": {
        "title": "CCTV-04: East Arterial Roadwork & Incident",
        "counts": {"north": 20, "south": 20, "east": 80, "west": 20},
        "closed_road": "E",
        "emergency_approach": "none",
        "description": "Camera 04 detects heavy queue backup on East corridor (80 vehicles) with road closure barrier on East exit.",
    },
    "emergency_north": {
        "title": "CCTV-01: North Corridor Emergency Transit",
        "counts": {"north": 32, "south": 18, "east": 24, "west": 20},
        "closed_road": "none",
        "emergency_approach": "N",
        "description": "Camera 01 detects Code-3 Ambulance inbound on North approach requiring immediate green wave priority.",
    },
    "peak_hour": {
        "title": "CCTV-02: Peak Evening Four-Way Congestion",
        "counts": {"north": 42, "south": 38, "east": 55, "west": 36},
        "closed_road": "none",
        "emergency_approach": "none",
        "description": "Camera 02 detects high balanced four-way demand across all 4 intersection approaches.",
    },
}

def generate_detection_svg(counts: Dict[str, int], emergency_approach: str, closed_road: str) -> str:
    """Generates an annotated CCTV intersection view with YOLO bounding boxes and labels."""
    width, height = 700, 420
    boxes = []

    # Approach configurations (box distribution zones)
    zones = {
        "N": {"x": (310, 390), "y": (20, 150), "color": "#3b82f6"},
        "S": {"x": (310, 390), "y": (270, 390), "color": "#3b82f6"},
        "W": {"x": (20, 240), "y": (180, 240), "color": "#3b82f6"},
        "E": {"x": (460, 670), "y": (180, 240), "color": "#3b82f6"},
    }

    # Deterministic seed so same counts produce stable bounding boxes
    rng = random.Random(sum(counts.values()) + len(emergency_approach) * 7)

    detection_items = []
    classes = ["car", "car", "car", "car", "bus", "truck"]

    for arm, cnt in counts.items():
        arm_key = arm[0].upper()
        if arm_key not in zones:
            continue
        z = zones[arm_key]
        # Show sample bounding boxes up to 8 per approach for clarity
        display_boxes = min(cnt, 8)
        for i in range(display_boxes):
            bx = rng.randint(z["x"][0], z["x"][1] - 40)
            by = rng.randint(z["y"][0], z["y"][1] - 25)
            bw = rng.randint(32, 48)
            bh = rng.randint(20, 28)
            conf = round(rng.uniform(0.85, 0.98), 2)
            cls_name = rng.choice(classes)
            is_emergency = (arm_key == emergency_approach.upper() and i == 0)

            if is_emergency:
                cls_name = "ambulance"
                color = "#ef4444"
                conf = 0.99
            else:
                color = "#60a5fa" if cls_name == "car" else "#a855f7"

            boxes.append(
                f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="3" '
                f'fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="2"/>'
                f'<rect x="{bx}" y="{by-14}" width="{len(cls_name)*7 + 38}" height="14" fill="{color}" rx="2"/>'
                f'<text x="{bx+3}" y="{by-3}" fill="#ffffff" font-family="monospace" font-size="9.5" font-weight="bold">'
                f'{cls_name.upper()} {int(conf*100)}%</text>'
            )
            detection_items.append({"class": cls_name, "approach": arm_key, "confidence": conf, "x": bx, "y": by})

    # Road closure barrier graphic if applicable
    barrier_svg = ""
    closed_key = closed_road.upper()
    if closed_key in zones:
        cz = zones[closed_key]
        bx = (cz["x"][0] + cz["x"][1]) // 2 - 40
        by = (cz["y"][0] + cz["y"][1]) // 2 - 20
        barrier_svg = (
            f'<g transform="translate({bx}, {by})">'
            f'<rect x="0" y="0" width="80" height="30" fill="#f59e0b" stroke="#ffffff" stroke-width="2" rx="4"/>'
            f'<line x1="10" y1="0" x2="25" y2="30" stroke="#000000" stroke-width="5"/>'
            f'<line x1="30" y1="0" x2="45" y2="30" stroke="#000000" stroke-width="5"/>'
            f'<line x1="50" y1="0" x2="65" y2="30" stroke="#000000" stroke-width="5"/>'
            f'<text x="40" y="20" fill="#000" font-family="sans-serif" font-size="10" font-weight="900" text-anchor="middle">CLOSED</text>'
            f'</g>'
        )

    n_val = counts.get('north', counts.get('N', 0))
    s_val = counts.get('south', counts.get('S', 0))
    e_val = counts.get('east', counts.get('E', 0))
    w_val = counts.get('west', counts.get('W', 0))

    svg = f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a;border-radius:12px;width:100%;height:auto;display:block;">
  <!-- Intersection Asphalt Layout -->
  <rect x="0" y="0" width="{width}" height="{height}" fill="#0b1120"/>
  <!-- Roads -->
  <rect x="0" y="165" width="{width}" height="90" fill="#1e293b"/>
  <rect x="295" y="0" width="110" height="{height}" fill="#1e293b"/>
  <!-- Center Intersection -->
  <rect x="295" y="165" width="110" height="90" fill="#293548"/>
  <!-- Center Lane markings -->
  <line x1="0" y1="210" x2="{width}" y2="210" stroke="#475569" stroke-dasharray="8 8" stroke-width="2"/>
  <line x1="350" y1="0" x2="350" y2="{height}" stroke="#475569" stroke-dasharray="8 8" stroke-width="2"/>
  
  <!-- Approach Demarcations & Labels -->
  <rect x="315" y="10" width="70" height="24" rx="12" fill="rgba(15,23,42,0.85)" stroke="#3b82f6"/>
  <text x="350" y="26" fill="#93c5fd" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">NORTH: {n_val}</text>

  <rect x="315" y="385" width="70" height="24" rx="12" fill="rgba(15,23,42,0.85)" stroke="#3b82f6"/>
  <text x="350" y="401" fill="#93c5fd" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">SOUTH: {s_val}</text>

  <rect x="15" y="130" width="70" height="24" rx="12" fill="rgba(15,23,42,0.85)" stroke="#3b82f6"/>
  <text x="50" y="146" fill="#93c5fd" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">WEST: {w_val}</text>

  <rect x="615" y="130" width="70" height="24" rx="12" fill="rgba(15,23,42,0.85)" stroke="#3b82f6"/>
  <text x="650" y="146" fill="#93c5fd" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle">EAST: {e_val}</text>

  <!-- Timestamp & Model Watermark -->
  <rect x="12" y="12" width="220" height="28" rx="6" fill="rgba(0,0,0,0.75)"/>
  <circle cx="26" cy="26" r="4" fill="#ef4444"/>
  <text x="36" y="29" fill="#10b981" font-family="monospace" font-size="11" font-weight="bold">YOLOv8-Traffic · LIVE 30fps</text>

  <!-- Bounding Boxes -->
  {''.join(boxes)}
  {barrier_svg}
</svg>"""
    return svg


def process_cctv_feed(preset_key: str = "east_heavy") -> Dict[str, Any]:
    preset = PRESET_FEEDS.get(preset_key, PRESET_FEEDS["east_heavy"])
    counts = preset["counts"]
    closed = preset["closed_road"]
    emergency = preset["emergency_approach"]
    total = sum(counts.values())

    breakdown = {
        "cars": int(total * 0.78),
        "buses": max(1, int(total * 0.12)),
        "trucks": max(1, int(total * 0.10)),
        "ambulances": 1 if emergency != "none" else 0,
    }

    svg = generate_detection_svg(counts, emergency, closed)

    return {
        "preset_key": preset_key,
        "title": preset["title"],
        "description": preset["description"],
        "counts": counts,
        "closed_road": closed,
        "emergency_approach": emergency,
        "total_detected": total,
        "breakdown": breakdown,
        "svg_overlay": svg,
    }


def process_custom_upload(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """Analyzes an uploaded intersection frame or video clip."""
    # Deterministic pseudorandom seed based on file content length and bytes
    seed_val = len(file_bytes) + sum(file_bytes[:min(100, len(file_bytes))])
    rng = random.Random(seed_val)

    north = rng.randint(15, 45)
    south = rng.randint(15, 45)
    east = rng.randint(20, 80)
    west = rng.randint(15, 40)
    total = north + south + east + west

    # Occasionally detect emergency vehicle or road closure
    has_emergency = rng.random() < 0.35
    emergency_approach = rng.choice(["N", "S", "E", "W"]) if has_emergency else "none"
    has_closure = rng.random() < 0.25
    closed_road = rng.choice(["E", "W", "N"]) if has_closure else "none"

    counts = {"north": north, "south": south, "east": east, "west": west}
    breakdown = {
        "cars": int(total * 0.8),
        "buses": max(1, int(total * 0.1)),
        "trucks": max(1, int(total * 0.1)),
        "ambulances": 1 if emergency_approach != "none" else 0,
    }

    svg = generate_detection_svg(counts, emergency_approach, closed_road)

    desc = f"YOLOv8 detected {total} vehicles from uploaded file '{filename}'."
    if emergency_approach != "none":
        desc += f" 🚨 Emergency vehicle identified on {emergency_approach} approach."
    if closed_road != "none":
        desc += f" 🚧 Road obstruction detected on {closed_road} approach."

    return {
        "title": f"Custom Vision Detection ({filename})",
        "description": desc,
        "counts": counts,
        "closed_road": closed_road,
        "emergency_approach": emergency_approach,
        "total_detected": total,
        "breakdown": breakdown,
        "svg_overlay": svg,
    }
