# Completed by Cayden Auyang  on Jul 31, 2025
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_parser.py – macOS terminal script that replicates an OmniTool-style GUI agent.
Goal: be flexible, app-agnostic, and adaptive (no hard-coded task flows).

Key capabilities:
• Window targeting: constrain actions to the front window (Chrome by URL/omnibox, or any app via AppleScript).
• Vision + OCR: extract candidate labels and icons from screenshots.
• Generic, goal-driven navigation (`reach_target`) guided by LLM-derived target tokens (no site-specific hacks).
• Cookie banners: proactively click “Accept all” / “Agree” / “Allow” (vision-first).
• DOM fallback: if a vision click doesn't navigate, click via Selenium/JS.
• Anti-looping: avoid re-clicking when the screen fingerprint doesn’t change.
• Success checks: verify by URL/text that we reached the intended state.
• Tiny experience memory: bandit-style preference for labels that worked on a domain (no training infra).

Models:
- Planning and Reflex (micro-steps) use configurable OpenAI model names via env:
  OPENAI_MODEL_PLANNER (default = OPENAI_MODEL_TEXT), OPENAI_MODEL_REFLEX (default = OPENAI_MODEL_TEXT).
"""

import os
os.environ.setdefault("PYAUTOGUI_HIDE_MOUSEINFO", "1")

import re
import sys
import json
import time
import math
import hashlib
import logging
import subprocess
import difflib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class Step:
    description: str
    action: str
    params: Dict[str, Any]

from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps

import mss
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import subprocess

def clipboard_write_text(text: str):
    """Copy `text` into macOS clipboard via pbcopy."""
    p = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    p.communicate(text.encode())

from openai import OpenAI


#######################################################################
#  Logging
#######################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d – %(message)s",
)
log = logging.getLogger("agent_parser")


#######################################################################
#  Configuration
#######################################################################
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "/Users/caydenauyang/Desktop/OmniParser/weights/icon_detect/model.pt")

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "REPLACE_WITH_YOUR_OPENAI_KEY")
if not OPENAI_KEY or OPENAI_KEY.startswith("REPLACE_") or OPENAI_KEY.strip() == "":
    raise RuntimeError("OpenAI API key missing. Set env var OPENAI_API_KEY or edit OPENAI_KEY in script.")
OPENAI_MODEL_TEXT = os.getenv("OPENAI_MODEL_TEXT", "gpt-4.1")  # more powerful model
OPENAI_MODEL_PLANNER = os.getenv("OPENAI_MODEL_PLANNER", OPENAI_MODEL_TEXT)
OPENAI_MODEL_REFLEX  = os.getenv("OPENAI_MODEL_REFLEX",  OPENAI_MODEL_TEXT)

# --- THINKING / ACCURACY KNOBS ---
LLM_DELIBERATE = int(os.getenv("LLM_DELIBERATE", "1"))     # Reflexor votes; 1=no extra thinking
PLANNER_N      = int(os.getenv("PLANNER_N", "1"))          # # of plan samples to consider
REFLEX_TEMP    = float(os.getenv("REFLEX_TEMP", "0.3"))    # reflex generation temperature
PLANNER_TEMP   = float(os.getenv("PLANNER_TEMP", "0.2"))   # planner temperature (keep low for stability)
VISIBLE_TEXT_LIMIT = int(os.getenv("VISIBLE_TEXT_LIMIT", "160"))  # how much of OCR text we pass to Reflexor
REACH_MAX_ITERS_DEFAULT = int(os.getenv("REACH_MAX_ITERS", "14")) # default micro-steps in reach_target

BOX_CONF_THRESHOLD = 0.05
IOU_NMS_THRESHOLD  = 0.70
MULTI_SCALES: List[Optional[int]] = [960]

OCR_TEXT_DET_BOX_THR   = 0.20
OCR_TEXT_REC_SCORE_THR = 0.40

OUT_DIR = "./agent_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Timing knobs
WAIT_AFTER_CLICK = 0.8
WAIT_AFTER_NAV   = 1.50

# Experience memory file (domain → label → score); tiny, local, no training infra
EXPERIENCE_PATH = os.path.join(OUT_DIR, "experience_memory.json")

mouse = MouseController()
keyboard = KeyboardController()

HAVE_PYAUTOGUI = False
try:
    sys.modules["mouseinfo"] = None  # avoid rubicon path
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    HAVE_PYAUTOGUI = True
    log.info("pyautogui enabled ✔️")
except Exception as e:
    HAVE_PYAUTOGUI = False
    log.warning("⚠️  pyautogui disabled (%s). Using fallback (pynput+mss).", e)

### [ADDED] Reuse/same-window toggles (CLI + env)
REUSE_CHROME_ENV = bool(int(os.getenv("REUSE_CHROME", "0")))
PASTE_LINK_ENV   = bool(int(os.getenv("PASTE_LINK_IN_SAME_WINDOW", "0")))
DEBUG_ADDRESS_ENV = os.getenv("CHROME_DEBUG_ADDRESS", "127.0.0.1:9222")


#######################################################################
#  Utilities
#######################################################################
def device_pick() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def needs_openai_output(goal: str, steps: List[Step]) -> bool:
    output_keywords = [
        "tell me", "report", "print", "summarize", "what is", "write out", "write for me",
        "copy result", "answer", "extract", "output", "paste for me", "send to terminal",
        "state", "show me", "give me", "find and report", "show result"
    ]
    g = norm_text(goal)
    if any(k in g for k in output_keywords):
        return True
    # or if last step is “read_text”, “reach_target”, or a non-UI action
    if steps and steps[-1].action in ("read_text", "reach_target"):
        return True
    return False

def load_font(size: int = 12):
    for p in [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def enhance(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.20)
    return ImageOps.autocontrast(img)


def get_screen_size() -> Tuple[int, int]:
    if HAVE_PYAUTOGUI:
        w, h = pyautogui.size()
        return int(w), int(h)
    with mss.mss() as sct:
        mon = sct.monitors[1]
        return int(mon["width"]), int(mon["height"])


def screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    if HAVE_PYAUTOGUI:
        if region is not None:
            return pyautogui.screenshot(region=region)
        return pyautogui.screenshot()
    with mss.mss() as sct:
        if region is None:
            mon = sct.monitors[1]
            grab = sct.grab(mon)
        else:
            left, top, width, height = region
            grab = sct.grab({"left": int(left), "top": int(top), "width": int(width), "height": int(height)})
        return Image.frombytes("RGB", grab.size, grab.rgb)


def xyxy_to_cxcywh(x1, y1, x2, y2):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = (x2 - x1)
    h  = (y2 - y1)
    return cx, cy, w, h


def iou_xyxy(a: List[float], b: List[float]) -> float:
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, (a[2] - a[0])) * max(0, (a[3] - a[1]))
    areaB = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
    return inter / (areaA + areaB - inter + 1e-6)


def nms_xyxy(boxes: List[List[float]], scores: List[float], thr: float) -> List[int]:
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if iou_xyxy(boxes[i], boxes[j]) > thr:
                suppressed[j] = True
    return keep


def normalize_xyxy(box: List[int], W: int, H: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [x1 / W, y1 / H, x2 / W, y2 / H]


def ensure_openai():
    if not OPENAI_KEY or OPENAI_KEY.startswith("REPLACE_") or OPENAI_KEY.strip() == "":
        raise RuntimeError("OpenAI API key missing. Set env var OPENAI_API_KEY or edit OPENAI_KEY.")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

def tokenize(s: str) -> List[str]:
    s = norm_text(re.sub(r"[^\w\s+]", " ", s))
    return [t for t in s.split() if t]

def fuzzy_score(a: str, b: str) -> float:
    """0..1 similarity based on SequenceMatcher with light tokenization."""
    a2, b2 = " ".join(tokenize(a)), " ".join(tokenize(b))
    if not a2 or not b2: return 0.0
    return difflib.SequenceMatcher(None, a2, b2).ratio()


def matches_hint(content: str, hint_norm: str) -> bool:
    c = norm_text(content)
    h = norm_text(hint_norm)
    if not h:
        return False
    return (h == c) or (h in c) or (c in h)


def synonyms_for_add_to_bag() -> List[str]:
    return [
        "add to bag", "add to cart", "add to basket",
        "add to shopping bag", "add bag", "add cart", "buy now",
        "add to bag ", "add bag ", "bag"
    ]

def synonyms_for_cookie_accept() -> List[str]:
    return [
        "accept all","accept","agree","i agree","allow all","allow",
        "got it","ok","okay","i understand","continue","accept cookies"
    ]


#######################################################################
#  Chat-like tracing (OmniTool style)
#######################################################################
class ChatTrace:
    def __init__(self, run_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.run_dir = run_dir
        self.log_path = os.path.join(run_dir, "trace.log")
        os.makedirs(run_dir, exist_ok=True)

    def _write(self, line: str):
        if self.enabled:
            print(line, flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def banner(self, title: str):
        self._write("\n" + "═" * 80)
        self._write(f"🧭  {title}")
        self._write("═" * 80)

    def plan(self, steps: List["Step"]):
        self._write("🧠 Plan:")
        for i, s in enumerate(steps, 1):
            self._write(f"  {i:02d}. {s.action} – {s.description}  params={s.params}")

    def info(self, msg: str):  self._write(f"ℹ️  {msg}")
    def warn(self, msg: str):  self._write(f"⚠️  {msg}")
    def error(self, msg: str): self._write(f"❌ {msg}")
    def action(self, msg: str): self._write(f"🎯 {msg}")

    # Think‑aloud helpers
    def think(self, msg: str): self._write(f"🧩 Thought: {msg}")
    def observe(self, msg: str): self._write(f"👀 Observation: {msg}")
    def outcome(self, msg: str): self._write(f"✅ Outcome: {msg}")
    def next(self, msg: str): self._write(f"➡️  Next: {msg}")

    def parse_snapshot(self, W: int, H: int, n_text: int, n_icons: int, fp: str, looking_for: Optional[List[str]] = None, sample_texts: Optional[List[str]] = None):
        lf = f" | looking_for={looking_for}" if looking_for else ""
        self._write(f"🖼️  Window snapshot {W}x{H}, texts={n_text}, icons={n_icons}, fp={fp}{lf}")
        if sample_texts:
            preview = "; ".join(sample_texts[:6])
            if preview:
                self._write(f"   · OCR preview: {preview}")


#######################################################################
#  Models – OmniParser‑like (no captions)
#######################################################################
class OmniParser:
    def __init__(self, yolo_weights: str):
        self.dev = device_pick()
        try:
            self.yolo = YOLO(yolo_weights)
            log.info("Loaded YOLO weights: %s", yolo_weights)
        except Exception as e:
            log.warning("Custom weights failed (%s); falling back to yolov8n.pt", e)
            self.yolo = YOLO("yolov8n.pt")

        self.ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            text_det_box_thresh=OCR_TEXT_DET_BOX_THR,
            text_rec_score_thresh=OCR_TEXT_REC_SCORE_THR,
        )

    def _annotate(self, image_np: np.ndarray, xyxy_list: List[List[int]]) -> Image.Image:
        img = Image.fromarray(image_np.copy())
        draw = ImageDraw.Draw(img)
        font = load_font(12)
        for idx, (x1, y1, x2, y2) in enumerate(xyxy_list):
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw.text((x1 + 3, y1 + 3), str(idx + 1), font=font, fill=(0, 255, 0))
        return img

    def _ocr_extract(self, image_np: np.ndarray):
        """Correctly return texts and their boxes (xyxy)."""
        ocr_texts: List[str] = []
        ocr_boxes_xyxy: List[List[int]] = []
        try:
            res = self.ocr.ocr(image_np, cls=True)
            if res:
                for batch in res:
                    for item in batch:
                        pts = item[0]
                        (text, score) = item[1]
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                        if x2 > x1 and y2 > y1 and float(score) >= OCR_TEXT_REC_SCORE_THR:
                            ocr_texts.append(str(text))
                            ocr_boxes_xyxy.append([x1, y1, x2, y2])
        except Exception as e:
            log.warning("OCR failed: %s", e)
        return ocr_texts, ocr_boxes_xyxy

    def parse(self, image: Image.Image) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        img = enhance(image)
        W, H = img.size
        image_np = np.array(img)

        # YOLO detect (icons/generic hitboxes)
        y_boxes: List[List[int]] = []
        y_scores: List[float] = []
        for sz in MULTI_SCALES:
            try:
                kwargs = dict(
                    source=img,
                    conf=BOX_CONF_THRESHOLD,
                    iou=0.40,
                    half=False,
                    verbose=False,
                    device="cpu",
                )
                if isinstance(sz, int):
                    kwargs["imgsz"] = sz
                pred = self.yolo.predict(**kwargs)
            except Exception as e:
                log.error("YOLO inference failed: %s", e)
                continue
            if not pred:
                continue
            b = pred[0].boxes
            if b is None or b.xyxy is None:
                continue
            xyxy = b.xyxy.cpu().numpy().astype(int).tolist()
            scrs = b.conf.cpu().numpy().tolist()
            for bx, sc in zip(xyxy, scrs):
                x1, y1, x2, y2 = bx
                if x2 > x1 and y2 > y1:
                    y_boxes.append([x1, y1, x2, y2])
                    y_scores.append(float(sc))

        # OCR
        ocr_texts, ocr_boxes_xyxy = self._ocr_extract(image_np)

        # NMS icons
        if y_boxes:
            keep = nms_xyxy(y_boxes, y_scores, IOU_NMS_THRESHOLD)
            y_boxes = [y_boxes[i] for i in keep]
            y_scores= [y_scores[i] for i in keep]

        elements: List[Dict[str, Any]] = []
        for (box, txt) in zip(ocr_boxes_xyxy, ocr_texts):
            elements.append(
                {"type": "text", "bbox_xyxy_px": box, "bbox_xyxy": normalize_xyxy(box, W, H),
                 "interactive": False, "content": f"{txt}".strip(), "source": "ocr"}
            )
        for box, sc in zip(y_boxes, y_scores):
            elements.append(
                {"type": "icon", "bbox_xyxy_px": box, "bbox_xyxy": normalize_xyxy(box, W, H),
                 "interactive": True, "content": "icon", "score": round(sc, 3), "source": "yolo"}
            )

        elements = sorted(elements, key=lambda x: (x["type"] != "text",))
        anno = self._annotate(image_np, [e["bbox_xyxy_px"] for e in elements])
        return anno, elements


#######################################################################
#  AppleScript helpers (robust quoting)
#######################################################################
def osa_quote(s: str) -> str:
    return s.replace('"', r'\"')

def run_osa(script: str, timeout: float = 2.0) -> str:
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "").strip()
    except Exception:
        return ""

def focus_app(appname: str):
    """Bring an app to front by name."""
    subprocess.run([
        "osascript", "-e",
        f'tell application "{appname}" to activate'
    ], check=False)

def open_or_focus_app(appname: str):
    """Open or, if already running, re-activate the app."""
    focus_app(appname)
    subprocess.run([
        "osascript", "-e",
        f'tell application "{appname}" to reopen'
    ], check=False)

def get_chrome_url() -> str:
    """Return the URL of Chrome’s active tab."""
    script = '''
tell application "Google Chrome"
    if (count of windows)=0 then return ""
    return URL of active tab of front window
end tell
'''
    return run_osa(script)

def clipboard_write_text(text: str):
    """Copy text into macOS clipboard."""
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(input=text.encode())

def clipboard_read_text() -> str:
    """Read text from macOS clipboard."""
    try:
        return subprocess.check_output(["pbpaste"]).decode("utf-8", errors="ignore")
    except:
        return ""
# ─────────────────────────────────────────────────────────────────────────────

def get_front_window_bounds(app: str) -> Optional[Tuple[int, int, int, int]]:
    esc = osa_quote(app)
    script = f'''
    tell application "{esc}"
        if not (exists front window) then return "NONE"
        set b to bounds of front window
        return (item 1 of b as string) & "," & (item 2 of b as string) & "," & (item 3 of b as string) & "," & (item 4 of b as string)
    end tell
    '''
    out = run_osa(script, timeout=2.0)
    if not out or out == "NONE":
        return None
    try:
        parts = [int(float(x.strip())) for x in out.split(",")]
        if len(parts) != 4:
            return None
        x1, y1, x2, y2 = parts
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w <= 0 or h <= 0:
            return None
        return (x1, y1, w, h)
    except Exception:
        return None


class AppWatcher:
    def __init__(self, trace: "ChatTrace", app_name: str):
        self.trace = trace
        self.app = app_name

    @staticmethod
    def _is_running(app: str) -> bool:
        esc = osa_quote(app)
        out = run_osa(f'tell application "System Events" to (exists process "{esc}")', timeout=1.0)
        return out.strip().lower() == "true"

    @staticmethod
    def _reopen(app: str):
        esc = osa_quote(app)
        run_osa(f'tell application "{esc}" to reopen', timeout=1.0)

    @staticmethod
    def _activate(app: str):
        esc = osa_quote(app)
        run_osa(f'tell application "{esc}" to activate', timeout=1.0)

    def wait_until_running(self, timeout_sec: float = 15.0, poll: float = 0.3) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            if self._is_running(self.app):
                return True
            time.sleep(poll)
        return False

    def wait_until_front_window(self, timeout_sec: float = 20.0, poll: float = 0.4) -> Optional[Tuple[int, int, int, int]]:
        t0 = time.time()
        nudged = False
        while time.time() - t0 < timeout_sec:
            b = get_front_window_bounds(self.app)
            if b:
                return b
            if not nudged:
                self._reopen(self.app)
                self._activate(self.app)
                nudged = True
            time.sleep(poll)
        return None


class ChromeWatcher:
    def __init__(self, trace: "ChatTrace", target_substring: str):
        self.trace = trace
        self.target_sub = target_substring or ""
        try:
            self.target_domain = urlparse(self.target_sub).netloc
        except Exception:
            self.target_domain = ""

    @staticmethod
    def _parse_bounds_csv(csv_str: str) -> Optional[Tuple[int, int, int, int]]:
        try:
            parts = [int(float(x.strip())) for x in csv_str.split(",")]
            if len(parts) != 4: return None
            x1, y1, x2, y2 = parts
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            if w <= 0 or h <= 0: return None
            return (x1, y1, w, h)
        except Exception:
            return None

    @staticmethod
    def _focus_by_substring(needle: str) -> Optional[Tuple[int, int, int, int]]:
        if not needle:
            return None
        esc = osa_quote(needle)
        script = f'''
        tell application "Google Chrome"
            if (count of windows) = 0 then
                make new window
            end if
            repeat with w in windows
                set ti to 1
                repeat with t in tabs of w
                    try
                        set u to URL of t
                    on error
                        set u to ""
                    end try
                    if u contains "{esc}" then
                        set active tab index of w to ti
                        set index of w to 1
                        set b to bounds of w
                        return (item 1 of b as string) & "," & (item 2 of b as string) & "," & (item 3 of b as string) & "," & (item 4 of b as string)
                    end if
                    set ti to ti + 1
                end repeat
            end repeat
            return "NONE"
        end tell
        '''
        out = run_osa(script, timeout=2.5)
        if out and out != "NONE":
            return ChromeWatcher._parse_bounds_csv(out)
        return None

    def focus_tab_and_get_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        run_osa('tell application "Google Chrome" to activate', timeout=1.0)
        b = self._focus_by_substring(self.target_sub)
        if b:
            return b
        if self.target_domain:
            self.trace.info(f"URL substring not found; trying domain fallback '{self.target_domain}' …")
            b = self._focus_by_substring(self.target_domain)
            if b:
                return b
        return None

    def wait_until_visible(self, timeout_sec: float = 45.0, poll: float = 0.7) -> Optional[Tuple[int, int, int, int]]:
        t0 = time.time()
        attempts = 0
        while time.time() - t0 < timeout_sec:
            b = self.focus_tab_and_get_bounds()
            attempts += 1
            if b:
                self.trace.observe(f"Chrome tab matched; bounds={b} (after {attempts} attempt{'s' if attempts>1 else ''})")
                return b
            if attempts % int(max(1, 3/poll)) == 0:
                self.trace.info("Waiting for Chrome to expose the matching tab in a frontmost window…")
            time.sleep(poll)
        self.trace.warn("Timed out waiting for Chrome tab with requested URL/domain.")
        return None


#######################################################################
#  Planner (LLM)
#######################################################################

def _has_explicit_url(text: str) -> bool:
    if re.search(r'https?://\S+', text, flags=re.I):
        return True
    return bool(re.search(
        r'\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|co|gov|edu|info|biz|app|dev|ai|gg|hk|jp|uk|ca|de|fr|es|it|nl|au|sg|in|cn|tw)\b',
        text, flags=re.I
    ))

def _extract_first_url_like(text: str) -> Optional[str]:
    m = re.search(r'(https?://\S+)', text, flags=re.I)
    if m:
        return m.group(1).strip()
    m = re.search(
        r'\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|co|gov|edu|info|biz|app|dev|ai|gg|hk|jp|uk|ca|de|fr|es|it|nl|au|sg|in|cn|tw)\b',
        text, flags=re.I
    )
    if m:
        return f"https://{m.group(0).strip().lower()}"
    return None

def _looks_like_web_intent(text: str) -> bool:
    t = norm_text(text)
    keywords = [
        "chrome","browser","safari","firefox","edge",
        "open website","open site","open page","go to","navigate to",
        "search","google","bing","duckduckgo","url","link"
    ]
    return any(k in t for k in keywords)

def _url_from_clarifs(clarifs: Dict[str, str]) -> Optional[str]:
    for v in (clarifs or {}).values():
        if not isinstance(v, str):
            continue
        url = _extract_first_url_like(v)
        if url:
            return url
    return None


class Planner:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.planner_n = PLANNER_N
        self.temperature = PLANNER_TEMP

    def clarifications(self, goal: str) -> Dict[str, str]:
        ensure_openai()
        sys_prompt = (
            "You help plan GUI automation tasks for macOS. Ask only necessary questions.\n"
            'Return JSON: {"questions":["..."]}. If a product page lacks a URL, ask for it.'
        )
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Goal: {goal}\nReturn JSON only."},
        ]

        questions: List[str] = []
        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL_PLANNER,
                messages=msgs,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            questions = [q for q in data.get("questions", []) if isinstance(q, str)]
        except Exception:
            questions = []

        if _looks_like_web_intent(goal) and not _has_explicit_url(goal):
            url_q = "Do you have a specific URL for this (paste it), or should I search instead? (You can leave blank to let me search.)"
            if not any(norm_text(url_q) == norm_text(x) for x in questions):
                questions.insert(0, url_q)

        answers: Dict[str, str] = {}
        for q in questions:
            try:
                ans = input(f"Clarify – {q}\nYour answer: ").strip()
            except KeyboardInterrupt:
                ans = ""
            answers[q] = ans

        url_from_answers = _url_from_clarifs(answers)
        if url_from_answers:
            answers["_preferred_url"] = url_from_answers
        return answers

    def plan(self, goal: str, clarifs: Dict[str, str]) -> List[Step]:
        ensure_openai()
        sys_prompt = (
            "You are a GUI-automation planner for macOS.\n"
            'Return JSON: {"steps":[{"description":"...","action":"open_application|navigate|omnibox_search|click|type|scroll|read_text|execute_command|notify|hotkey|move|double_click|select_size|add_to_bag|open_settings_section|reach_target","params":{}}]}.\n'
            "Prefer generic actions. Use reach_target when the user wants a section/page/feature but no explicit selector is given."
        )
        usr = f"GOAL:\n{goal}\n\nCLARIFICATIONS:\n{json.dumps(clarifs, ensure_ascii=False)}\nReturn JSON only."
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr}]

        steps: List[Step] = []
        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL_PLANNER,
                messages=msgs,
                max_tokens=800,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                n=max(1, int(self.planner_n)),
            )
        except Exception as e:
            log.error("Plan failed: %s", e)
            steps = []
        else:
            candidates: List[List[Step]] = []
            for choice in getattr(resp, "choices", []):
                try:
                    data = json.loads(choice.message.content)
                    raw_steps = data.get("steps", [])
                    steps_i = [
                        Step(description=str(s.get("description", "")).strip(),
                             action=str(s.get("action", "")).strip(),
                             params=dict(s.get("params", {}) or {}))
                        for s in raw_steps
                    ]
                    candidates.append(steps_i)
                except Exception:
                    continue

            def _score(plan: List[Step]) -> float:
                if not plan: return -1e9
                acts = [s.action for s in plan]
                score = 0.0
                score += 2.0 if "reach_target" in acts else 0.0
                score += 0.4 * min(len(plan), 10)                 # reasonable depth
                score -= 0.2 * abs(len(plan) - 6)                 # avoid too short/long
                score += 0.5 if any(a in ("navigate","omnibox_search") for a in acts) else 0.0
                return score

            steps = max(candidates, key=_score) if candidates else []

        # Mild normalization (browser intents)
        url_pref = clarifs.get("_preferred_url") or _extract_first_url_like(goal)
        mention_browser = _looks_like_web_intent(goal)

        if mention_browser:
            # Ensure Chrome is opened
            steps = [s for s in steps if s.action not in ("open_application",)]
            steps.insert(0, Step("Open Google Chrome", "open_application", {"app": "Google Chrome"}))
            if url_pref:
                # Navigate to the provided URL explicitly
                if not any(s.action == "navigate" for s in steps):
                    steps.insert(1, Step("Navigate to the provided URL", "navigate", {"url": url_pref}))
            else:
                # Ensure there is at least one search step
                if not any(s.action in ("navigate","omnibox_search") for s in steps):
                    # --- new: ask the planner‐LLM to rewrite the goal into a concise search query ---
                    prompt  = (
                        "Please rewrite the following user goal into a concise search query "
                        "suitable for a browser omnibox (remove filler words):\n\n"
                        f"{goal}"
                    )
                    resp = self.client.chat.completions.create(
                        model=OPENAI_MODEL_PLANNER,
                        messages=[
                            {"role":"system","content": "You are a search‐query rewriter."},
                            {"role":"user",  "content": prompt},
                        ],
                        max_tokens=64,
                        temperature=0.2,
                    )
                    q = resp.choices[0].message.content.strip()
                    steps.insert(1, Step(
                        "Search via Chrome omnibox",
                        "omnibox_search",
                        {"query": q},
                    ))

            # Inject openai_terminal_output if output/reporting intent detected
        if needs_openai_output(goal, steps):
            steps.append(
                Step(
                    description="OpenAI-powered answer/output to user's goal",
                    action="openai_terminal_output",
                    params={
                        "prompt": f"Based on everything currently shown, answer the user's goal: '{goal}'.",
                    }
                )
            )
        return steps or [Step("Notify fallback", "notify", {"message": "Plan failed; please retry."})]


class UIPriors:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)

    def synonyms(self, app: str, role: str, fallback: List[str]) -> List[str]:
        ensure_openai()
        sys_p = ("You are a concise UI assistant. Given an app and a control role, "
                 "return a short list of exact on-screen labels/placeholders/synonyms the app uses. "
                 'Respond as JSON: {"variants":["...","..."]}. No explanations.')
        user_p = f"app={app}\nrole={role}\nknown_fallback={fallback}"
        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL_PLANNER,
                messages=[{"role":"system","content":sys_p},
                          {"role":"user","content":user_p}],
                max_tokens=150,
                response_format={"type":"json_object"},
                temperature=0.1
            )
            data = json.loads(resp.choices[0].message.content)
            out = [v for v in data.get("variants", []) if isinstance(v, str)]
            seen = {norm_text(x) for x in fallback}
            for v in out:
                if norm_text(v) not in seen:
                    fallback.append(v)
            return fallback
        except Exception as e:
            log.warning("UIPriors fallback (synonyms failed): %s", e)
            return fallback


#######################################################################
#  Experience memory (tiny bandit) & Target lexicon
#######################################################################
class ExperienceMemory:
    """Tiny bandit memory: remembers which labels worked per domain."""
    def __init__(self, path: str):
        self.path = path
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        except Exception:
            self.db = {}

    def _key(self, domain: str) -> str:
        return norm_text(domain or "global")

    def score(self, domain: str, label: str) -> float:
        d = self.db.get(self._key(domain), {})
        return float(d.get(norm_text(label), 0.0))

    def bump(self, domain: str, label: str, delta: float):
        k = self._key(domain)
        if k not in self.db: self.db[k] = {}
        lab = norm_text(label)
        self.db[k][lab] = float(self.db[k].get(lab, 0.0) + delta)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.db, f, indent=2)
        except Exception:
            pass


class TargetLexicon:
    """
    Given a high-level subgoal and (optionally) app/domain context, produce dynamic tokens:
      - targets: labels/buttons/links likely to advance the task
      - exposure: things to open first (e.g. “Menu”, “More”, “☰”) if targets are hidden
      - success_text: text that indicates success on-screen
      - url_substrings: URL hints that indicate success
    """
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)

    def from_goal(self, goal: str, app: str = "", domain: str = "") -> Dict[str, List[str]]:
        ensure_openai()
        sys_p = (
            "You are a UI navigator. Given a user's subgoal and context, return JSON:\n"
            '{"targets":["..."],"exposure":["..."],"success_text":["..."],"url_substrings":["..."]}.\n'
            "Use short, literal on-screen labels; include emojis if likely (e.g., ☰). Keep under 8 per list."
        )
        user_p = json.dumps({"goal": goal, "app": app, "domain": domain}, ensure_ascii=False)
        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL_PLANNER,
                messages=[{"role":"system","content":sys_p},
                          {"role":"user","content":user_p}],
                max_tokens=220,
                response_format={"type":"json_object"},
                temperature=0.1
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception:
            data = {}
        targets = [t for t in data.get("targets", []) if isinstance(t, str)] or tokenize(goal)[:6]
        exposure = [t for t in data.get("exposure", []) if isinstance(t, str)] or ["Menu","More","☰","Browse","Categories"]
        success_text = [t for t in data.get("success_text", []) if isinstance(t, str)] or [goal]
        url_substrings = [t for t in data.get("url_substrings", []) if isinstance(t, str)] or []
        def _dedup(xs):
            seen=set(); out=[]
            for x in xs:
                xn=norm_text(x)
                if xn and xn not in seen: seen.add(xn); out.append(x)
            return out[:8]
        return {
            "targets": _dedup(targets),
            "exposure": _dedup(exposure),
            "success_text": _dedup(success_text),
            "url_substrings": _dedup(url_substrings)
        }


###########################
# Element search helpers  #
###########################
def find_text_exact(elements: List[Dict[str, Any]], candidates: List[str], anchor: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
    cand_norm = [norm_text(c) for c in candidates if str(c).strip()]
    if not cand_norm: return None
    best, best_score = None, -1e9
    for e in elements:
        if e.get("type") != "text": continue
        content = norm_text(str(e.get("content", "")))
        if not any(c == content or c in content for c in cand_norm): continue
        score = 10.0
        if anchor and e.get("bbox_xyxy_px"):
            ax1, ay1, ax2, ay2 = anchor
            acx, acy, _, _ = xyxy_to_cxcywh(ax1, ay1, ax2, ay2)
            ex1, ey1, ex2, ey2 = e["bbox_xyxy_px"]
            ecx, ecy, _, _ = xyxy_to_cxcywh(ex1, ey1, ex2, ey2)
            dist = max(1.0, ((ecx - acx)**2 + (ecy - ecy)**2) ** 0.5)
            score += 2000.0 / dist
        if score > best_score:
            best_score, best = score, e
    return best


def find_best_element(elements: List[Dict[str, Any]], hint: str, anchor: Optional[List[int]] = None, prefer_text: bool = False) -> Optional[Dict[str, Any]]:
    hint_l = norm_text(hint)
    best = None
    best_score = -1.0
    for e in elements:
        content = norm_text(str(e.get("content", "")))
        score = 0.0
        if matches_hint(content, hint_l): score += 2.5
        if prefer_text and e.get("type") == "text": score += 1.0
        if e.get("type") == "icon": score += 0.4
        if anchor and e.get("bbox_xyxy_px"):
            ax1, ay1, ax2, ay2 = anchor
            acx, acy, _, _ = xyxy_to_cxcywh(ax1, ay1, ax2, ay2)
            ex1, ey1, ex2, ey2 = e["bbox_xyxy_px"]
            ecx, ecy, _, _ = xyxy_to_cxcywh(ex1, ey1, ex2, ey2)
            dist = max(1.0, ((ecx - acx)**2 + (ecy - ecy)**2) ** 0.5)
            score += 2000.0 / dist
        if score > best_score:
            best_score = score
            best = e
    return best


def find_icon_near_text(elements: List[Dict[str, Any]], anchor_text_variants: List[str], max_px: int = 500) -> Optional[Dict[str, Any]]:
    anchor = find_text_exact(elements, anchor_text_variants)
    if not anchor or not anchor.get("bbox_xyxy_px"):
        return None
    ax1, ay1, ax2, ay2 = anchor["bbox_xyxy_px"]
    acx, acy, _, _ = xyxy_to_cxcywh(ax1, ay1, ax2, ay2)

    best = None
    best_d = float("inf")
    for e in elements:
        if e.get("type") != "icon" or not e.get("bbox_xyxy_px"):
            continue
        x1, y1, x2, y2 = e["bbox_xyxy_px"]
        cx, cy, _, _ = xyxy_to_cxcywh(x1, y1, x2, y2)
        d = ((cx - acx) ** 2 + (cy - acy) ** 2) ** 0.5
        if d < best_d and d <= max_px:
            best_d = d
            best = e
    return best


def pick_with_confidence(elements: List[Dict[str, Any]], hints: List[str],
                         anchor: Optional[List[int]]=None,
                         prefer_text: bool=True) -> Tuple[Optional[Dict[str, Any]], float, str]:
    best_e, best_c, best_h = None, 0.0, ""
    for h in hints:
        h_norm = norm_text(h)
        e = find_text_exact(elements, [h_norm], anchor=anchor)
        if e:
            conf = 0.95
        else:
            e = find_best_element(elements, h_norm, anchor=anchor, prefer_text=prefer_text)
            if e:
                conf = 0.55 + 0.35 * fuzzy_score(str(e.get("content","")), h_norm)
            else:
                conf = 0.0
        if e and conf > best_c:
            best_e, best_c, best_h = e, conf, h
    return best_e, best_c, best_h


#######################################################################
#  Reflex: micro-step inference & backtracking
#######################################################################
class Reflexor:
    def __init__(self, trace: ChatTrace):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.trace = trace
        self.n_votes = LLM_DELIBERATE
        self.temperature = REFLEX_TEMP
        self.visible_text_limit = VISIBLE_TEXT_LIMIT

    def propose(self, goal: str, visible_text: List[str], temperature: Optional[float]=None) -> Dict[str, Any]:
        ensure_openai()
        sys_p = (
            "You propose the next **small** GUI action (click text, click text-near-icon, type+enter, scroll) "
            "to move towards a high-level goal on a website or app. You can infer missing steps "
            "(e.g., open a menu before selecting an item). Prefer clicking the most promising visible label.\n"
            'Return JSON: {"action":"click|typeenter|scroll|noop","target":"<label or hint>","rationale":"..."}'
        )
        user_p = json.dumps({
            "goal": goal,
            "visible_text_preview": visible_text[: self.visible_text_limit]
        }, ensure_ascii=False)
        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL_REFLEX,
                messages=[{"role":"system","content":sys_p},
                          {"role":"user","content":user_p}],
                max_tokens=180,
                response_format={"type":"json_object"},
                temperature=self.temperature if temperature is None else temperature,
            )
            data = json.loads(resp.choices[0].message.content)
            a = data.get("action","noop")
            t = data.get("target","")
            r = data.get("rationale","")
            return {"action": a, "target": t, "why": r}
        except Exception as e:
            self.trace.warn(f"Reflexor failed: {e}")
            return {"action": "noop", "target": "", "why": "fallback"}

    def propose_consensus(self, goal: str, visible_text: List[str], k: Optional[int]=None) -> Dict[str, Any]:
        k = max(1, int(k or self.n_votes))
        if k == 1:
            return self.propose(goal, visible_text)
        ballots = []
        # small temperature sweep encourages diversity without chaos
        base = self.temperature
        temps = [max(0.1, min(0.9, base + delta)) for delta in np.linspace(0, 0.4, num=k)]
        for t in temps:
            ballots.append(self.propose(goal, visible_text, temperature=t))
        def key(b):
            return (b.get("action","noop"), norm_text(b.get("target","")))
        counts = Counter(key(b) for b in ballots if b.get("action"))
        if not counts:
            return ballots[0] if ballots else {"action":"noop","target":"","why":"no ballots"}
        winner = counts.most_common(1)[0][0]
        for b in ballots:
            if key(b) == winner:
                b["why"] = f"{b.get('why','')} [consensus k={k}]"
                return b
        return ballots[0]

# ------------------------------------------------------------------
# PenaltyMemory: Remembers failed clicks/targets per domain/label.
class PenaltyMemory:
    def __init__(self, path=os.path.join(OUT_DIR, "penalty_memory.json")):
        self.path = path
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        except Exception:
            self.db = {}

    def key(self, domain, label):
        return f"{norm_text(domain)}::{norm_text(label)}"

    def bump(self, domain, label):
        k = self.key(domain, label)
        self.db[k] = int(self.db.get(k, 0)) + 1
        self._save()

    def penalty(self, domain, label):
        k = self.key(domain, label)
        return int(self.db.get(k, 0))

    def reset(self, domain=None):
        if domain:
            keys = [k for k in self.db if k.startswith(norm_text(domain) + "::")]
            for k in keys:
                self.db.pop(k, None)
        else:
            self.db = {}
        self._save()

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.db, f, indent=2)
        except Exception:
            pass
# ------------------------------------------------------------------

# ------------------------------------------------------------------
def adaptive_backtrack(trace, domain, executor, subgoal, elems, base_img):
    """Try Esc, browser back, and LLM replan for stuck state."""
    trace.think("Adaptive backtrack: Esc, browser Back, then LLM alternative.")
    if HAVE_PYAUTOGUI:
        pyautogui.press("esc")
    else:
        keyboard.press(Key.esc); keyboard.release(Key.esc)
    time.sleep(0.7)
    try:
        if executor._get_current_url():
            if HAVE_PYAUTOGUI:
                pyautogui.hotkey("command", "[")
            else:
                keyboard.press(Key.cmd); keyboard.press("["); keyboard.release("["); keyboard.release(Key.cmd)
            time.sleep(0.6)
    except Exception:
        pass
    dynamic_replan_with_llm(trace, domain, executor, subgoal, elems, base_img)

def dynamic_replan_with_llm(trace, domain, executor, subgoal, elems, base_img):
    """Ask LLM for a last-resort strategy, given UI state."""
    visible_ui = [
        {
            "type": e.get("type"),
            "text": e.get("content", ""),
            "box": e.get("bbox_xyxy_px"),
        } for e in elems if e.get("type") in {"text", "icon"}
    ]
    prompt = (
        "You are an advanced UI automation agent. "
        f"You failed to reach the subgoal: '{subgoal}'. "
        f"Here are the UI elements detected: {json.dumps(visible_ui)}. "
        "Suggest exactly what element (by label) should be clicked, or what alternative strategy to use. "
        "If you suggest an action, format as JSON: {'action':'click/type/scroll','target':'...','reason':'...'}."
    )
    trace.think(f"Dynamic replan prompt: {prompt}")
    llm = OpenAI(api_key=OPENAI_KEY)
    try:
        resp = llm.chat.completions.create(
            model=OPENAI_MODEL_REFLEX,
            messages=[{"role": "system", "content": "You are a UI agent re-planning after a failure."},
                      {"role": "user", "content": prompt}],
            max_tokens=200,
            response_format={"type": "json_object"},
            temperature=0.35,
        )
        alt = json.loads(resp.choices[0].message.content)
        act = alt.get("action", "noop")
        tgt = alt.get("target", "")
        if act == "click" and tgt:
            for e in elems:
                if norm_text(e.get("content", "")) == norm_text(tgt):
                    executor._click_box(e["bbox_xyxy_px"], rescan_after=True, label=tgt)
                    trace.outcome(f"Dynamic replan: LLM chose to click '{tgt}'.")
                    return
        elif act == "scroll":
            if HAVE_PYAUTOGUI: pyautogui.scroll(-600)
            else:
                for _ in range(30): mouse.scroll(0, -3); time.sleep(0.005)
            executor._after_ui_change()
            trace.outcome("Dynamic replan: LLM chose to scroll.")
        elif act == "type" and tgt:
            executor._ensure_focus(executor.active_app or "Google Chrome")
            if HAVE_PYAUTOGUI:
                pyautogui.write(tgt, interval=0.01)
            else:
                keyboard.type(tgt)
            executor._after_ui_change()
            trace.outcome(f"Dynamic replan: LLM typed '{tgt}'.")
    except Exception as e:
        trace.warn(f"Dynamic replan LLM failed: {e}")

def goal_state_reached(goal: str, current_url: str, ocr_texts: List[str]) -> bool:
    """Heuristic: Is the goal (by keywords/phrases/URL hints) present on screen or in the URL?"""
    # This can be more sophisticated; here's a generic heuristic:
    goal = norm_text(goal)
    tokens = [t for t in goal.split() if t]
    url_hit = any(t in norm_text(current_url) for t in tokens)
    text_hit = any(t in norm_text(ot) for ot in ocr_texts for t in tokens)
    return url_hit or text_hit

# ------------------------------------------------------------------

#######################################################################
#  Executor
#######################################################################
class Executor:
    def __init__(self, parser_model: OmniParser, trace: ChatTrace, deliberate: int = LLM_DELIBERATE, reach_max_iters: int = REACH_MAX_ITERS_DEFAULT):
        self.driver: Optional[webdriver.Chrome] = None
        self.parser = parser_model
        self.trace = trace
        self.anchor_box: Optional[List[int]] = None
        self.target_url_sub: Optional[str] = None
        self.chrome_bounds: Optional[Tuple[int, int, int, int]] = None
        self.active_app: Optional[str] = None
        self.current_save_dir: Optional[str] = None
        self.ui_priors = UIPriors()
        self.penalty_memory = PenaltyMemory()
        self.reflexor = Reflexor(trace)
        self._last_fp: Optional[str] = None
        self._repeat_clicks = 0
        self._last_click_label: Optional[str] = None
        self._cookie_attempts = 0

        ### [ADDED] runtime toggles
        self.reuse_chrome: bool = False
        self.paste_link_same_window: bool = False
        self.debugger_address: str = "127.0.0.1:9222"

        # NEW knobs
        self.deliberate = max(1, int(deliberate))
        self.reach_max_iters = int(reach_max_iters)

    # ------------------ Driver lifecycle ------------------
    def _ensure_driver(self, headless: bool = False):
        if self.driver:
            return
        opts = Options()
        opts.add_argument("--start-maximized")

        ### [ADDED] Try to attach to existing Chrome via remote debugging if asked
        if self.reuse_chrome:
            try:
                opts.add_experimental_option("debuggerAddress", self.debugger_address)
                self.driver = webdriver.Chrome(options=opts)
                self.trace.info(f"Attached to existing Chrome at {self.debugger_address}")
                return
            except Exception as e:
                self.trace.warn(f"Attach to existing Chrome failed ({e}); starting managed Chrome.")

        # Original behavior (managed Chrome). Keep "--new-window" only if not reusing.
        if not self.reuse_chrome:
            opts.add_argument("--new-window")
        if headless:
            opts.add_argument("--headless=new")
        try:
            self.driver = webdriver.Chrome(options=opts)
        except Exception as e:
            self.trace.warn(f"ChromeDriver could not start: {e}")
            raise
    # ------------------------------------------------------

    def _save_artifacts(self, anno_img: Image.Image, elems: List[Dict[str, Any]], tag: str):
        if not self.current_save_dir:
            return
        ts = int(time.time())
        try:
            anno_img.save(os.path.join(self.current_save_dir, f"annotated_{tag}_{ts}.png"))
            with open(os.path.join(self.current_save_dir, f"elements_{tag}_{ts}.json"), "w") as f:
                json.dump({"elements": elems}, f, indent=2)
            self.trace.info(f"Saved artifacts to {self.current_save_dir} (tag={tag})")
        except Exception:
            pass

    def _rescan_active_window(self, looking_for: Optional[List[str]] = None) -> Tuple[Image.Image, List[Dict[str, Any]], Image.Image]:
        screenW, screenH = get_screen_size()

        # Chrome targeted crop
        if self.active_app and "chrome" in self.active_app.lower():
            # Try to lock to the specific tab (via URL substring) if available
            if self.target_url_sub:
                cw = ChromeWatcher(self.trace, self.target_url_sub)
                self.chrome_bounds = cw.focus_tab_and_get_bounds() or self.chrome_bounds

            # Fallback: any front Chrome window
            if not self.chrome_bounds:
                b = get_front_window_bounds("Google Chrome")
                if b:
                    self.chrome_bounds = b

            # If we still don't have bounds, full-screen fallback
            if not self.chrome_bounds:
                base = screenshot()
                anno, elems = self.parser.parse(base)
                fp = hashlib.sha1(base.tobytes()).hexdigest()[:8]
                n_text = sum(1 for e in elems if e.get("type") == "text")
                n_icons = sum(1 for e in elems if e.get("type") == "icon")
                sample_texts = [e.get("content","") for e in elems if e.get("type") == "text"][:6]
                self._last_fp = fp
                self.trace.parse_snapshot(base.width, base.height, n_text, n_icons, fp, looking_for, sample_texts)
                return anno, elems, base

            # We have Chrome window bounds—crop to them
            left, top, width, height = self.chrome_bounds
            base = screenshot(region=(left, top, width, height))
            anno_local, elems_local = self.parser.parse(base)

            # Promote local coords to global/screen coords
            elems: List[Dict[str, Any]] = []
            for e in elems_local:
                x1, y1, x2, y2 = e.get("bbox_xyxy_px", [0, 0, 0, 0])
                gx1, gy1, gx2, gy2 = int(x1 + left), int(y1 + top), int(x2 + left), int(y2 + top)
                e2 = dict(e)
                e2["bbox_xyxy_px"] = [gx1, gy1, gx2, gy2]
                e2["bbox_xyxy"] = normalize_xyxy([gx1, gy1, gx2, gy2], screenW, screenH)
                elems.append(e2)

            fp = hashlib.sha1(base.tobytes()).hexdigest()[:8]
            n_text = sum(1 for e in elems_local if e.get("type") == "text")
            n_icons = sum(1 for e in elems_local if e.get("type") == "icon")
            sample_texts = [e.get("content","") for e in elems_local if e.get("type") == "text"][:6]
            self._last_fp = fp
            self.trace.parse_snapshot(base.width, base.height, n_text, n_icons, fp, looking_for, sample_texts)
            return anno_local, elems, base

        # Other apps: crop to their front window if possible
        if self.active_app:
            b = get_front_window_bounds(self.active_app)
            if b:
                left, top, width, height = b
                base = screenshot(region=(left, top, width, height))
                anno_local, elems_local = self.parser.parse(base)
                elems: List[Dict[str, Any]] = []
                for e in elems_local:
                    x1, y1, x2, y2 = e.get("bbox_xyxy_px", [0, 0, 0, 0])
                    gx1, gy1, gx2, gy2 = int(x1 + left), int(y1 + top), int(x2 + left), int(y2 + top)
                    e2 = dict(e)
                    e2["bbox_xyxy_px"] = [gx1, gy1, gx2, gy2]
                    e2["bbox_xyxy"] = normalize_xyxy([gx1, gy1, gx2, gy2], screenW, screenH)
                    elems.append(e2)
                fp = hashlib.sha1(base.tobytes()).hexdigest()[:8]
                n_text = sum(1 for e in elems_local if e.get("type") == "text")
                n_icons = sum(1 for e in elems_local if e.get("type") == "icon")
                sample_texts = [e.get("content","") for e in elems_local if e.get("type") == "text"][:6]
                self._last_fp = fp
                self.trace.parse_snapshot(base.width, base.height, n_text, n_icons, fp, looking_for, sample_texts)
                return anno_local, elems, base

        # Full-screen fallback
        base = screenshot()
        anno, elems = self.parser.parse(base)
        fp = hashlib.sha1(base.tobytes()).hexdigest()[:8]
        n_text = sum(1 for e in elems if e.get("type") == "text")
        n_icons = sum(1 for e in elems if e.get("type") == "icon")
        sample_texts = [e.get("content","") for e in elems if e.get("type") == "text"][:6]
        self._last_fp = fp
        self.trace.parse_snapshot(base.width, base.height, n_text, n_icons, fp, looking_for, sample_texts)
        return anno, elems, base

    def _after_ui_change(self, wait: float = WAIT_AFTER_CLICK, looking_for: Optional[List[str]] = None):
        # If you just clicked "MENU", wait longer!
        time.sleep(wait)
        anno, elems, _ = self._rescan_active_window(looking_for=looking_for)
        self._save_artifacts(anno, elems, tag="afterclick")
        return anno, elems

    def _hover_box(self, box: List[int], pause: float = 0.6):
        x1, y1, x2, y2 = box
        cx, cy, _, _ = xyxy_to_cxcywh(x1, y1, x2, y2)
        if HAVE_PYAUTOGUI:
            pyautogui.moveTo(int(cx), int(cy))
        else:
            mouse.position = (int(cx), int(cy))
        time.sleep(pause)

    def _click_box(self, box: List[int], clicks: int = 1, rescan_after: bool = True, looking_for: Optional[List[str]] = None, label: Optional[str]=None):
        x1, y1, x2, y2 = box
        cx, cy, _, _ = xyxy_to_cxcywh(x1, y1, x2, y2)
        if HAVE_PYAUTOGUI:
            pyautogui.moveTo(int(cx), int(cy))
            if clicks == 2: pyautogui.doubleClick()
            else: pyautogui.click()
        else:
            mouse.position = (int(cx), int(cy))
            time.sleep(0.05)
            mouse.click(Button.left)
            if clicks == 2:
                time.sleep(0.05); mouse.click(Button.left)
        self.anchor_box = [x1, y1, x2, y2]
        lbl = norm_text(label or "")
        if lbl and lbl == (self._last_click_label or ""):
            self._repeat_clicks += 1
        else:
            self._repeat_clicks = 0
        self._last_click_label = lbl or self._last_click_label

        # ---- PATCH START ----
        menu_labels = {"menu", "☰", "browse", "categories", "more"}
        if label and norm_text(label) in menu_labels:
            time.sleep(1.5)  # Wait for menu to animate open
            anno, elems, _ = self._rescan_active_window(looking_for=looking_for)
            self._save_artifacts(anno, elems, tag="after_menu")
        # ---- PATCH END ----

        if rescan_after:
            self._after_ui_change(looking_for=looking_for)

    def _double_click_box(self, box: List[int], looking_for: Optional[List[str]]=None, label: Optional[str]=None):
        self._click_box(box, clicks=2, rescan_after=True, looking_for=looking_for, label=label)

    def _ensure_focus(self, app_name: str):
        run_osa(f'tell application "{osa_quote(app_name)}" to activate', timeout=1.0)
        b = get_front_window_bounds(app_name)
        if b:
            x, y, w, h = b
            cx, cy = int(x + w*0.5), int(y + h*0.1)
            if HAVE_PYAUTOGUI: pyautogui.click(cx, cy)
            else:
                mouse.position = (cx, cy); time.sleep(0.05); mouse.click(Button.left)
            time.sleep(0.2)

    def _focus_search_field(self, app_name: str, base_hints: List[str], min_conf: float=0.62) -> bool:
        anno, elements, _ = self._rescan_active_window(looking_for=base_hints)
        cand, conf, used = pick_with_confidence(elements, base_hints, anchor=None, prefer_text=True)
        self.trace.think(f"Search-field pick '{used}' confidence={conf:.2f}.")
        if not cand or conf < min_conf:
            prior_hints = self.ui_priors.synonyms(app_name, "search field", base_hints[:])
            self.trace.think(f"Low confidence; trying UI priors {prior_hints}")
            anno, elements, _ = self._rescan_active_window(looking_for=prior_hints)
            cand, conf, used = pick_with_confidence(elements, prior_hints, anchor=None, prefer_text=True)
            self.trace.think(f"After priors, pick '{used}' confidence={conf:.2f}.")
        if cand and conf >= 0.45:
            self.trace.observe(f"Focusing search field via '{cand.get('content','')}' at {cand.get('bbox_xyxy_px')}")
            self._click_box(cand["bbox_xyxy_px"], clicks=1, rescan_after=True, looking_for=None, label=cand.get("content",""))
            return True
        b = get_front_window_bounds(self.active_app or app_name) or self.chrome_bounds
        if b:
            x, y, w, h = b
            cx, cy = int(x + w*0.55), int(y + h*0.07)
            self.trace.think(f"Heuristic click to top bar at {(cx,cy)}.")
            if HAVE_PYAUTOGUI: pyautogui.click(cx, cy)
            else:
                mouse.position = (cx, cy); time.sleep(0.05); mouse.click(Button.left)
            self._after_ui_change()
            return True
        return False

    def _dom_click_text(self, texts: List[str]) -> bool:
        """Try to click by DOM using Selenium/JS by matching innerText (case-insensitive)."""
        if not self.driver:
            return False
        try:
            js = r'''
                const needles = JSON.parse(arguments[0]).map(s => s.toLowerCase());
                function visible(el){
                    const r = el.getBoundingClientRect();
                    const style = window.getComputedStyle(el);
                    return r.width>1 && r.height>1 && style.visibility!=="hidden" && style.display!=="none";
                }
                const candidates = Array.from(document.querySelectorAll('a,button,[role="menuitem"],[onclick], *'));
                const scored = [];
                for (const el of candidates){
                    if(!visible(el)) continue;
                    const t = (el.innerText || el.textContent || "").trim().toLowerCase();
                    if(!t) continue;
                    for (const n of needles){
                        if (t===n || t.includes(n)){
                            scored.push({el, t, score: t===n ? 2 : 1});
                            break;
                        }
                    }
                }
                if (scored.length===0) return false;
                scored.sort((a,b)=>b.score - a.score);
                const hit = scored[0].el;
                hit.scrollIntoView({block:"center", inline:"center"});
                hit.click();
                return true;
            '''
            ok = self.driver.execute_script(js, json.dumps(texts))
            return bool(ok)
        except Exception as e:
            self.trace.warn(f"DOM click failed: {e}")
            return False

    def _try_dismiss_banners(self, elements: List[Dict[str, Any]]) -> bool:
        if self._cookie_attempts > 3:
            return False
        labels = synonyms_for_cookie_accept()
        cand = find_text_exact(elements, labels, anchor=None)
        if not cand:
            cand, conf, used = pick_with_confidence(elements, labels, anchor=None, prefer_text=True)
        if cand:
            self.trace.think(f"Consent banner: clicking '{cand.get('content','')}'.")
            self._cookie_attempts += 1
            self._click_box(cand["bbox_xyxy_px"], rescan_after=True, label=cand.get("content",""))
            return True
        return False

    ### [ADDED] helper to get current URL even when not using Selenium navigation
    def _get_current_url(self) -> str:
        try:
            if self.driver:
                return (self.driver.current_url or "").strip()
        except Exception:
            pass
        try:
            script = '''
            tell application "Google Chrome"
                if (count of windows) = 0 then return ""
                try
                    return URL of active tab of front window
                on error
                    return ""
                end try
            end tell
            '''
            return (run_osa(script, timeout=1.0) or "").strip()
        except Exception:
            return ""

    # --------------------------------- Core executor ---------------------------------
    def execute(self, step, headless=False, save_dir=None):
        a = step.action
        p = step.params or {}
        desc = step.description
        self.current_save_dir = save_dir

        # ─── Always bring the correct app to front ───────────────────────────────
        if a in (
            "open_application","navigate","omnibox_search",
            "click","double_click","type","scroll",
            "open_settings_section","reach_target"
        ) and self.active_app:
            focus_app(self.active_app)
        # ──────────────────────────────────────────────────────────────────────────

        self.trace.action(f"{a} – {desc} (params={p})")

        # --- open_application ---
        if a == "open_application":
            app = (p.get("app") or p.get("application") or p.get("application_name") or "").strip()
            if not app:
                m = re.search(r"open\s+(.+)", desc, flags=re.I)
                if m: app = m.group(1).strip()
            app = re.sub(r"\s+application$", "", app, flags=re.I)
            if not app:
                self.trace.warn("No application name provided for open_application.")
                return False, None

            self.trace.think(f"Intent: launch/activate '{app}', then wait for a usable front window before continuing.")
            try:
                open_or_focus_app(app)
                watcher = AppWatcher(self.trace, app)
                ok_run = watcher.wait_until_running(timeout_sec=15.0, poll=0.3)
                if not ok_run:
                    self.trace.error(f"Process for '{app}' did not appear.")
                    return False, None
                watcher._activate(app)
                b = watcher.wait_until_front_window(timeout_sec=20.0, poll=0.4)
                if b:
                    self.trace.observe(f"{app} front window bounds={b}")
                else:
                    self.trace.warn(f"{app} has no visible front window yet. Proceeding anyway.")
                self.active_app = app
                if "chrome" in app.lower():
                    self.chrome_bounds = b or get_front_window_bounds("Google Chrome")
                else:
                    self.chrome_bounds = None
                self.trace.outcome(f"'{app}' is running and frontmost{' with window' if b else ''}.")
                anno, elems, _ = self._rescan_active_window()
                self._save_artifacts(anno, elems, tag="preaction")
                return True, None
            except Exception as e:
                self.trace.error(f"open_application failed: {e}")
                return False, None

        # --- navigate (URL or omnibox) ---
        if a == "navigate":
            url = p.get("url", "") or p.get("URL","")
            query = p.get("query","") or p.get("search","")
            if not url and not query:
                self.trace.warn("navigate called without a URL or query.")
                return False, None

            self.active_app = "Google Chrome"
            self.trace.think("Intent: open Chrome at URL or perform omnibox search; lock screenshots to that window.")
            try:
                # prefer reusing same window/tab when requested
                if url and self.paste_link_same_window:
                    self._ensure_focus("Google Chrome")
                    self.target_url_sub = url
                    self.trace.think("Using omnibox (Cmd+L) to paste URL in current tab (same window).")
                    if HAVE_PYAUTOGUI:
                        pyautogui.hotkey("command","l")
                        clipboard_write_text(url)
                        pyautogui.hotkey("command","v")
                        pyautogui.press("enter")
                    else:
                        keyboard.press(Key.cmd); keyboard.press('l'); keyboard.release('l'); keyboard.release(Key.cmd)
                        keyboard.type(url); keyboard.press(Key.enter); keyboard.release(Key.enter)
                    time.sleep(WAIT_AFTER_NAV)
                    self.chrome_bounds = get_front_window_bounds("Google Chrome")
                else:
                    # Fall back to Selenium-managed navigation/search
                    self._ensure_driver(headless=headless)
                    run_osa('tell application "Google Chrome" to activate', timeout=1.0)

                    if url:
                        self.target_url_sub = url
                        self.trace.think("Selenium get(url) → focus matching tab by URL/domain → read window bounds.")
                        self.driver.get(url)  # type: ignore
                        try:
                            WebDriverWait(self.driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))  # type: ignore
                        except Exception:
                            self.trace.warn("Body presence wait elapsed; continuing.")
                        cw = ChromeWatcher(self.trace, self.target_url_sub)
                        b = cw.wait_until_visible(timeout_sec=45.0, poll=0.7)
                        if not b:
                            self.trace.outcome("Failed to focus target tab/window.")
                            return False, None
                        self.chrome_bounds = b
                    else:
                        self.target_url_sub = ""
                        self._ensure_focus("Google Chrome")
                        self.trace.think("No URL provided; using omnibox (Cmd+L) then typing query + Enter.")
                        if HAVE_PYAUTOGUI:
                            pyautogui.hotkey("command","l")
                            pyautogui.write(query)
                            pyautogui.press("enter")
                        else:
                            keyboard.press(Key.cmd); keyboard.press('l'); keyboard.release('l'); keyboard.release(Key.cmd)
                            keyboard.type(query); keyboard.press(Key.enter); keyboard.release(Key.enter)
                        time.sleep(WAIT_AFTER_NAV)
                        self.chrome_bounds = get_front_window_bounds("Google Chrome")

                self.trace.outcome(f"Navigation succeeded. Window bounds={self.chrome_bounds}.")
                self._after_ui_change(wait=WAIT_AFTER_NAV)

                self.trace.next("Proceed to on‑page actions as needed.")
                return True, None
            except Exception as e:
                self.trace.error(f"navigate failed: {e}")
                anno, elems, _ = self._rescan_active_window()
                self._save_artifacts(anno, elems, tag="preaction")
                return False, None

        if a == "omnibox_search":
            q = (p.get("query") or p.get("q") or "").strip()
            if not q:
                self.trace.warn("omnibox_search missing query.")
                return False, None
            self.active_app = "Google Chrome"
            self._ensure_focus("Google Chrome")
            self.trace.think(f"Cmd+L to omnibox → type '{q}' + Enter.")
            if HAVE_PYAUTOGUI:
                pyautogui.hotkey("command","l")
                pyautogui.write(q)
                pyautogui.press("enter")
            else:
                keyboard.press(Key.cmd); keyboard.press('l'); keyboard.release('l'); keyboard.release(Key.cmd)
                keyboard.type(q); keyboard.press(Key.enter); keyboard.release(Key.enter)
            self._after_ui_change(wait=WAIT_AFTER_NAV)
            return True, None

        if a == "open_settings_section":
            section = (p.get("section") or p.get("target") or desc).strip() or "Battery"
            self.trace.think(f"Intent: open System Settings → search '{section}' → click section entry.")
            if not self.active_app or norm_text(self.active_app) != "system settings":
                subprocess.Popen(["open", "-a", "System Settings"])
                self.active_app = "System Settings"
                time.sleep(0.6)
            self._ensure_focus("System Settings")
            base_hints = ["Search", "search", "Search…", "Search settings", "Search in System Settings"]
            ok = self._focus_search_field("System Settings", base_hints, min_conf=0.45)
            if not ok:
                self.trace.warn("Could not confidently focus System Settings search.")
            if HAVE_PYAUTOGUI:
                pyautogui.write(section, interval=0.01); pyautogui.press("enter")
            else:
                keyboard.type(section); keyboard.press(Key.enter); keyboard.release(Key.enter)
            self._after_ui_change(wait=0.9, looking_for=[section])
            anno, elements, _ = self._rescan_active_window(looking_for=[section])
            cand, conf, used = pick_with_confidence(elements, [section], prefer_text=True)
            if cand:
                self._click_box(cand["bbox_xyxy_px"], rescan_after=True, label=cand.get("content",""))
                self.trace.outcome(f"Opened '{section}' in System Settings.")
                return True, None
            self.trace.warn(f"Section '{section}' not visible after search.")
            return False, None

        # Pre-scan for other actions
        looking_for = None
        if a == "select_size":
            size = str(p.get("size","")).strip().lower()
            looking_for = [f"{size}", f"{size} cm", f"{size}cm"]
            self.trace.think(f"Intent: choose size '{size}'. Strategy: open size control → search tokens inside dropdown/panel.")
        elif a == "add_to_bag":
            looking_for = synonyms_for_add_to_bag()
            self.trace.think(f"Intent: click Add‑to‑Bag. Strategy: OCR exact among {looking_for} → best candidate nearby.")
        elif a in ("click","double_click"):
            el = p.get("element") or p.get("target") or desc
            looking_for = [el]
            self.trace.think(f"Intent: interact with element '{el}'. Strategy: OCR exact → best textual candidate near anchor.")

        anno, elements, raw = self._rescan_active_window(looking_for=looking_for)
        self._save_artifacts(anno, elements, tag="preaction")

        def _select_and_click_by_hint(hint: str, clicks: int = 1) -> bool:
            # Ensure hint is valid
            if not hint:
                self.trace.warn("No hint provided for _select_and_click_by_hint.")
                return False

            # Rescan the active window to get the latest elements
            anno, elems, base = self._rescan_active_window(looking_for=[hint])
            self.trace.think(f"Looking for click target: '{hint}'")

            # Try exact match first
            cand = find_text_exact(elems, [hint], anchor=self.anchor_box)
            if cand:
                lbl = cand.get("content", "")
                self.trace.think(f"Found exact match for '{hint}' at {cand.get('bbox_xyxy_px')}")
                self._click_box(cand["bbox_xyxy_px"], clicks=clicks, rescan_after=True, label=lbl)
                return True

            # Fallback to fuzzy matching with best candidate
            cand, conf, used = pick_with_confidence(elems, [hint], anchor=self.anchor_box, prefer_text=True)
            if cand and conf >= 0.6:
                lbl = cand.get("content", "")
                self.trace.think(f"Found fuzzy match for '{hint}' ('{used}', conf={conf:.2f}) at {cand.get('bbox_xyxy_px')}")
                self._click_box(cand["bbox_xyxy_px"], clicks=clicks, rescan_after=True, label=lbl)
                return True

            # Use TargetLexicon to generate alternative labels
            domain = ""
            try:
                domain = urlparse(self._get_current_url()).netloc or "global"
            except Exception:
                pass
            lex = TargetLexicon().from_goal(hint, app=self.active_app or "", domain=domain)
            alt_labels = lex["targets"] + lex["exposure"]
            self.trace.think(f"No direct match; trying lexicon alternatives: {alt_labels}")
            cand, conf, used = pick_with_confidence(elems, alt_labels, anchor=self.anchor_box, prefer_text=True)
            if cand and conf >= 0.55:
                lbl = cand.get("content", "")
                self.trace.think(f"Lexicon match for '{hint}' ('{used}', conf={conf:.2f}) at {cand.get('bbox_xyxy_px')}")
                self._click_box(cand["bbox_xyxy_px"], clicks=clicks, rescan_after=True, label=lbl)
                return True

            # Try DOM click as a last resort
            if self._dom_click_text([hint] + alt_labels):
                self.trace.think(f"DOM click fallback succeeded for '{hint}' or alternatives.")
                self._after_ui_change(wait=WAIT_AFTER_CLICK)
                return True

            self.trace.warn(f"No match found for hint '{hint}' or alternatives in OCR/DOM results.")
            return False

        if a == "click":
            hint = p.get("target") or p.get("element") or desc
            if not hint:
                self.trace.warn("No target specified for click action, skipping.")
                return False, None
            try:
                domain = ""
                try:
                    domain = urlparse(self._get_current_url()).netloc or "global"
                except Exception:
                    pass
                prev_fp = self._last_fp
                ok = _select_and_click_by_hint(hint, clicks=1)
                if not ok:
                    self.trace.warn(f"Failed to click target '{hint}'.")
                    self.penalty_memory.bump(domain, hint)
                    return False, None
                # Check if the UI changed (fingerprint comparison)
                anno, elems, _ = self._rescan_active_window(looking_for=[hint])
                new_fp = self._last_fp
                if prev_fp and new_fp == prev_fp:
                    self.trace.think(f"Click on '{hint}' caused no UI change (fp={new_fp}).")
                    self.penalty_memory.bump(domain, hint)
                    adaptive_backtrack(self.trace, domain, self, hint, elems, anno)
                    return False, None
                # Success: update experience memory
                mem = ExperienceMemory(EXPERIENCE_PATH)
                mem.bump(domain, hint, +1.0)
                self._after_ui_change()
                self.trace.outcome(f"Clicked '{hint}' successfully.")
                return True, None
            except Exception as e:
                self.trace.warn(f"Error in click action for target '{hint}': {str(e)}")
                self.penalty_memory.bump(domain, hint)
                return False, None

        if a == "double_click":
            hint = p.get("element") or p.get("target") or desc
            ok = _select_and_click_by_hint(hint, clicks=2)
            self.trace.outcome("Double‑click succeeded." if ok else "Double‑click failed.")
            return ok, None

        if a == "type":
            text = p.get("text", "")
            hit_enter = bool(p.get("hit_enter", False))
            hint = p.get("element") or p.get("target")
            if self.active_app:
                self._ensure_focus(self.active_app)
            if hint:
                _select_and_click_by_hint(hint, clicks=1)
            else:
                time.sleep(0.15)
            if HAVE_PYAUTOGUI:
                pyautogui.write(text)
                if hit_enter or re.search(r"\b(search|query|url|site)\b", norm_text(desc)):
                    pyautogui.press("enter")
            else:
                keyboard.type(text)
                if hit_enter or re.search(r"\b(search|query|url|site)\b", norm_text(desc)):
                    keyboard.press(Key.enter); keyboard.release(Key.enter)
            self._after_ui_change()
            self.trace.outcome(f"Typed {len(text)} characters{' + Enter' if hit_enter else ''}.")
            return True, None

        if a == "hotkey":
            keys = [str(k) for k in (p.get("keys") or [])]
            self.trace.think(f"Intent: send hotkey {keys}.")
            try:
                if HAVE_PYAUTOGUI:
                    pyautogui.hotkey(*keys)
                else:
                    for k in keys:
                        kl = k.lower()
                        if kl in ("command","cmd"): keyboard.press(Key.cmd)
                        elif kl in ("control","ctrl"): keyboard.press(Key.ctrl)
                        elif kl == "shift": keyboard.press(Key.shift)
                        elif kl in ("option","alt"): keyboard.press(Key.alt)
                    for k in keys:
                        if len(k) == 1:
                            keyboard.press(k); keyboard.release(k)
                    for k in reversed(keys):
                        kl = k.lower()
                        if kl in ("command","cmd"): keyboard.release(Key.cmd)
                        elif kl in ("control","ctrl"): keyboard.release(Key.ctrl)
                        elif kl == "shift": keyboard.release(Key.shift)
                        elif kl in ("option","alt"): keyboard.release(Key.alt)
            except Exception as e:
                self.trace.warn(f"Hotkey send failed: {e}")
                return False, None
            self._after_ui_change()
            self.trace.outcome("Hotkey sent.")
            return True, None

        if a == "scroll":
            direction = str(p.get("direction", "down")).lower()
            self.trace.think(f"Intent: scroll {direction} within page.")
            if HAVE_PYAUTOGUI:
                pyautogui.scroll(-600 if direction == "down" else 600)
            else:
                dy = -20 if direction == "down" else 20
                for _ in range(30):
                    mouse.scroll(0, dy); time.sleep(0.01)
            self._after_ui_change()
            self.trace.outcome("Scroll issued.")
            return True, None

        if a == "read_text":
            el_hint = p.get("element", desc)
            cand = find_best_element(elements, el_hint, anchor=self.anchor_box, prefer_text=True)
            if cand:
                txt = cand.get("content","")
                self.trace.outcome(f"Read text='{txt[:64]}'…")
                return True, txt
            self.trace.warn("read_text: no match.")
            return False, None

        if a == "execute_command":
            cmd = p.get("command")
            if not cmd: return False, None
            self.trace.think(f"Intent: run shell command.")
            try:
                out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                self._after_ui_change()
                self.trace.outcome(f"Command exited {out.returncode}.")
                return True, out.stdout.strip()
            except Exception as e:
                self.trace.error(f"execute_command failed: {e}")
                return False, None

        if a == "notify":
            msg = p.get("message", "Done.")
            self._after_ui_change()
            self.trace.outcome(f"🔔 {msg}")
            print(f"\n🔔 {msg}\n")
            return True, None

        # --------- ADAPTIVE SIZE SELECTION (generic e‑commerce control) ---------
        if a == "select_size":
            size = str(p.get("size","")).strip().lower()
            if not size:
                self.trace.warn("No size provided.")
                return False, None

            size_ctrl_variants = ["select size", "choose size", "size", "size:"]
            digits = re.sub(r"[^0-9]", "", size)
            tokens = {size, f"{size} cm", f"{size}cm", f"{size} cm /", f"{size}/"}
            if digits:
                tokens.update({digits, f"{digits} cm", f"{digits}cm", f"{digits} /"})
                try:
                    cm = int(digits)
                    inch_approx = int(round(cm / 2.54))
                    for ins in {inch_approx, max(1, inch_approx-1), inch_approx+1}:
                        tokens.update({f"{ins} in", f'{ins}"', f"{digits} / {ins}", f"{ins} / {digits}"})
                except Exception:
                    pass
            size_choice_tokens = sorted(tokens)

            self.trace.think(f"Step 1: open size control. Variants={size_ctrl_variants}")
            ctrl = find_text_exact(elements, ["select size"]) or find_text_exact(elements, size_ctrl_variants)
            if ctrl:
                self.trace.observe(f"Clicking size control '{ctrl.get('content','')}' at {ctrl.get('bbox_xyxy_px')}")
                self._click_box(ctrl["bbox_xyxy_px"], clicks=1, rescan_after=True, looking_for=list(size_choice_tokens))
            else:
                maybe_icon = find_icon_near_text(elements, ["select size", "choose size", "size"])
                if maybe_icon:
                    self.trace.observe(f"Using icon near size control at {maybe_icon.get('bbox_xyxy_px')}")
                    self._click_box(maybe_icon["bbox_xyxy_px"], clicks=1, rescan_after=True, looking_for=list(size_choice_tokens))
                else:
                    self.trace.warn("Size control not found initially; clicking page center then rescan.")
                    if HAVE_PYAUTOGUI:
                        sw, sh = get_screen_size()
                        pyautogui.click(int(sw*0.5), int(sh*0.5))
                    self._after_ui_change(looking_for=list(size_choice_tokens))

            attempts = 8
            typed_filter_once = False

            for attempt in range(1, attempts+1):
                self.trace.think(f"Step 2: attempt {attempt}/{attempts} – scan for {list(size_choice_tokens)}")
                _, elems2 = self._after_ui_change(wait=0.25, looking_for=list(size_choice_tokens))
                found = find_text_exact(elems2, list(size_choice_tokens), anchor=self.anchor_box) or \
                        find_text_exact(elems2, list(size_choice_tokens), anchor=None)
                if found:
                    self._click_box(found["bbox_xyxy_px"], clicks=1, rescan_after=True, label=found.get("content",""))
                    self.trace.outcome(f"Size option chosen: '{found.get('content','')}'.")
                    _, _ = self._after_ui_change(wait=0.35)
                    return True, None

                self.trace.think("Size not visible yet → try local scroll within dropdown/panel.")
                try:
                    if HAVE_PYAUTOGUI: pyautogui.scroll(-220)
                    else:
                        for _ in range(10): mouse.scroll(0, -3); time.sleep(0.005)
                except Exception:
                    pass

                if digits and not typed_filter_once:
                    self.trace.think(f"Fallback: type-ahead filter by typing '{digits}'.")
                    try:
                        if HAVE_PYAUTOGUI: pyautogui.typewrite(digits)
                        else: keyboard.type(digits)
                    except Exception:
                        pass
                    typed_filter_once = True

            self.trace.warn("Size not found after scans; triggering chooser via 'Add To Bag'.")
            add_labels = synonyms_for_add_to_bag()
            push = find_text_exact(elements, add_labels) or find_best_element(elements, "add to bag", prefer_text=True)
            if push:
                self._click_box(push["bbox_xyxy_px"], rescan_after=True, label=push.get("content",""))
                _, elems3 = self._after_ui_change(wait=0.6, looking_for=list(size_choice_tokens))
                found = find_text_exact(elems3, list(size_choice_tokens), anchor=None)
                if found:
                    self._click_box(found["bbox_xyxy_px"], rescan_after=True, label=found.get("content",""))
                    self.trace.outcome(f"Size option chosen after prompt: '{found.get('content','')}'.")
                    return True, None

            self.trace.warn(f"Size '{size}' not found after opening dropdown/panel and searching.")
            self.trace.next("Manual scroll may be needed or the size is out of stock.")
            return False, None

        if a == "add_to_bag":
            add_labels = synonyms_for_add_to_bag()
            self.trace.think(f"Looking for add‑to‑bag among {add_labels}")
            # Gather all candidates (not just best)
            cands = [e for e in elements if e.get("type") == "text" and any(norm_text(e.get("content", "")) == norm_text(lbl) or norm_text(lbl) in norm_text(e.get("content", "")) for lbl in add_labels)]
            # Prefer bottom-most, right-most (common UX for add-to-bag)
            if cands:
                cands.sort(key=lambda e: (e["bbox_xyxy_px"][1], e["bbox_xyxy_px"][0]), reverse=True)
                best = cands[0]
                self._click_box(best["bbox_xyxy_px"], clicks=1, rescan_after=True, label=best.get("content",""))
                self.trace.outcome("Add‑to‑bag clicked via bottom/right heuristic.")
                return True, None
            # Fallback: use anchor and best candidate
            cand = find_best_element(elements, "add to bag", anchor=None, prefer_text=True)
            if cand:
                self._click_box(cand["bbox_xyxy_px"], clicks=1, rescan_after=True, label=cand.get("content",""))
                self.trace.outcome("Add‑to‑bag clicked via fallback.")
                return True, None
            # Last-resort: DOM click
            if self._dom_click_text(add_labels):
                self.trace.outcome("Add‑to‑bag DOM click fallback executed.")
                return True, None
            self.trace.warn("Add‑to‑bag control not found.")
            return False, None
        

        # ------------------- Generic reach_target navigation -------------------
        if a == "reach_target":
            subgoal = (p.get("goal") or p.get("target") or desc).strip()
            max_iters = int(p.get("max_iters", self.reach_max_iters))
            domain = ""
            try:
                href_now = self._get_current_url()
                domain = urlparse(href_now).netloc
            except Exception:
                try:
                    if self.driver:
                        domain = urlparse(self.driver.current_url).netloc
                except Exception:
                    domain = ""
            lex = TargetLexicon().from_goal(subgoal, app=self.active_app or "", domain=domain)
            targets = lex["targets"]
            exposure = lex["exposure"]
            success_text = lex["success_text"]
            url_hints = [norm_text(u) for u in lex["url_substrings"]]
            mem = ExperienceMemory(EXPERIENCE_PATH)
            fail_count = 0

            # Initialize fingerprint
            anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
            cur_fp = self._last_fp  # Use fingerprint from _rescan_active_window
            prev_fp = None

            for it in range(1, max_iters + 2):
                # Success check (URL/text)
                current_url = (self._get_current_url() or "").lower()
                ocr_texts = [e["content"].lower() for e in elems if e["type"] == "text"]
                url_ok = any(tok in current_url for tok in url_hints)
                text_ok = any(st.lower() in t for t in ocr_texts for st in success_text)
                if url_ok and text_ok:
                    self.trace.outcome("Verified goal via URL + OCR → stopping early.")
                    self.penalty_memory.reset(domain)
                    self._save_artifacts(anno, elems, tag="success")
                    return True, None

                try:
                    href = (self._get_current_url() or "").lower()
                    if any(h in href for h in url_hints if h):
                        self.trace.outcome("reach_target: URL indicates success.")
                        self.penalty_memory.reset(domain)
                        self._save_artifacts(anno, elems, tag="success")
                        return True, None
                except Exception:
                    pass
                visible_texts = [norm_text(e.get("content","")) for e in elems if e.get("type")=="text"]
                success_on_screen = [s for s in success_text if norm_text(s) in visible_texts]
                still_interactable = False

                # If success text is present, check if it’s clickable
                for s in success_on_screen:
                    for e in elems:
                        if norm_text(e.get("content","")) == norm_text(s) and e.get("type") in ("text","icon"):
                            if e.get("type") == "text" or (e.get("type") == "icon" and e.get("interactive", True)):
                                self.trace.think(f"Success text '{s}' is still interactable. Will click and continue.")
                                self._click_box(e["bbox_xyxy_px"], rescan_after=True, label=s)
                                still_interactable = True
                                break
                    if still_interactable:
                        break

                if success_on_screen and not still_interactable:
                    self.trace.outcome("reach_target: success text is visible and no further action required.")
                    self.penalty_memory.reset(domain)
                    self._save_artifacts(anno, elems, tag="success")
                    return True, None

                # Penalized candidate selection
                def ranked_pick_with_penalty(corpus):
                    best_e, best_val = None, -1e9
                    for tok in corpus:
                        cand, conf, _ = pick_with_confidence(elems, [tok], anchor=self.anchor_box, prefer_text=True)
                        if not cand: continue
                        val = conf + 0.18 * mem.score(domain, cand.get("content","")) - 0.5 * self.penalty_memory.penalty(domain, cand.get("content",""))
                        if val > best_val:
                            best_val, best_e = val, cand
                    return best_e

                # 1) Try target click
                hit = ranked_pick_with_penalty(targets)
                if hit:
                    lbl = hit.get("content", "")
                    self._click_box(hit["bbox_xyxy_px"], rescan_after=True, label=lbl)
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    new_fp = self._last_fp
                    if prev_fp and new_fp == prev_fp:
                        if self._try_dismiss_banners(elems):
                            self.trace.outcome("Clicked consent banner to clear blocking overlay.")
                            prev_fp = None
                            continue
                        self.penalty_memory.bump(domain, lbl)
                        fail_count += 1
                        self.trace.think(f"Target '{lbl}' failed (no state change). Penalty incremented.")
                        if fail_count > 2:
                            adaptive_backtrack(self.trace, domain, self, subgoal, elems, base)
                    else:
                        mem.bump(domain, lbl, +1.0)
                        fail_count = 0
                    prev_fp = new_fp
                    cur_fp = new_fp
                    continue

                # 2) Try exposure click
                exp = ranked_pick_with_penalty(exposure)
                if exp:
                    lbl = exp.get("content","")
                    self.trace.observe(f"reach_target: clicking exposure '{lbl}' @ {exp.get('bbox_xyxy_px')}")
                    self._hover_box(exp["bbox_xyxy_px"], pause=0.5)
                    self._click_box(exp["bbox_xyxy_px"], rescan_after=True, label=lbl)
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    new_fp = self._last_fp
                    if prev_fp and new_fp == prev_fp:
                        self.penalty_memory.bump(domain, lbl)
                        fail_count += 1
                        self.trace.think(f"Exposure '{lbl}' failed (no state change). Penalty incremented.")
                        if fail_count > 2:
                            adaptive_backtrack(self.trace, domain, self, subgoal, elems, base)
                    else:
                        mem.bump(domain, lbl, +0.5)
                        fail_count = 0
                    prev_fp = new_fp
                    cur_fp = new_fp
                    # Quick rescan for targets
                    _, elems2, _ = self._rescan_active_window(looking_for=targets)
                    cand2, conf2, _ = pick_with_confidence(elems2, targets, anchor=None, prefer_text=True)
                    if cand2:
                        lbl2 = cand2.get("content","")
                        self._click_box(cand2["bbox_xyxy_px"], rescan_after=True, label=lbl2)
                        anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                        cur_fp = self._last_fp
                        prev_fp = cur_fp
                        continue

                # 3) LLM-guided micro-step
                visible = [e.get("content","") for e in elems if e.get("type")=="text"]
                elements_for_llm = [
                    {
                        "type": e.get("type"),
                        "text": e.get("content", ""),
                        "box": e.get("bbox_xyxy_px"),
                    } for e in elems if e.get("type") in {"text", "icon"}
                ]
                prop = self.reflexor.propose_consensus(subgoal, visible, k=self.deliberate)
                act, tgt = prop.get("action","noop"), prop.get("target","")
                if act == "click" and tgt:
                    c, conf, _ = pick_with_confidence(elems, [tgt], anchor=self.anchor_box, prefer_text=True)
                    if c:
                        lbl = c.get("content","")
                        self._click_box(c["bbox_xyxy_px"], rescan_after=True, label=lbl)
                        anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                        new_fp = self._last_fp
                        if prev_fp and new_fp == prev_fp:
                            self.penalty_memory.bump(domain, lbl)
                            fail_count += 1
                            self.trace.think(f"LLM click '{lbl}' failed (no state change). Penalty incremented.")
                            if fail_count > 2:
                                adaptive_backtrack(self.trace, domain, self, subgoal, elems, base)
                        else:
                            fail_count = 0
                        prev_fp = new_fp
                        cur_fp = new_fp
                        continue
                elif act == "typeenter" and tgt:
                    self._ensure_focus(self.active_app or "Google Chrome")
                    if HAVE_PYAUTOGUI:
                        pyautogui.typewrite(tgt, interval=0.01); pyautogui.press("enter")
                    else:
                        keyboard.type(tgt); keyboard.press(Key.enter); keyboard.release(Key.enter)
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    cur_fp = self._last_fp
                    prev_fp = cur_fp
                    continue
                elif act == "scroll":
                    if HAVE_PYAUTOGUI: pyautogui.scroll(-600)
                    else:
                        for _ in range(30): mouse.scroll(0, -3); time.sleep(0.005)
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    cur_fp = self._last_fp
                    prev_fp = cur_fp
                    continue

                # 4) DOM click fallback
                if self._dom_click_text(targets + success_text):
                    self._after_ui_change(wait=0.8)
                    self.trace.info("reach_target: DOM click fallback executed.")
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    cur_fp = self._last_fp
                    prev_fp = cur_fp
                    continue

                # 5) LLM re-plan if too many failures
                if it == max_iters or fail_count > 3:
                    self.trace.think("Target not reached after all attempts, invoking dynamic replan with LLM.")
                    dynamic_replan_with_llm(self.trace, domain, self, subgoal, elems, base)
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    cur_fp = self._last_fp
                    prev_fp = cur_fp
                    continue

                # Anti-loop: if fingerprint unchanged, jitter with small scroll
                if prev_fp and cur_fp == prev_fp:
                    if HAVE_PYAUTOGUI: pyautogui.scroll(-300)
                    else:
                        for _ in range(15): mouse.scroll(0, -3); time.sleep(0.005)
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    cur_fp = self._last_fp
                    prev_fp = cur_fp

            # Adaptive fallback section (unchanged from original)
            tried_back = tried_forward = tried_scroll_up = tried_url_paste = False
            if self.active_app and "chrome" in self.active_app.lower():
                self.trace.think("Adaptive fallback: Trying browser back navigation (Cmd + [)")
                try:
                    if HAVE_PYAUTOGUI:
                        pyautogui.hotkey("command", "[")
                    else:
                        keyboard.press(Key.cmd); keyboard.press("["); keyboard.release("["); keyboard.release(Key.cmd)
                    time.sleep(1)
                    _, elems, _ = self._after_ui_change(wait=1.5)
                    tried_back = True
                    href_now = (self._get_current_url() or "").lower()
                    if any(h in href_now for h in url_hints if h) or any(norm_text(s) in [norm_text(e.get("content","")) for e in elems if e.get("type")=="text"] for s in success_text):
                        self.trace.info("Browser back navigation succeeded.")
                        self._save_artifacts(anno, elems, tag="success")
                        return True, None
                except Exception:
                    self.trace.warn("Browser back navigation failed.")

                if tried_back:
                    self.trace.think("Adaptive fallback: Trying browser forward navigation (Cmd + ])")
                    try:
                        if HAVE_PYAUTOGUI:
                            pyautogui.hotkey("command", "]")
                        else:
                            keyboard.press(Key.cmd); keyboard.press("]"); keyboard.release("]"); keyboard.release(Key.cmd)
                        time.sleep(1)
                        _, elems, _ = self._after_ui_change(wait=1.5)
                        tried_forward = True
                        href_now = (self._get_current_url() or "").lower()
                        if any(h in href_now for h in url_hints if h) or any(norm_text(s) in [norm_text(e.get("content","")) for e in elems if e.get("type")=="text"] for s in success_text):
                            self.trace.info("Browser forward navigation succeeded.")
                            self._save_artifacts(anno, elems, tag="success")
                            return True, None
                    except Exception:
                        self.trace.warn("Browser forward navigation failed.")

                self.trace.think("Adaptive fallback: Try scrolling up and DOM click")
                for attempt in range(2):
                    if HAVE_PYAUTOGUI:
                        pyautogui.scroll(1000)
                    else:
                        for _ in range(50): mouse.scroll(0, 3); time.sleep(0.005)
                    self._after_ui_change()
                    if self._dom_click_text(targets + success_text):
                        self.trace.info("DOM click after scroll up succeeded.")
                        anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                        self._save_artifacts(anno, elems, tag="success")
                        return True, None
                    tried_scroll_up = True

                self.trace.think("Final fallback: scroll down and DOM click (2x)")
                for attempt in range(2):
                    if HAVE_PYAUTOGUI:
                        pyautogui.scroll(-1000)
                    else:
                        for _ in range(50): mouse.scroll(0, -3); time.sleep(0.005)
                    self._after_ui_change()
                    if self._dom_click_text(targets + success_text):
                        self.trace.info("DOM click after scroll down succeeded.")
                        anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                        self._save_artifacts(anno, elems, tag="success")
                        return True, None

                if self.active_app and "chrome" in self.active_app.lower() and self.target_url_sub and not tried_url_paste:
                    self.trace.think("Adaptive fallback: Repasting URL in omnibox.")
                    self._ensure_focus("Google Chrome")
                    if HAVE_PYAUTOGUI:
                        pyautogui.hotkey("command", "l")
                        pyautogui.typewrite(self.target_url_sub)
                        pyautogui.press("enter")
                    else:
                        keyboard.press(Key.cmd); keyboard.press('l'); keyboard.release('l'); keyboard.release(Key.cmd)
                        keyboard.type(self.target_url_sub); keyboard.press(Key.enter); keyboard.release(Key.enter)
                    self._after_ui_change(wait=WAIT_AFTER_NAV)
                    if self._dom_click_text(targets + success_text):
                        self.trace.info("DOM click after repasting URL succeeded.")
                        anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                        self._save_artifacts(anno, elems, tag="success")
                        return True, None
                    tried_url_paste = True

                if self._dom_click_text(targets + success_text):
                    self.trace.info("Final fallback: DOM click executed.")
                    anno, elems, base = self._rescan_active_window(looking_for=targets + exposure)
                    self._save_artifacts(anno, elems, tag="success")
                    return True, None

            self.trace.warn("reach_target: gave up before confirming success after maximum retries.")
            return False, None

    def close(self, really_close_browser: bool = False):
        if really_close_browser:
            try:
                if self.driver: self.driver.quit()
            except Exception:
                pass

def dynamic_goal_postcheck(goal, current_url, ocr_text, openai_key=OPENAI_KEY, model=OPENAI_MODEL_TEXT):
    """
    Uses OpenAI to check if the user goal is accomplished.
    Returns (achieved:bool, explanation:str).
    """
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    prompt = (
        f"You are an AI agent. The user gave this goal:\n\n"
        f"GOAL: {goal}\n\n"
        f"The current URL is: {current_url}\n\n"
        f"The visible text on the screen is:\n{ocr_text[:600]}\n\n"
        f"Based on the above, has the goal been fully accomplished? "
        f"Reply ONLY as JSON: {{\"achieved\":true/false,\"reason\":\"...\"}}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an accuracy-checking agent."},
                  {"role": "user", "content": prompt}],
        max_tokens=130,
        response_format={"type":"json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return bool(data.get("achieved")), str(data.get("reason",""))

#######################################################################
#  Main CLI
#######################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive GUI Agent (window‑targeted, think‑aloud)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless for browser steps")
    parser.add_argument("--image", help="Debug: run on provided screenshot file instead of live screen")
    parser.add_argument("--save", action="store_true", help="Save annotated images and JSON extractions")
    parser.add_argument("--fast", action="store_true", help="(Optional) Minor speed tweaks")
    parser.add_argument("--close-browser", action="store_true", help="Close Selenium Chrome at the end (default: leave open)")

    ### [ADDED] flags to reuse same window / paste URL / attach
    parser.add_argument("--reuse-chrome", action="store_true",
                        help="Reuse an existing Chrome window/tab when possible; attempt debugger attach instead of new window.")
    parser.add_argument("--paste-link", action="store_true",
                        help="When navigating to a URL, paste it into the omnibox in the current tab instead of Selenium get().")
    parser.add_argument("--debug-address", default=None,
                        help="Debugger address for Chrome attach (e.g., 127.0.0.1:9222).")

    # NEW thinking knobs
    parser.add_argument("--deliberate", type=int, default=LLM_DELIBERATE,
        help="Self-consistency votes for Reflexor (1=no extra thinking)")
    parser.add_argument("--planner-n", type=int, default=PLANNER_N,
        help="How many plan candidates to sample and choose from")
    parser.add_argument("--max-iters", type=int, default=REACH_MAX_ITERS_DEFAULT,
        help="Default micro-steps budget for reach_target")

    args = parser.parse_args()

    if not OPENAI_KEY or OPENAI_KEY.startswith("REPLACE_") or OPENAI_KEY.strip() == "":
        log.warning("OPENAI_API_KEY not set. Set env var or edit OPENAI_KEY in script.")

    ts = int(time.time())
    run_dir = os.path.join(OUT_DIR, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    trace = ChatTrace(run_dir, enabled=True)
    trace.banner("GUI Agent – Window‑Targeted (Think‑Aloud)")

    if HAVE_PYAUTOGUI:
        trace.info("pyautogui is active. Ensure Terminal/Python has Accessibility & Screen Recording permissions.")
    else:
        trace.info("Using pynput+mss. Ensure Accessibility & Screen Recording permissions are granted.")

    goal = input("Enter your goal: ").strip()
    trace.banner(f"Goal: {goal}")

    parser_model = OmniParser(YOLO_WEIGHTS)
    planner = Planner()
    planner.planner_n = args.planner_n  # expose knob
    clarifs = planner.clarifications(goal)
    trace.info(f"Clarifications collected: {json.dumps(clarifs, ensure_ascii=False)}")
    steps = planner.plan(goal, clarifs)
    trace.plan(steps)

    execu = Executor(parser_model, trace, deliberate=args.deliberate, reach_max_iters=args.max_iters)

    ### [ADDED] wire runtime toggles from CLI/env
    execu.reuse_chrome = bool(args.reuse_chrome or REUSE_CHROME_ENV)
    execu.paste_link_same_window = bool(args.paste_link or PASTE_LINK_ENV or execu.reuse_chrome)
    execu.debugger_address = (args.debug_address or DEBUG_ADDRESS_ENV)

    def final_goal_postcheck_and_trigger(goal, parser_model, execu, trace):
        # As a last resort, reach_target again
        trace.warn("Post-check: Could not guarantee goal execution via direct/DOM click. Attempting reach_target.")
        dummy = Step(description="Final reach_target retry", action="reach_target", params={"goal": goal, "max_iters": 6})
        ok, _ = execu.execute(dummy, headless=False)
        # Re-scan UI after reach_target to avoid double firing DOM clicks
        _, elements, _ = execu._rescan_active_window(looking_for=[goal])
        goal_tokens = [goal] + synonyms_for_add_to_bag()
        goal_tokens = list({norm_text(t): t for t in goal_tokens if t.strip()}.values())  # dedup
        cand, conf, used = pick_with_confidence(elements, goal_tokens, anchor=None, prefer_text=True)
        if cand and conf > 0.8:
            trace.action(f"(Post-reach_target) Final goal element found ('{used}', conf={conf:.2f}) – clicking.")
            execu._click_box(cand["bbox_xyxy_px"], clicks=1, rescan_after=True, label=used)
            return True
        # DOM click only if not already clicked by reach_target
        dom_ok = execu._dom_click_text(goal_tokens)
        if dom_ok:
            trace.action("(Post-reach_target) Final DOM click fallback succeeded.")
            return True
        return ok

    for idx, step in enumerate(steps, start=1):
        step_dir = os.path.join(run_dir, f"step_{idx:02d}")
        os.makedirs(step_dir, exist_ok=True)

        if args.image and os.path.exists(args.image):
            img = Image.open(args.image).convert("RGB")
            anno, elems = parser_model.parse(img)
            if args.save:
                anno.save(os.path.join(step_dir, f"annotated_debug.png"))
                with open(os.path.join(step_dir, f"elements_debug.json"), "w") as f:
                    json.dump({"elements": elems}, f, indent=2)
            continue

        result = execu.execute(step, headless=args.headless, save_dir=(step_dir if args.save else None))
        if result is None:
            ok = False
        else:
            ok = result[0]
        if not ok:
            trace.error(f"Step failed: {step.description}")
            break
        time.sleep(0.6 if not args.fast else 0.25)

        # --- FINAL GOAL EXECUTION GUARANTEE ---

    # --- Call the post-check after running the plan ---
    final_goal_postcheck_and_trigger(goal, parser_model, execu, trace)
    """
    After executing all planned steps, this function attempts to guarantee the user's goal
    is truly executed. It follows a fallback chain:
      1. Tries to find and click the goal element using OCR confidence.
      2. Tries DOM click fallback (JS click in browser).
      3. As a last resort, re-invokes reach_target on the goal.
    This reduces the chance of ending the session with the final step unexecuted.
    """
        # -- DYNAMIC GOAL POSTCHECK (OpenAI LLM-based) --
    try:
        _, elements, _ = execu._rescan_active_window(looking_for=[goal])
        current_url = execu._get_current_url()
        ocr_text = "\n".join([e.get("content", "") for e in elements if e.get("type") == "text"])
        achieved, reason = dynamic_goal_postcheck(goal, current_url, ocr_text)
        # Boxed pretty print for clarity
        def print_boxed(msg, success=True):
            lines = msg.strip().splitlines()
            maxlen = max(len(line) for line in lines)
            banner = "✅ FINAL LLM POST-CHECK: Goal is accomplished" if success else "❌ FINAL LLM POST-CHECK: Goal NOT accomplished"
            box_top = "╔" + "═"*(maxlen+2) + "╗"
            box_bot = "╚" + "═"*(maxlen+2) + "╝"
            print("\n" + banner)
            print(box_top)
            for line in lines:
                print("║ " + line.ljust(maxlen) + " ║")
            print(box_bot + "\n")
        if achieved:
            trace.banner("✅ FINAL LLM POST-CHECK: Goal is accomplished")
            print_boxed(f"LLM says: {reason}", success=True)
        else:
            trace.banner("❌ FINAL LLM POST-CHECK: Goal NOT accomplished")
            print_boxed(f"LLM says: {reason}", success=False)
    except Exception as e:
        trace.warn(f"LLM post-check failed: {e}")


    execu.close(really_close_browser=args.close_browser)
    trace.banner("✅ Finished.")


if __name__ == "__main__":
    main()

# End of Code
# Completed by Cayden Auyang on Jul 31, 2025
