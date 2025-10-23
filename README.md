# Screen-Parsing-GUI-Agent-

Screen Parsing GUI Agent Functionality, Possibilities, Limitations and Code

1. What This Code Is & What It Does
This is a macOS-based, window-targeted GUI automation agent that:
“Screen Parsing” - Locates on-screen text and icons via a YOLO-based detector and PaddleOCR.


Plans high-level multi-step goals (open application, open website, extract information, play song, add to cart, type, create document, etc.) with an LLM (OpenAI).


Executes generic actions (click, type, scroll, reach_target) by matching OCR’d labels or icon boxes.


Falls back to DOM clicks via Selenium/JavaScript if vision clicks don’t register.


Backtracks and re-plans when stuck—presses Esc/back, asks the LLM for an alternate strategy.


Remembers what worked (and what failed) per domain with lightweight “experience” and “penalty” memories.


Verifies success by URL/text heuristics and even does a final LLM-based post-check.



2. Core Technologies
Computer Vision & OCR


Ultralytics YOLOv8 for generic icon hitboxes


PaddleOCR for reading on-screen text


LLM Planning & Reflex


OpenAI Chat API (configurable models) for both macro-planning and micro-step suggestions


macOS Automation


AppleScript (osascript) for focusing or querying apps/tabs


pyautogui / pynput + mss for mouse/keyboard and screenshots


Browser Automation


Selenium WebDriver (ChromeDriver) + JavaScript DOM clicks


State Management


SHA1 fingerprints to detect UI change and avoid loops


Tiny JSON-backed memory for successes and failures



3. Improvements Over GitHub OmniTool
Capability
OmniTool
This Agent
Goal-driven planning
Basic one-step LLM proposals (no multi-step plan scoring)
LLM-based planner + reflexor that adapts on the fly with multi-sample voting
Vision-first interaction
✅ Screenshot-only parsing (custom icon detector + captioning)
✅ YOLO icon detection + PaddleOCR → unified text + icon UI parsing
Cookie-banner handling
❌ None
✅ Proactive “Accept all” via vision match
DOM fallback
❌ Relies purely on vision
✅ Optional Selenium/JS click when OCR misses
Loop avoidance
❌ No mechanism
✅ Fingerprint comparison + penalty memory
Backtracking
❌ No built-in recovery
✅ Esc/back + dynamic LLM re-planning
Experience memory
❌ Stateless
✅ Bandit-style scoring per domain → prefers labels that worked before
Runtime toggles
❌ Fixed behavior
✅ CLI flags for reusing Chrome tabs, pasting links in-place, remote-debug


Description
Building on OmniTool’s integration of OmniParser V2—which pairs a YOLOv8-based detection module with Florence-2 captioning to identify and semantically label UI elements—the tool substitutes PaddleOCR for lightweight, on-device text extraction alongside the same YOLOv8 icon detector. It introduces a two-tiered OpenAI LLM architecture (GPT-4 for high-level planning and a lower-latency reflex model for micro-step actions) wrapped in a feedback-driven memory system that rewards successful UI interactions and penalizes failed attempts. Adaptive backtracking (Esc/back navigation and LLM-guided replanning), proactive cookie-banner dismissal, and a generic, LLM-driven reach_target algorithm allow it to navigate arbitrary macOS apps without site-specific hacks. Deep AppleScript integration for window targeting and optional Chrome remote-debug reuse replaces the need for Dockerized Windows VMs, yielding a leaner, more responsive local agent. These enhancements deliver greater flexibility and autonomy—while acknowledging the inherent OCR/LLM accuracy trade-offs and occasional inconsistencies.




4. Key Additions & Smart Logic Flows
PenaltyMemory & ExperienceMemory to bias toward successful labels and avoid repeats of failures


Dynamic “reach_target”: builds a lexicon of targets, exposure controls, success_text and URL hints via LLM


LLM-driven micro-steps (Reflexor) that vote on next clicks when generic heuristics stall


Adaptive backtracking: on no UI change → press Esc/back and ask LLM for an alternative click


Final LLM post-check: verifies end state (“did we really add-to-cart?”) and prints boxed feedback


Flexible Chrome handling: can attach to an existing window via remote debugging or open new one




5. Limitations & Known Issues
OCR/Detection errors: small text, low contrast or custom fonts may be missed or mis-read


LLM inconsistency: occasional hallucinations or overly generic token suggestions


Site variability: some sites use non-standard controls, dynamic loading, or require captcha/login


Timing & animations: hard to tune waits for every UI; sometimes clicks “through” modals or too early


Not bullet-proof: still can get stuck in loops or miss the exact “Buy” vs. “Customize” step without clarification


Permissions: requires macOS Accessibility & Screen Recording permissions, plus Chrome remote-debug flags



6. Why This Matters Today
Bridging vision-based UI parsing (YOLO + OCR) with LLM reasoning is a promising step toward truly app-agnostic automation. Rather than brittle, per-site scripts, this agent:
Thinks: plans and replans with context


Sees: extracts all visible labels and icons


Acts: chooses generically via OCR, DOM or LLM affordance inference


Learns: adapts to domains over time


It’s not production-ready—accuracy and consistency vary—but it demonstrates how combining CV and LLMs can move the needle on autonomous GUI agents in 2025.

7. Quick Dependencies & Setup
Python 3.8+


pip install:
 pip install openai ultralytics torch paddlepaddle paddleocr numpy pillow mss pynput pyautogui selenium
Chrome & ChromeDriver (matching versions)


Environment vars:


OPENAI_API_KEY
optional: YOLO_WEI
GHTS, OPENAI_MODEL_PLANNER, etc.

Recommended Tools
Editor: Sublime Text (or any code editor)


Runner: macOS Terminal (with Accessibility & Screen Recording permissions granted)


This agent represents a next-generation prototype—not a polished product—but it highlights the potential of melding computer vision with generative AI for flexible, adaptive UI automation.
