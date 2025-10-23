# 🧠 Screen Parsing GUI Agent

A macOS-based, window-targeted **GUI automation agent** that combines **computer vision (YOLO + OCR)** with **LLM reasoning** to perform autonomous, goal-driven interactions across arbitrary apps and websites.

---

## 🚀 1. What This Code Is & What It Does

This project implements a **vision + LLM-based automation agent** that can:

- **See:** Locate on-screen text and icons using YOLOv8 (for icon detection) and PaddleOCR (for text extraction).  
- **Think:** Use an OpenAI LLM (GPT-based) to plan and replan multi-step tasks like:  
  * opening applications  
  * visiting websites  
  * extracting data  
  * typing or clicking buttons  
  * adding to cart  
  * creating documents  
- **Act:** Execute generic actions (`click`, `type`, `scroll`, `reach_target`) by matching OCR’d labels or detected icons.  
- **Recover:** Backtrack (Esc/back) and replan when stuck.  
- **Learn:** Store “experience” and “penalty” memories to remember what worked and what didn’t.  
- **Verify:** Confirm successful outcomes with URL/text heuristics and a final LLM-based post-check.  

---

## 🧩 2. Core Technologies

| Area | Technologies Used |
|------|--------------------|
| **Computer Vision & OCR** | Ultralytics YOLOv8 (icon detection), PaddleOCR (text recognition) |
| **LLM Planning & Reflex** | OpenAI Chat API (configurable GPT models for macro/micro planning) |
| **macOS Automation** | AppleScript (focus/query apps/tabs), `pyautogui` / `pynput` + `mss` |
| **Browser Automation** | Selenium WebDriver (ChromeDriver) + JavaScript DOM click fallback |
| **State Management** | SHA1 UI fingerprints + JSON-based memory for successes/failures |

---

## ⚙️ 3. Improvements Over OmniTool

| Capability | OmniTool | This Agent |
|-------------|-----------|-------------|
| Goal-driven planning | Basic one-step LLM prompts | Multi-step LLM planner + adaptive reflexor |
| Vision-first interaction | Screenshot parsing only | YOLOv8 + PaddleOCR unified parsing |
| Cookie-banner handling | ❌ None | ✅ Auto “Accept All” via vision match |
| DOM fallback | ❌ Vision-only | ✅ Selenium/JS click fallback |
| Loop avoidance | ❌ None | ✅ Fingerprint + penalty memory |
| Backtracking | ❌ No recovery | ✅ Esc/back + dynamic replan |
| Experience memory | ❌ Stateless | ✅ Bandit-style per-domain scoring |
| Runtime toggles | ❌ Fixed | ✅ CLI flags for Chrome reuse, remote debug, etc. |

**Description:**  
Building on OmniTool’s OmniParser V2 (YOLOv8 + Florence-2), this agent replaces Florence captioning with PaddleOCR for fast, on-device text extraction.  
It features a **two-tier LLM system** — GPT-4 for high-level plans and a low-latency reflex model for micro-actions — all under a feedback-driven memory system.  
Adaptive backtracking, proactive cookie handling, Chrome reuse, and AppleScript-based window targeting make it a leaner, macOS-native automation framework.

---

## 🔁 4. Key Additions & Smart Logic Flows

- **PenaltyMemory & ExperienceMemory** → bias toward successful labels; avoid known failures.  
- **Dynamic `reach_target`** → LLM builds target lexicon, exposure controls, and success/URL hints.  
- **Reflexor** → micro-step LLMs vote on next action when heuristics stall.  
- **Adaptive Backtracking** → detect no-UI-change → trigger Esc/back + replan.  
- **Final LLM Post-Check** → confirms task success (“Did we really add to cart?”).  
- **Chrome Handling** → attach to existing session or open a new tab via remote debug.

---

## ⚠️ 5. Limitations & Known Issues

- **OCR errors** on small/low-contrast/custom fonts.  
- **LLM inconsistency** — may hallucinate or generalize excessively.  
- **Site variability** — custom controls, CAPTCHAs, dynamic loaders.  
- **Timing issues** — can click too early or through modals.  
- **Loop risks** — fingerprinting mitigates but doesn’t eliminate stuck loops.  
- **macOS permissions** — requires Accessibility & Screen Recording access + Chrome debug flags.

---

## 🌍 6. Why This Matters

This project bridges **vision-based UI parsing** and **LLM reasoning**, a crucial step toward **app-agnostic automation**.

The agent:
- **Thinks:** plans, evaluates, and replans.  
- **Sees:** reads any on-screen text or icon.  
- **Acts:** performs generic UI actions via OCR, DOM, or inferred affordances.  
- **Learns:** improves over time with success/failure memory.

> ⚡ It’s still experimental — accuracy and consistency vary — but it demonstrates the future of flexible, autonomous GUI agents.

---

## 🧱 7. Quick Setup & Dependencies

### Requirements
- macOS  
- Python **3.8+**  
- Chrome + matching **ChromeDriver**

### Install
```bash
pip install openai ultralytics torch paddlepaddle paddleocr numpy pillow mss pynput pyautogui selenium
