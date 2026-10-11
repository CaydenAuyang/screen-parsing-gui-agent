# screen-parsing-gui-agent

A macOS agent that operates a computer by looking at the screen. You type a goal, and it reads the screen with computer vision and OCR, uses a language model to plan the steps, and then clicks, types and scrolls across apps and websites to carry them out.

I built it in summer 2025 during my internship at Wisers Information in Hong Kong, after my mentor pointed out that no reliable tool on the market could yet operate a computer on its own.

## Built on

This project builds on other people's open-source work:

- **[Microsoft OmniParser](https://github.com/microsoft/OmniParser) (V2).** The agent uses OmniParser's icon detection model (`icon_detect`, a fine-tuned YOLOv8) to find buttons and icons on screen. Its overall design follows Microsoft's OmniParser and OmniTool: turn a screenshot into a list of labeled elements, then let a language model decide what to do next. The model weights are not included in this repo. They are licensed under AGPL-3.0 by Microsoft.
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** reads the text on screen. I used it in place of OmniParser's Florence-2 captioning model.
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** runs the detection model.
- **OpenAI models** (GPT-4.1 by default) plan the steps and choose the next action.
- **Selenium, pynput, pyautogui and mss** control Chrome, the mouse and keyboard, and take screenshots.

## How it works

1. **Plan.** The goal goes to the language model, which returns a list of steps: open an app, go to a URL, search, click, type, scroll or read text.
2. **See.** For each step the agent takes a screenshot of the front window, finds icons with the detection model and reads text with OCR, so each element on screen gets a label and a position.
3. **Act.** It matches the step to an on-screen label and clicks or types there. In Chrome, if a click doesn't change the page, it clicks the element through Selenium instead. It also clicks "Accept all" on cookie banners when it sees them.
4. **Recover.** It fingerprints the screen to notice when an action changed nothing. When it's stuck it tries Esc, then the browser's Back button, then asks the model for a different approach. When simple matching fails, the model picks the next action from the visible text ("Reflexor").
5. **Check.** It checks the URL or page text to confirm each step, and a final model call checks whether the whole goal was met.
6. **Remember.** It keeps a small JSON file, per website, of which labels led somewhere useful and which didn't, and prefers the ones that worked next time.

Only on-screen text and the current URL are sent to OpenAI, not screenshots.

## Setup

You need macOS, Python 3.8+, Google Chrome with a matching ChromeDriver, and an OpenAI API key.

```bash
pip install openai ultralytics torch paddlepaddle paddleocr numpy pillow mss pynput pyautogui selenium
```

Download the OmniParser V2 icon detection weights (`icon_detect/model.pt`) from [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0) on Hugging Face, then:

```bash
export OPENAI_API_KEY=your-key
export YOLO_WEIGHTS=/path/to/icon_detect/model.pt
```

If `YOLO_WEIGHTS` isn't found, the agent falls back to the generic `yolov8n.pt` model, which is much worse at finding UI icons.

The first time you run it, macOS will ask you to give your terminal Screen Recording and Accessibility permission.

## Running it

```bash
python agent_parser.py
```

It asks `Enter your goal:` and then starts working on the front window. Useful options:

| Option | What it does |
|---|---|
| `--save` | Save annotated screenshots and the detected elements to `agent_outputs/` |
| `--reuse-chrome` | Reuse an existing Chrome window or tab instead of opening a new one |
| `--debug-address 127.0.0.1:9222` | Attach to a Chrome started with remote debugging |
| `--paste-link` | Paste URLs into the address bar of the current tab instead of using Selenium |
| `--image screenshot.png` | Run the screen parser on a saved screenshot instead of the live screen |
| `--max-iters N` | Limit how many small steps it tries per sub-goal (default 14) |
| `--headless`, `--close-browser` | Run Selenium's Chrome headless, or close it at the end |

## What it can't do

- It only runs on macOS. It uses AppleScript to find and focus windows.
- It takes over your real mouse and keyboard, so you can't use the computer while it runs.
- OCR misreads small, low-contrast or unusual fonts, and then the agent can click the wrong thing.
- The language model sometimes plans the wrong steps or decides a goal is done when it isn't. The final check catches some of this, not all.
- CAPTCHAs, logins, custom controls and slow-loading pages often stop it.
- Waits between actions are fixed, so it can click before a page finishes loading or click through a pop-up.
- It can still get stuck in loops. The screen fingerprint makes this less likely but doesn't prevent it.
- Results vary from run to run, and there are no automated tests.
- On-screen text is sent to OpenAI, so don't run it with private information on screen.

## Status

A prototype from my summer 2025 internship; not under active development.

## License

This project is licensed under AGPL-3.0 (see [LICENSE](LICENSE)), the same license as the OmniParser icon detection weights and the Ultralytics library it uses.
