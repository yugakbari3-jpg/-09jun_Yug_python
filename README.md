# SkinScope

A photo-based acne and skin tracker that runs entirely on your own computer.
Take or upload a selfie and SkinScope gives you a skin report, a breakout
outlook, a nutrition watch-list and a simple plan, then tracks how things change.

> **Not medical advice.** Spot detection is rule-based computer vision and has not
> been clinically validated. It can miss spots or mistake freckles and moles for
> them. See a doctor or dermatologist for any skin concern.

## Run it

You need Python 3 (tested with Python 3.13).

**macOS / Linux**

```bash
git clone https://github.com/yugakbari3-jpg/-09jun_Yug_python.git
cd ./-09jun_Yug_python
python3 -m venv .venv
source .venv/bin/activate
pip install "opencv-python<5" numpy
python skinscope_app.py
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/yugakbari3-jpg/-09jun_Yug_python.git
cd .\-09jun_Yug_python
python -m venv .venv
.venv\Scripts\activate
pip install "opencv-python<5" numpy
python skinscope_app.py
```

The app opens at <http://127.0.0.1:8765>. Press **Ctrl+C** in the terminal to stop it.
Keep the `./` (or `.\`) in the `cd` command: the folder name starts with `-`.

| Option | What it does |
|---|---|
| `--port 8800` | Use a different port |
| `--no-browser` | Don't open the browser automatically |
| `--reset-password you@example.com` | Set a new password for an account |
| `--detector rules` | Force the rule-based spot detector |

## What it does

- **Scan**: front photo plus optional side photos, by upload or live camera. The
  camera checks lighting, sharpness, distance and face position as you go, and
  can capture automatically.
- **Skin report**: skin score, active spots and dark marks by face zone, oiliness
  and redness evenness, with the detections drawn on your photo.
- **Breakout outlook**: an estimate for the next 7 days, with the factors behind it.
- **Nutrition watch**: nutrients you may be low on, based on your food answers.
  Evidence labels are sourced in [EVIDENCE_NOTES.md](EVIDENCE_NOTES.md).
- **Plan**: foods, habits and a skincare routine that respects pregnancy status
  and ingredients you react to.
- **History**: score trend, scan-to-scan comparison and a hotspot map of where
  spots keep appearing.
- **Diary and Trigger Lab**: log sleep, stress, food and breakouts, then run a
  4–6 week personal test (for example, cutting dairy) with honest statistics.
- **Doctor summary**: a printable overview to take to an appointment.

## Privacy

- Accounts, results and diary entries are stored only on this computer, in
  `~/skinscope_data/skinscope_app.db` (set `SKINSCOPE_DIR` to change the folder).
- Photos are analysed in memory and never saved; only numbers and spot positions
  are kept.
- Passwords are stored as salted hashes.
- The app makes no requests to outside services.
