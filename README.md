# DLBDSEAIS02

Portfolio for Course Project Artificial Intelligence (DLBDSEAIS02)

# Installation & Usage Guide

This project provides two versions of a **Sentiment Analysis Tool** for customer reviews:

- **V1**: Baseline using VADER + CustomTkinter GUI
- **V2**: Improved version with aspect-based and sarcasm-aware heuristics

---

## 1. Clone the Repository

```bash
git clone https://github.com/iosif-gogolos/DLBDSEAIS02.git
cd DLBDSEAIS02
```

---

## 2. Create & Activate a Virtual Environment

### Windows (PowerShell)

```powershell
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

> If you see an *execution policy error*, run PowerShell as Administrator once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 3. Install Dependencies

If there is a single requirements file:

```bash
pip install -r requirements.txt
```

If there are per-version files:


Minimal requirements include:

```
vaderSentiment
customtkinter
```

Optional for evaluation:
```
scikit-learn
spacy
```

---

## 4. Run V1 and V2

From the repo root:

```bash
# Run baseline (V1)
python SentimentDetector_V1.py

# Run improved (V2)
python SentimentDetector_V2.py
```


---

## 5. Test Cases

- **Sarcasm Test**  
  Input: `Great, another software update that breaks everything!`  
  - V1: neutral/positive  
  - V2: **Negative**

- **Multi-polarity Test**  
  Input: `Great camera quality, terrible battery life.`  
  - V1: mixed single score  
  - V2: **camera: Positive, battery: Negative**

---

## 6. Freeze Dependencies

After installation:

```bash
pip freeze > requirements.txt
```

---

## 7. Troubleshooting

- **`ModuleNotFoundError: customtkinter`**  
  Ensure the virtual environment is active and run `pip install customtkinter`.

- **Tk errors on macOS**  
  Install Tk via Homebrew:  
  ```bash
  brew install python-tk@3.12
  ```

- **Fonts/emoji rendering**  
  Use ASCII labels or configure a supported font in CustomTkinter.

---
