# DLBDSEAIS02

Portfolio for Course Project Artificial Intelligence (DLBDSEAIS02)

# Final References & Documentation

## How this fulfills the specification from Phase-1

- **Preprocessing/Tokenization** – handled by Hugging Face tokenizers.  
- **Feature Extraction Engine** – BERT/RoBERTa embeddings inside the classifiers.  
- **Linguistic Rules Engine** – contrastive clause split + aspect keywords + minimal sarcasm cues.  
- **Classification Layer** – BERT sentiment + NLI (textual entailment) for sarcasm; soft ensemble by clause aggregation.  
- **Validation & Output** – overall + per-aspect labels and a sarcasm flag rendered in the GUI.  

---

## Notes for Abstract / Final Report

- Replaced lexicon baseline with **transformer sentiment**; improved negation handling as expected with sequence models (Singh, 2021).  
- Added **NLI-based sarcasm detector**; contradiction between review and “This review is positive/negative.” is a strong irony cue (Sinha et al., 2023).  
- Kept **transparent aspect view** via keywords to surface multi-polarity.  
- Literature supports **deep models outperforming classic ML** for review sentiment (Ashbaugh & Zhang, 2024).  
- Background on **negation/rule hybrids** and **open challenges** (Pandey et al., 2018; Shwetha, 2023).  

---

## References & Bibliography

Ashbaugh, L., & Zhang, Y. (2024). A comparative study of sentiment analysis on customer reviews using machine learning and deep learning. *Computers, 13*(12), 340. https://doi.org/10.3390/computers13120340  

Pandey, S., Sagnika, S., & Mishra, B. S. P. (2018). A technique to handle negation in sentiment analysis on movie reviews. In *2018 International Conference on Communication and Signal Processing (ICCSP)* (pp. 0478–0482). IEEE. https://doi.org/10.1109/ICCSP.2018.8524562  

Shwetha, C. H. (2023). Exploring sentiment analysis: Applications and challenges — A comprehensive survey. *International Journal of Scientific Research in Engineering and Management (IJSREM), 7*(4), 1–9.  

Singh, P. K. (2021). Deep learning approach for negation handling in sentiment analysis. *IEEE Access, 9*, 120463–120475. https://doi.org/10.1109/ACCESS.2021.3109010  

Sinha, S., Vijeta, T., Kubde, P., Gajbhiye, A., Radke, M. A., & Jones, C. B. (2023). Sarcasm detection in product reviews using textual entailment approach. In *Proceedings of the 7th International Conference on Natural Language Processing and Information Retrieval (NLPIR 2023)* (pp. 101–108). ACM. https://doi.org/10.1145/3625068.3626505  

---

### Additional Supporting Sources

Chatterjee, A., et al. (2020). Detection of sarcasm on Amazon product reviews using machine learning algorithms under sentiment analysis. *EasyChair Preprint 2389*.  

Joshi, A., & Desai, A. (2021). Negation handling in sentiment analysis at sentence level. *International Journal of Computer Applications, 183*(34), 19–24.  

Khattri, N., & Kumar, P. (2022). Techniques of sarcasm detection: A review. *International Journal of Engineering and Advanced Technology, 11*(5), 10–15.  

Mishra, P., & Taterh, S. (2020). Sentiment analysis: A survey of techniques and applications. *EasyChair Preprint 1989*.  

Sinha, S., et al. (2023). Sarcasm detection in product reviews using textual entailment approach. *NLPIR 2023 Proceedings.*  


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
