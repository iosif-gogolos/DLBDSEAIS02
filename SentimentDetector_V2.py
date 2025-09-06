from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import customtkinter as CTk
from tkinter import END
import re

# ------- simple resources -------
CONTRASTIVES = {"but","however","though","yet","although","even though","nevertheless","nonetheless"}
POS_WORDS = {"great","amazing","awesome","fantastic","love","perfect","wonderful","excellent"}
NEG_ACTIONS = {"breaks","crashes","ruins","fails","bugs","freezes","lags","drains","overheats"}
ASPECT_KEYWORDS = {
    "camera": {"camera","photo","image","picture"},
    "battery": {"battery","charge","power","life"},
    "performance": {"performance","speed","lag","slow","fast"},
    "display": {"display","screen","brightness","resolution"},
    "software": {"software","update","app","ui","os"},
}

sid = SentimentIntensityAnalyzer()

def clearAll():
    negativeField.delete(0,END)
    neutralField.delete(0,END)
    positiveField.delete(0, END)
    overallField.delete(0, END)
    aspectsField.delete(1.0, END)
    textArea.delete(1.0, END)

def _vader_label(compound: float) -> str:
    if compound >= 0.05: return "Positive"
    if compound <= -0.05: return "Negative"
    return "Neutral"

def _possible_sarcasm(text: str, base_scores: dict) -> bool:
    t = text.lower()
    # cue 1: positive word near negative action
    for pw in POS_WORDS:
        if pw in t:
            for na in NEG_ACTIONS:
                if na in t:
                    # require proximity within ~6 tokens
                    if re.search(rf"{pw}(?:\W+\w+){{0,6}}\W+{na}", t) or re.search(rf"{na}(?:\W+\w+){{0,6}}\W+{pw}", t):
                        return True
    # cue 2: strong contrastive structure
    if any(f" {c} " in f" {t} " for c in CONTRASTIVES) and base_scores["pos"] > base_scores["neg"]:
        return True
    # cue 3: exclamation with mixed polarity
    if "!" in text and base_scores["pos"] > 0.3 and base_scores["neg"] > 0.2:
        return True
    return False

def _adjust_for_sarcasm(scores: dict) -> dict:
    # down-weight positive, up-weight negative slightly; recompute compound proxy
    adj = scores.copy()
    adj["pos"] *= 0.6
    adj["neg"] = min(1.0, adj["neg"] * 1.3 + 0.05)
    # simple recompute for overall; keep neu scaled
    total = adj["pos"] + adj["neg"] + adj["neu"] + 1e-9
    compound_like = (adj["pos"] - adj["neg"]) / total
    adj["compound"] = compound_like
    return adj

def _split_clauses(text: str):
    # split on contrastives and punctuation
    parts = re.split(r"(?:,|;|\.|\bbut\b|\bhowever\b|\byet\b|\balthough\b)", text, flags=re.I)
    # clean
    return [p.strip() for p in parts if p and p.strip()]

def _detect_aspects(clause: str):
    found = set()
    lower = clause.lower()
    for asp, kws in ASPECT_KEYWORDS.items():
        if any(k in lower for k in kws):
            found.add(asp)
    return found or {"general"}

def detect_sentiment():
    text = textArea.get("1.0", "end").strip()
    if not text:
        return

    base = sid.polarity_scores(text)

    scores = base
    # sarcasm adjustment
    if _possible_sarcasm(text, base):
        scores = _adjust_for_sarcasm(base)

    # update top-level fields
    negativeField.delete(0, END); neutralField.delete(0, END); positiveField.delete(0, END); overallField.delete(0, END)

    negativeField.insert(0, f"{round(scores['neg']*100,1)}% Negative")
    neutralField.insert(0, f"{round(scores['neu']*100,1)}% Neutral")
    positiveField.insert(0, f"{round(scores['pos']*100,1)}% Positive")
    overallField.insert(0, _vader_label(scores['compound']))

    # aspect-level analysis
    aspectsField.delete(1.0, END)
    aspect_results = []
    for clause in _split_clauses(text):
        clause_scores = sid.polarity_scores(clause)
        if _possible_sarcasm(clause, clause_scores):
            clause_scores = _adjust_for_sarcasm(clause_scores)
        label = _vader_label(clause_scores["compound"])
        for asp in _detect_aspects(clause):
            aspect_results.append((asp, label, clause.strip()))

    # Merge duplicate aspects by last clause sentiment (simple); you can average if you prefer
    merged = {}
    for asp, lab, cl in aspect_results:
        merged[asp] = (lab, cl)

    # display
    lines = []
    for asp, (lab, cl) in merged.items():
        lines.append(f"- {asp}: {lab}   |  “{cl}”")
    if lines:
        aspectsField.insert(1.0, "Aspect-level:\n" + "\n".join(lines))

def button_callback():
    print("Sentiment analysis started...")

if __name__ == "__main__":
    app = CTk.CTk()
    app.title("Sentiment Detector")
    app.geometry("520x620")

    enterText = CTk.CTkLabel(app, text="Enter Your Review")
    textArea = CTk.CTkTextbox(app, height=120, width=360, corner_radius=10, pady=10)

    check = CTk.CTkButton(app, text="Check Sentiment", fg_color="red", corner_radius=10, command=detect_sentiment)

    negative = CTk.CTkLabel(app, text="Negative:")
    neutral = CTk.CTkLabel(app, text="Neutral:")
    positive = CTk.CTkLabel(app, text="Positive:")
    overall = CTk.CTkLabel(app, text="Overall:")

    negativeField = CTk.CTkEntry(app, width=260)
    neutralField  = CTk.CTkEntry(app, width=260)
    positiveField = CTk.CTkEntry(app, width=260)
    overallField  = CTk.CTkEntry(app, width=260)

    aspectsLabel = CTk.CTkLabel(app, text="Details by Aspect:")
    aspectsField = CTk.CTkTextbox(app, height=140, width=360, corner_radius=10)

    clear = CTk.CTkButton(app, text="Clear", fg_color="red", corner_radius=10, command=clearAll)
    Exit  = CTk.CTkButton(app, text="Exit", fg_color="red", corner_radius=10, command=exit)

    # layout
    enterText.grid(row=0, column=0, padx=20, pady=(15,5), sticky="w", columnspan=2)
    textArea.grid(row=1, column=0, padx=20, pady=5, sticky="w", columnspan=2)
    check.grid(row=2, column=0, padx=20, pady=10, sticky="w")

    negative.grid(row=3, column=0, padx=20, sticky="w"); negativeField.grid(row=3, column=1, padx=10, pady=4, sticky="w")
    neutral.grid(row=4, column=0, padx=20, sticky="w");  neutralField.grid(row=4, column=1, padx=10, pady=4, sticky="w")
    positive.grid(row=5, column=0, padx=20, sticky="w"); positiveField.grid(row=5, column=1, padx=10, pady=4, sticky="w")
    overall.grid(row=6, column=0, padx=20, sticky="w");  overallField.grid(row=6, column=1, padx=10, pady=4, sticky="w")

    aspectsLabel.grid(row=7, column=0, padx=20, pady=(10,0), sticky="w", columnspan=2)
    aspectsField.grid(row=8, column=0, padx=20, pady=5, sticky="w", columnspan=2)

    clear.grid(row=9, column=0, padx=20, pady=8, sticky="w")
    Exit.grid(row=9, column=1, padx=10, pady=8, sticky="e")

    app.mainloop()
