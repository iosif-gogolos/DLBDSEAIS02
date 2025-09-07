"""
SentimentDetectorFinal.py
Phase 3 (Finalization): BERT + NLI + heuristics + aspect view in a stable CustomTkinter GUI

Architecture mapping (see Phase-1 UML):
- Input Layer: GUI textbox (CustomTkinter)
- Processing Pipeline: cleaning + clause split
- Feature Engineering:
    * Transformer embeddings via pretrained models (BERT/RoBERTa)
    * Lightweight linguistic rules: contrastives, aspect keywords
- Classification Layer:
    * Transformer classifier for valence (pos/neu/neg)
    * NLI (entailment/contradiction) to flag sarcasm/irony
    * Simple ensemble: clause-level evidence + sarcasm adjustment
- Validation & Output:
    * Overall and per-aspect labels; display in GUI

Key references (APA-style in-code citations):
- Negation handling benefits from sequence models (BiLSTM/transformers): Singh, P. K. (2021). Deep learning approach for negation handling in sentiment analysis. IEEE Access. :contentReference[oaicite:0]{index=0}
- Sarcasm via textual entailment (contradiction signal): Sinha, S., Vijeta, T., Kubde, P., Gajbhiye, A., Radke, M. A., & Jones, C. B. (2023). Sarcasm detection in product reviews using textual entailment. In NLPIR 2023 (ACM). :contentReference[oaicite:1]{index=1}
- Comparative evidence that deep models outperform classic ML on reviews: Ashbaugh, L., & Zhang, Y. (2024). A comparative study of sentiment analysis on customer reviews using machine learning and deep learning. Computers, 13(12), 340. :contentReference[oaicite:2]{index=2}
- Negation/scope and rule hybrids background: Pandey, S., Sagnika, S., & Mishra, B. S. P. (2018). A technique to handle negation in sentiment analysis on movie reviews. IEEE ICCSP. :contentReference[oaicite:3]{index=3}
- Challenge surveys (sarcasm, multipolarity, etc.): Shwetha, C. H. (2023). Exploring sentiment analysis: Applications and challenges. IJSREM. :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# ---- GUI ----
import customtkinter as CTk
from tkinter import END

# ---- Transformers (BERT/RoBERTa) ----
# NOTE: first run will download models; ensure internet access.
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


# ---------------------------- Config ----------------------------
# Contrastives help us split multi-polar statements into clauses.
CONTRASTIVES = {"but","however","though","yet","although","even though","nevertheless","nonetheless"}

# Simple aspect lexicons (kept from V2 to preserve explainability).
ASPECT_KEYWORDS = {
    "camera": {"camera","photo","image","picture"},
    "battery": {"battery","charge","power","life"},
    "performance": {"performance","speed","lag","slow","fast"},
    "display": {"display","screen","brightness","resolution"},
    "software": {"software","update","app","ui","os"},
}

# Heuristic cues used only as tie-breakers (kept minimal; transformers dominate).
POS_WORDS = {"great","amazing","awesome","fantastic","love","perfect","wonderful","excellent"}
NEG_ACTIONS = {"breaks","crashes","ruins","fails","bugs","freezes","lags","drains","overheats"}


# ---------------------------- Utilities ----------------------------
def split_clauses(text: str) -> List[str]:
    """Split by punctuation + contrastives (V2 carry-over).
    This addresses multipolarity by scoring smaller units first.
    """
    parts = re.split(r"(?:[.;]|,|\bbut\b|\bhowever\b|\byet\b|\balthough\b|\bnevertheless\b|\bnonetheless\b)", text, flags=re.I)
    return [p.strip() for p in parts if p and p.strip()]


def detect_aspects(span: str) -> List[str]:
    """Map a text span to aspect buckets using keyword matches (transparent explainability)."""
    found = set()
    lower = span.lower()
    for asp, kws in ASPECT_KEYWORDS.items():
        if any(k in lower for k in kws):
            found.add(asp)
    return list(found) or ["general"]


def normalize_label(sent_label: str) -> str:
    """Map model-specific labels to {Positive, Neutral, Negative}."""
    s = sent_label.lower()
    if "pos" in s or "positive" in s or s in {"5 stars", "4 stars"}: return "Positive"
    if "neg" in s or "negative" in s or s in {"1 star", "2 stars"}:   return "Negative"
    return "Neutral"


@dataclass
class ModelBundle:
    sentiment: TextClassificationPipeline
    nli: TextClassificationPipeline


# ---------------------------- Core Analyzer ----------------------------
class Analyzer:
    """
    Final analyzer implementing:
    - Transformer valence classification (BERT/RoBERTa).
    - NLI-based sarcasm flag (premise: review; hypotheses: +/- stance),
      inspired by textual entailment signal of contradiction (Sinha et al., 2023).  # APA: Sinha et al., 2023 :contentReference[oaicite:5]{index=5}
    - Clause-level aggregation for multipolarity (per-aspect).
    - Light rules as guardrails; negation largely handled by the transformer
      (sequence models learn cue+scope better than static lexicons; Singh, 2021).  # APA: Singh, 2021 :contentReference[oaicite:6]{index=6}
    """

    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        nli_model: str = "roberta-large-mnli",
        device: int = -1,
    ):
        # Sentiment pipeline
        tok_s = AutoTokenizer.from_pretrained(sentiment_model)
        mod_s = AutoModelForSequenceClassification.from_pretrained(sentiment_model)
        self.sentiment = TextClassificationPipeline(model=mod_s, tokenizer=tok_s, device=device, top_k=None, return_all_scores=True)

        # NLI pipeline (for sarcasm via contradiction)
        tok_n = AutoTokenizer.from_pretrained(nli_model)
        mod_n = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.nli = TextClassificationPipeline(model=mod_n, tokenizer=tok_n, device=device, top_k=None, return_all_scores=True)

    # ---------- Sentiment ----------
    def _sent_probs(self, text: str) -> Dict[str, float]:
        """Get normalized {pos, neu, neg} probabilities from the sentiment model."""
        out = self.sentiment(text)[0]  # list of dicts: [{'label':'LABEL_0','score':...}, ...]
        label_to_score = {normalize_label(d['label']): d['score'] for d in out}
        # Ensure all keys exist
        for k in ("Positive","Neutral","Negative"):
            label_to_score.setdefault(k, 0.0)
        # Normalize
        s = sum(label_to_score.values()) + 1e-9
        return {k: v/s for k, v in label_to_score.items()}

    def _label_from_probs(self, p: Dict[str, float]) -> str:
        best = max(p.items(), key=lambda kv: kv[1])[0]
        return best

    # ---------- Sarcasm via NLI ----------
    def _sarcasm_flag(self, text: str, probs: Dict[str, float]) -> bool:
        """
        Use textual entailment to detect contradiction between the review (premise)
        and polarity statements (hypotheses). If the review strongly contradicts the
        *dominant* sentiment hypothesis, treat as a sarcasm cue (Sinha et al., 2023).
        """
        premise = text
        hyp_pos = "This review is positive."
        hyp_neg = "This review is negative."

        def norm_nli_scores(res_item):
            """
            Normalize NLI scores into a dict with keys {ENTAILMENT, NEUTRAL, CONTRADICTION}.
            Works for:
              - [ {'label': 'ENTAILMENT','score':...}, ... ]  (list)
              - {'label':'CONTRADICTION','score':...}          (single dict)
            """
            if isinstance(res_item, list):
                pairs = [(d.get("label", "").upper(), float(d.get("score", 0.0))) for d in res_item]
                return {k: v for k, v in pairs}
            elif isinstance(res_item, dict):
                # Single best label — create a one-hot-like mapping
                lab = res_item.get("label", "").upper()
                sc  = float(res_item.get("score", 0.0))
                base = {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}
                # Some checkpoints use LABEL_0/1/2. Heuristic remap if needed.
                if lab.startswith("LABEL_"):
                    # Common mapping used by roberta-large-mnli: LABEL_0=CONTRADICTION, LABEL_1=NEUTRAL, LABEL_2=ENTAILMENT
                    mapping = {"LABEL_0": "CONTRADICTION", "LABEL_1": "NEUTRAL", "LABEL_2": "ENTAILMENT"}
                    lab = mapping.get(lab, lab)
                base[lab] = sc
                return base
            else:
                # Unexpected — return zeros
                return {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}

        def contradiction_prob(hypothesis: str) -> float:
            out = self.nli({"text": premise, "text_pair": hypothesis})
            # pipeline returns a list; take first item
            first = out[0] if isinstance(out, list) else out
            scores = norm_nli_scores(first)
            return float(scores.get("CONTRADICTION", 0.0))

        c_pos = contradiction_prob(hyp_pos)
        c_neg = contradiction_prob(hyp_neg)

        dominant = "Positive" if probs.get("Positive", 0.0) >= probs.get("Negative", 0.0) else "Negative"
        return (dominant == "Positive" and c_pos > 0.6) or (dominant == "Negative" and c_neg > 0.6)


    # ---------- Clause / Aspect aggregation ----------
    def analyze(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {"overall": "Neutral", "probs": {"Positive":0,"Neutral":1,"Negative":0}, "sarcasm": False, "aspects": {}}

        # Clause scoring (multipolarity handling: split then score)
        clauses = split_clauses(text) or [text]
        aspect_scores: Dict[str, List[Dict[str,float]]] = {}
        for c in clauses:
            p = self._sent_probs(c)
            labs = detect_aspects(c)
            for a in labs:
                aspect_scores.setdefault(a, []).append(p)

        # Aggregate per aspect (mean probs)
        aspect_results: Dict[str, Tuple[str, Dict[str,float], List[str]]] = {}
        for a, plist in aspect_scores.items():
            agg = {"Positive":0.0,"Neutral":0.0,"Negative":0.0}
            for p in plist:
                for k in agg: agg[k] += p[k]
            n = max(len(plist), 1)
            agg = {k: v/n for k,v in agg.items()}
            aspect_results[a] = (self._label_from_probs(agg), agg, plist)

        # Overall = weighted by number of clauses per aspect (simple average across all clause scores)
        all_probs = {"Positive":0.0,"Neutral":0.0,"Negative":0.0}
        count = 0
        for _,(_, agg, _) in aspect_results.items():
            for k in all_probs: all_probs[k] += agg[k]
            count += 1
        if count:
            all_probs = {k: v/count for k,v in all_probs.items()}
        overall_label = self._label_from_probs(all_probs)

        # Sarcasm flag using NLI (textual entailment) over the full review, then a light heuristic nudge
        sarcasm = self._sarcasm_flag(text, all_probs)
        if not sarcasm:
            # Guardrail: positive word near negative action (legacy cue).
            t = text.lower()
            for pw in POS_WORDS:
                if pw in t:
                    for na in NEG_ACTIONS:
                        if na in t and re.search(rf"{pw}(?:\W+\w+){{0,6}}\W+{na}|{na}(?:\W+\w+){{0,6}}\W+{pw}", t):
                            sarcasm = True
                            break

        # If sarcasm is flagged and overall looks positive, soften to neutral/negative (small nudge).
        if sarcasm and overall_label == "Positive" and all_probs["Positive"] - all_probs["Negative"] < 0.25:
            # Minimal adjustment to reflect suspicion; we do not arbitrarily flip.
            all_probs["Neutral"] = min(1.0, all_probs["Neutral"] + 0.10)
            all_probs["Positive"] = max(0.0, all_probs["Positive"] - 0.10)
            overall_label = self._label_from_probs(all_probs)

        # Notes on negation:
        # Transformers encode negation cue+scope via attention over sequences (Singh, 2021), reducing the need
        # for manual polarity flips; rule-based negation remains brittle (Pandey et al., 2018).  # APA: Singh, 2021; Pandey et al., 2018 

        return {
            "overall": overall_label,
            "probs": all_probs,
            "sarcasm": sarcasm,
            "aspects": {a: {"label": lab, "probs": probs} for a,(lab, probs, _) in aspect_results.items()},
        }


# ---------------------------- GUI wiring ----------------------------
class App:
    def __init__(self):
        self.model = Analyzer()  # default models; change names here if desired

        self.app = CTk.CTk()
        self.app.title("Sentiment Detector (Final)")
        self.app.geometry("580x800")

        self.enterText = CTk.CTkLabel(self.app, text="Enter Your Review")
        self.textArea = CTk.CTkTextbox(self.app, height=140, width=420, corner_radius=10, pady=10)

        self.check = CTk.CTkButton(self.app, text="Analyze", fg_color="red", corner_radius=10, command=self.detect_sentiment)

        self.negative = CTk.CTkLabel(self.app, text="Negative:")
        self.neutral  = CTk.CTkLabel(self.app, text="Neutral:")
        self.positive = CTk.CTkLabel(self.app, text="Positive:")
        self.overall  = CTk.CTkLabel(self.app, text="Overall:")
        self.sarcasmL = CTk.CTkLabel(self.app, text="Sarcasm/Irony:")

        self.negativeField = CTk.CTkEntry(self.app, width=320)
        self.neutralField  = CTk.CTkEntry(self.app, width=320)
        self.positiveField = CTk.CTkEntry(self.app, width=320)
        self.overallField  = CTk.CTkEntry(self.app, width=320)
        self.sarcasmField  = CTk.CTkEntry(self.app, width=320)

        self.aspectsLabel = CTk.CTkLabel(self.app, text="Details by Aspect:")
        self.aspectsField = CTk.CTkTextbox(self.app, height=260, width=420, corner_radius=10)

        self.clear = CTk.CTkButton(self.app, text="Clear", fg_color="red", corner_radius=10, command=self.clear_all)
        self.Exit  = CTk.CTkButton(self.app, text="Exit",  fg_color="red", corner_radius=10, command=exit)

        # layout
        self.enterText.grid(row=0, column=0, padx=20, pady=(15,5), sticky="w", columnspan=2)
        self.textArea.grid(row=1, column=0, padx=20, pady=5, sticky="w", columnspan=2)
        self.check.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        self.negative.grid(row=3, column=0, padx=20, sticky="w"); self.negativeField.grid(row=3, column=1, padx=10, pady=4, sticky="w")
        self.neutral.grid (row=4, column=0, padx=20, sticky="w"); self.neutralField .grid(row=4, column=1, padx=10, pady=4, sticky="w")
        self.positive.grid(row=5, column=0, padx=20, sticky="w"); self.positiveField.grid(row=5, column=1, padx=10, pady=4, sticky="w")
        self.overall.grid (row=6, column=0, padx=20, sticky="w"); self.overallField .grid(row=6, column=1, padx=10, pady=4, sticky="w")
        self.sarcasmL.grid(row=7, column=0, padx=20, sticky="w"); self.sarcasmField.grid(row=7, column=1, padx=10, pady=4, sticky="w")

        self.aspectsLabel.grid(row=8, column=0, padx=20, pady=(10,0), sticky="w", columnspan=2)
        self.aspectsField.grid(row=9, column=0, padx=20, pady=5, sticky="w", columnspan=2)

        self.clear.grid(row=10, column=0, padx=20, pady=8, sticky="w")
        self.Exit.grid (row=10, column=1, padx=10, pady=8, sticky="e")

    def clear_all(self):
        self.negativeField.delete(0,END)
        self.neutralField .delete(0,END)
        self.positiveField.delete(0, END)
        self.overallField .delete(0, END)
        self.sarcasmField .delete(0, END)
        self.aspectsField.delete(1.0, END)
        self.textArea.delete(1.0, END)

    def detect_sentiment(self):
        text = self.textArea.get("1.0", "end").strip()
        if not text:
            return
        result = self.model.analyze(text)

        p = result["probs"]
        self.negativeField.delete(0,END); self.neutralField.delete(0,END); self.positiveField.delete(0,END); self.overallField.delete(0,END); self.sarcasmField.delete(0,END)
        self.negativeField.insert(0, f"{round(p['Negative']*100,1)}%")
        self.neutralField .insert(0, f"{round(p['Neutral']*100,1)}%")
        self.positiveField.insert(0, f"{round(p['Positive']*100,1)}%")
        self.overallField .insert(0, result["overall"])
        self.sarcasmField.insert(0, "Likely" if result["sarcasm"] else "Unlikely")

        # Aspect display
        self.aspectsField.delete(1.0, END)
        lines = []
        for asp, info in sorted(result["aspects"].items()):
            lp = info["probs"]
            lines.append(f"- {asp}: {info['label']}  |  P:{lp['Positive']:.2f} N:{lp['Negative']:.2f} U:{lp['Neutral']:.2f}")
        if lines:
            self.aspectsField.insert(1.0, "Aspect-level:\n" + "\n".join(lines))

    def run(self):
        self.app.mainloop()


if __name__ == "__main__":
    # Deep models have shown consistent gains over classic ML on customer reviews (Ashbaugh & Zhang, 2024). 
    App().run()
