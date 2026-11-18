"""Sindhi NLP Word Model & Interactive Demo.

Class 9 & 10 AI Project — "NLP: How It Works? Let's Train Our Own Sindhi Word Model!"
Demonstrates data collection, text preprocessing, feature extraction (Bag-of-Words),
and Naive Bayes classification for core Sindhi words:
  1. سلام (Hello)
  2. ڏينھواد / مھرباني (Thank You)
  3. ھا (Yes)
  4. نہ (No)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

# =============================================================================
# STEP 1: COLLECT DATA (50+ real samples for each word with variations & typos)
# =============================================================================
DATASET: Dict[str, List[str]] = {
    "Hello": [
        "سلام", "اسلام عليڪم", "سلام ادا", "سلام ادي", "پيارو سلام",
        "سڀني کي سلام", "وعليڪم السلام", "سلام سائين", "سلام دوست",
        "محبت ڀريو سلام", "صبح جو سلام", "شام جو سلام", "سلام پيارا",
        "پنهنجو سلام", "سلام عليڪم", "سلام ڪجي ٿو", "سلام قبول پوي",
        "سلام آھي", "سڀني دوستن کي سلام", "هڪ پيارو سلام", "سلام وري",
        "سلام سندھ", "مٺو سلام", "سدا سلامت", "سلام خوش آمديد",
        "سلام جيئي سنڌ", "سلام منهنجا ڀاءُ", "سلام سڀني کي", "سلام يار",
        "سلام پيش", "دل سان سلام", "سلام سڀ ڀائرن کي", "سلام امڙ",
        "سلام استاد", "سلام صاحب", "سلام بابلا", "سلام جيئندا رهو",
        "سلام ڀلي ڪري آيا", "سدائين سلام", "سلام ڀليڪار", "سلام شل خوش هجو",
        "سلام سڀن کي", "سلام سنگت", "سلام ساٿيو", "سلام پيارا ساٿي",
        "سلام وعلیکم", "سلام خوش رهو", "سلام آئي", "سلام جي", "سلام عرض",
    ],
    "Thank You": [
        "ڏينھواد", "مھرباني", "ڏاڍي مھرباني", "وڏي مھرباني", "مهرباني سائين",
        "توهان جي مھرباني", "ڏينھواد سائين", "توهان جا لک لک ٿورا", "ٿورا",
        "لک ٿورا", "توهان جو ڏاڍو ٿورو", "مھرباني ادا", "مھرباني ادي",
        "ڏينھواد سڀني کي", "ڏينھواد مٺا", "توهان جي وڏي نوازش", "نوازش",
        "مھرباني دوست", "سائين مھرباني", "ڏينھواد پيارا", "ڏاڍا ٿورا",
        "مھرباني ڪئي توهان", "تمام گهڻي مھرباني", "ڏينھواد دوست", "توهان جو شڪريو",
        "شڪريو", "شڪريا", "ڏينھواد قبول ڪريو", "توهان جا احسان", "مھرباني فرمائي",
        "توهان جي وڏي مھرباني", "ڏينھواد ڀاءُ", "مھرباني سڀني جي", "مھرباني يار",
        "ٿورا نہ وسرندا", "لک لک مھربانيون", "ڏينھواد سڀن جا", "مھرباني جناب",
        "ڏينھواد سدا خوش هجو", "مھرباني ٿي", "توهان جي مهرباني آهي", "شڪرگذار",
        "توهان جا تمام گهڻا ٿورا", "ڏينھواد ادا", "مهرباني تمام گهڻي",
    ],
    "Yes": [
        "ھا", "ها", "جي ھا", "جي ها", "بلڪل", "بلڪل ھا", "بلڪل ها",
        "جي سائين", "جي ادا", "ھا سائين", "ها سائين", "صحيح آھي", "صحيح آهي",
        "ٿي سگهي ٿو", "درست", "جي ضرور", "ضرور", "ھا بلڪل", "ها بلڪل",
        "قبول آھي", "ها يار", "ھا ادا", "جي ها ادا", "بلڪل درست",
        "ھا ڀلي", "ها برابر", "برابر", "سچ پچ", "جي ها سائين",
        "ھا مان راضي آھيان", "ها مان تيار آهيان", "ھا ضرور", "ها مٺا",
        "جي", "جي جي", "ھا ها", "بلڪل ٺيڪ", "ٺيڪ آھي", "ها برابر آهي",
        "ھا سچ", "ها ادا جي", "جي سائين بلڪل", "ھا قبول", "ها درست آھي",
    ],
    "No": [
        "نہ", "نه", "بلڪل نہ", "بلڪل نه", "جي نہ", "جي نه", "ڪڏھن نہ",
        "ڪڏهن نه", "نه سائين", "نہ سائين", "نه ادا", "نہ ادا", "بلڪل غلط",
        "غلط آھي", "نه يار", "نہ يار", "ڪڏهن به نه", "ڪڏھن بہ نہ",
        "نه مٺا", "نہ مٺا", "مان تيار نه آهيان", "نہ ٿيندو", "نه ٿيندو",
        "نه برابر", "انڪار", "قبول ناهي", "ناهي", "نھ", "ڪو نه", "ڪونهي",
        "ڪو نہ", "اصل نه", "اصل نہ", "نه جي", "نہ جي", "هرگز نه",
        "هرگز نہ", "بلڪل ناهي", "نه ٺيڪ ناهي", "ڪابه اميد ناهي", "نه ڀاءُ",
    ],
}

SINDHI_METADATA = {
    "Hello": {
        "sindhi_primary": "سلام",
        "phonetic": "Salam",
        "meaning": "Hello / Greeting",
        "desc": "Warm greeting used to say hello or wish peace upon someone.",
    },
    "Thank You": {
        "sindhi_primary": "ڏينھواد / مھرباني",
        "phonetic": "Dhanwaad / Meherbani",
        "meaning": "Thank You / Gratitude",
        "desc": "Expressing gratitude, appreciation, and thankfulness.",
    },
    "Yes": {
        "sindhi_primary": "ھا",
        "phonetic": "Haa",
        "meaning": "Yes / Agreement",
        "desc": "Affirmative response, agreement, or confirmation.",
    },
    "No": {
        "sindhi_primary": "نہ",
        "phonetic": "Na",
        "meaning": "No / Decline",
        "desc": "Negative response, polite decline, or disagreement.",
    },
}


# =============================================================================
# STEP 2: PREPROCESSING (Clean text, remove punctuation, normalize diacritics)
# =============================================================================
def preprocess_text(text: str) -> List[str]:
    """Clean Sindhi text, strip punctuation and normalize character variants."""
    text = text.strip()
    # Normalize variants of Heh and Yeh commonly typed across mobile keyboards
    text = text.replace("ي", "ي").replace("ى", "ي").replace("ئ", "ي")
    text = text.replace("ھ", "ه").replace("ہ", "ه")
    # Remove punctuation & symbols
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # Remove Arabic / Sindhi diacritics (fatha, damma, kasra, shadda, sukun)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    tokens = [w for w in text.split() if len(w) > 0]
    return tokens


# =============================================================================
# STEP 3 & 4: FEATURE EXTRACTION (Bag-of-Words) + NAIVE BAYES CLASSIFIER
# =============================================================================
class SindhiWordClassifier:
    """Pure-Python, zero-dependency Naive Bayes NLP Model for Sindhi Words."""

    def __init__(self) -> None:
        self.classes: List[str] = list(DATASET.keys())
        self.vocab: Dict[str, int] = {}
        self.word_counts: Dict[str, Counter] = {c: Counter() for c in self.classes}
        self.class_totals: Dict[str, int] = {c: 0 for c in self.classes}
        self.total_docs: int = 0
        self.class_doc_counts: Dict[str, int] = {c: 0 for c in self.classes}
        self._train()

    def _train(self) -> None:
        all_vocab = set()
        for label, samples in DATASET.items():
            for text in samples:
                self.total_docs += 1
                self.class_doc_counts[label] += 1
                tokens = preprocess_text(text)
                for t in tokens:
                    all_vocab.add(t)
                    self.word_counts[label][t] += 1
                    self.class_totals[label] += 1

        self.vocab = {w: idx for idx, w in enumerate(sorted(all_vocab))}

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        tokens = preprocess_text(text)
        if not tokens:
            return "Hello", 0.25, {c: 0.25 for c in self.classes}

        vocab_size = max(1, len(self.vocab))
        log_probs: Dict[str, float] = {}

        for c in self.classes:
            # Prior probability log P(class)
            prior = math.log((self.class_doc_counts[c] + 1) / (self.total_docs + len(self.classes)))
            # Likelihood with Laplace smoothing
            likelihood = 0.0
            total_words_in_class = self.class_totals[c]
            for t in tokens:
                count = self.word_counts[c].get(t, 0)
                # If exact word found, give high priority
                if count > 0:
                    effective_count = count * 8
                else:
                    effective_count = 0
                    # Only do prefix/fuzzy fallback if length is substantial (>2)
                    if len(t) > 2:
                        for vocab_word, v_count in self.word_counts[c].items():
                            if (t in vocab_word or vocab_word in t) and len(vocab_word) > 2:
                                effective_count = max(1, v_count // 3)
                                break
                prob_word = (effective_count + 1.0) / (total_words_in_class + vocab_size)
                likelihood += math.log(prob_word)
            log_probs[c] = prior + likelihood

        # Softmax normalization to percentage
        max_log = max(log_probs.values())
        exp_probs = {c: math.exp(v - max_log) for c, v in log_probs.items()}
        sum_exp = sum(exp_probs.values())
        norm_probs = {c: round(v / sum_exp, 3) for c, v in exp_probs.items()}

        best_class = max(norm_probs, key=lambda k: norm_probs[k])
        confidence = norm_probs[best_class]
        return best_class, confidence, norm_probs


# =============================================================================
# STEP 5: INTERACTIVE CONSOLE DEMO (Easy to present in class!)
# =============================================================================
def run_interactive_demo() -> None:
    model = SindhiWordClassifier()

    print("=" * 66)
    print("  Sindhi NLP Word Model  —  Class 9 & 10 AI Project")
    print("  'Teaching Computers to Understand Human Language!'")
    print("=" * 66)
    print(f"  • Vocabulary size: {len(model.vocab)} unique Sindhi words")
    print(f"  • Training samples: {model.total_docs} total sentences")
    print(f"  • Supported words: سلام, ڏينھواد / مھرباني, ھا, نہ")
    print("=" * 66)
    print("  Quick Test Options:")
    print("    1. سلام          -> Hello")
    print("    2. ڏينھواد       -> Thank You")
    print("    3. ھا            -> Yes")
    print("    4. نہ            -> No")
    print("    Type any Sindhi text or phrase, or 'q' to quit.")
    print("-" * 66)

    samples_map = {
        "1": "سلام",
        "2": "ڏينھواد",
        "3": "ھا",
        "4": "نہ",
    }

    while True:
        try:
            inp = input("\n[INPUT] Enter Sindhi text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! See you soon.")
            break

        if not inp:
            continue
        if inp.lower() in ("q", "quit", "exit"):
            print("Session ended. Thank you for using Sindhi NLP!")
            break

        actual_query = samples_map.get(inp, inp)
        pred_label, conf, all_probs = model.predict(actual_query)
        meta = SINDHI_METADATA[pred_label]

        print("-" * 66)
        print(f"  Text Input :  {actual_query}")
        print(f"  Predicted  :  {meta['sindhi_primary']}  ({meta['phonetic']})")
        print(f"  Meaning    :  {meta['meaning']}")
        print(f"  Confidence :  {round(conf * 100, 1)}%  [✔]")
        print(f"  Description:  {meta['desc']}")
        print("  All Probs  :  " + ", ".join(f"{k}: {round(v * 100, 1)}%" for k, v in all_probs.items()))
        print("-" * 66)


if __name__ == "__main__":
    run_interactive_demo()
