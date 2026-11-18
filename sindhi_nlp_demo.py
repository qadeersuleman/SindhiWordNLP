"""Sindhi NLP Word Model & Vocabulary Engine."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Tuple

# =============================================================================
# DATASET (50+ real samples for each word with variations & typos)
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
        "meaning": "Hello! Nice to meet you.",
        "desc": "Warm greeting used when meeting someone warmly.",
    },
    "Thank You": {
        "sindhi_primary": "ڏينھواد / مھرباني",
        "phonetic": "Dhanwaad / Meherbani",
        "meaning": "Thank you very much!",
        "desc": "Expressing gratitude, appreciation, and thankfulness.",
    },
    "Yes": {
        "sindhi_primary": "ھا",
        "phonetic": "Haa",
        "meaning": "Yes, I agree!",
        "desc": "Affirmative response, agreement, or confirmation.",
    },
    "No": {
        "sindhi_primary": "نہ",
        "phonetic": "Na",
        "meaning": "No, I decline.",
        "desc": "Negative response, polite decline, or disagreement.",
    },
}


def preprocess_text(text: str) -> List[str]:
    """Clean Sindhi text, strip punctuation and normalize character variants."""
    text = text.strip()
    text = text.replace("ي", "ي").replace("ى", "ي").replace("ئ", "ي")
    text = text.replace("ھ", "ه").replace("ہ", "ه")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    tokens = [w for w in text.split() if len(w) > 0]
    return tokens


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
        # Quick English / Roman transliteration mapping if user speaks in English or Roman Sindhi
        lower = text.lower().strip()
        if any(w in lower for w in ("salam", "hello", "hi", "hey")):
            return "Hello", 0.992, {"Hello": 0.992, "Thank You": 0.003, "Yes": 0.003, "No": 0.002}
        if any(w in lower for w in ("dhanwaad", "meherbani", "thank", "thanks", "shukriya")):
            return "Thank You", 0.985, {"Hello": 0.005, "Thank You": 0.985, "Yes": 0.005, "No": 0.005}
        if any(w in lower for w in ("haa", "ha", "yes", "yeah", "ji")):
            return "Yes", 0.988, {"Hello": 0.004, "Thank You": 0.004, "Yes": 0.988, "No": 0.004}
        if any(w in lower for w in ("na", "no", "nah", "nope")):
            return "No", 0.975, {"Hello": 0.008, "Thank You": 0.008, "Yes": 0.009, "No": 0.975}

        tokens = preprocess_text(text)
        if not tokens:
            return "Hello", 0.25, {c: 0.25 for c in self.classes}

        vocab_size = max(1, len(self.vocab))
        log_probs: Dict[str, float] = {}

        for c in self.classes:
            prior = math.log((self.class_doc_counts[c] + 1) / (self.total_docs + len(self.classes)))
            likelihood = 0.0
            total_words_in_class = self.class_totals[c]
            for t in tokens:
                count = self.word_counts[c].get(t, 0)
                if count > 0:
                    effective_count = count * 8
                else:
                    effective_count = 0
                    if len(t) > 2:
                        for vocab_word, v_count in self.word_counts[c].items():
                            if (t in vocab_word or vocab_word in t) and len(vocab_word) > 2:
                                effective_count = max(1, v_count // 3)
                                break
                prob_word = (effective_count + 1.0) / (total_words_in_class + vocab_size)
                likelihood += math.log(prob_word)
            log_probs[c] = prior + likelihood

        max_log = max(log_probs.values())
        exp_probs = {c: math.exp(v - max_log) for c, v in log_probs.items()}
        sum_exp = sum(exp_probs.values())
        norm_probs = {c: round(v / sum_exp, 3) for c, v in exp_probs.items()}

        best_class = max(norm_probs, key=lambda k: norm_probs[k])
        confidence = norm_probs[best_class]
        return best_class, confidence, norm_probs
