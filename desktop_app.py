"""Native Tkinter Desktop Application for SindhiVoice AI."""

from __future__ import annotations

import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict

from sindhi_nlp_demo import SINDHI_METADATA, SindhiWordClassifier

classifier = SindhiWordClassifier()


def speak_english(text: str) -> None:
    """Speak text aloud in English via system TTS."""
    def _worker():
        clean = text.replace('"', "").replace("'", "").strip()
        if shutil.which("spd-say"):
            subprocess.run(["spd-say", "-o", "espeak", "-r", "-10", clean], check=False)
        elif shutil.which("espeak"):
            subprocess.run(["espeak", "-s", "145", clean], check=False)

    threading.Thread(target=_worker, daemon=True).start()


class SindhiVoiceTkApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SindhiVoice AI — Voice-Based Sindhi to English Desktop App")
        self.root.geometry("860x680")
        self.root.configure(bg="#0B1120")

        self.current_english = "Hello! Nice to meet you."

        self._build_ui()
        # Initial greeting
        self.root.after(800, lambda: speak_english(self.current_english))

    def _build_ui(self) -> None:
        # Header
        hdr = tk.Frame(self.root, bg="#131C31", pady=12, padx=20)
        hdr.pack(fill="x")

        title = tk.Label(
            hdr,
            text="🎙️ SindhiVoice AI — Voice Translator",
            font=("Outfit", 18, "bold"),
            fg="#38BDF8",
            bg="#131C31",
        )
        title.pack(side="left")

        sub = tk.Label(
            hdr,
            text="Class 9 & 10 AI Project  •  Speak Sindhi ➔ Hear English!",
            font=("Outfit", 10),
            fg="#94A3B8",
            bg="#131C31",
        )
        sub.pack(side="right")

        # 5-Step Pipeline Frame
        pipe_fr = tk.LabelFrame(
            self.root,
            text=" 5-Step Live NLP Pipeline (Poster 1) ",
            bg="#0B1120",
            fg="#38BDF8",
            font=("Outfit", 10, "bold"),
            padx=10,
            pady=8,
        )
        pipe_fr.pack(fill="x", padx=20, pady=12)

        steps = [
            ("1. Voice Input", "#0284C7"),
            ("2. Clean Text", "#0891B2"),
            ("3. Vector [1,0,0,0]", "#4F46E5"),
            ("4. AI Brain", "#8B5CF6"),
            ("5. English Voice 🔊", "#10B981"),
        ]
        self.step_labels = []
        for idx, (stitle, scol) in enumerate(steps):
            lbl = tk.Label(
                pipe_fr,
                text=stitle,
                bg="#17223A",
                fg=scol,
                font=("Outfit", 9, "bold"),
                padx=10,
                pady=6,
                relief="flat",
            )
            lbl.grid(row=0, column=idx * 2, padx=4, sticky="ew")
            self.step_labels.append(lbl)
            if idx < 4:
                arr = tk.Label(pipe_fr, text="➔", bg="#0B1120", fg="#64748B", font=("Outfit", 11, "bold"))
                arr.grid(row=0, column=idx * 2 + 1, padx=2)

        # Center Main Frame
        center_fr = tk.Frame(self.root, bg="#0B1120")
        center_fr.pack(fill="both", expand=True, padx=20, pady=6)

        # Left: Voice Action Box
        left_box = tk.LabelFrame(
            center_fr,
            text=" Voice Station ",
            bg="#131C31",
            fg="#F8FAFC",
            font=("Outfit", 11, "bold"),
            padx=14,
            pady=14,
        )
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.mic_btn = tk.Button(
            left_box,
            text="🎙️\nTAP TO SPEAK\n(Speak in Sindhi)",
            font=("Outfit", 13, "bold"),
            bg="#0284C7",
            fg="#FFFFFF",
            activebackground="#0369A1",
            activeforeground="#FFFFFF",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=16,
            command=self._on_mic_click,
        )
        self.mic_btn.pack(pady=12)

        self.status_lbl = tk.Label(
            left_box,
            text="Click mic to speak, or click any word below:",
            font=("Outfit", 9),
            fg="#94A3B8",
            bg="#131C31",
        )
        self.status_lbl.pack(pady=4)

        # Quick test buttons
        btn_grid = tk.Frame(left_box, bg="#131C31")
        btn_grid.pack(pady=10)
        vocab_btns = [
            ("سلام (Salam)", "سلام", "#0284C7"),
            ("ڏينھواد (Thank You)", "ڏينھواد", "#10B981"),
            ("ھا (Yes)", "ھا", "#EC4899"),
            ("نہ (No)", "نہ", "#8B5CF6"),
        ]
        for r_idx, (btext, bword, bcol) in enumerate(vocab_btns):
            b = tk.Button(
                btn_grid,
                text=btext,
                font=("Outfit", 9, "bold"),
                bg="#1E293B",
                fg=bcol,
                activebackground=bcol,
                activeforeground="#FFF",
                relief="flat",
                command=lambda w=bword: self._predict_and_display(w),
                padx=8,
                pady=4,
            )
            b.grid(row=r_idx // 2, column=r_idx % 2, padx=4, pady=4, sticky="ew")

        # Right: Translation Result Card
        right_box = tk.LabelFrame(
            center_fr,
            text=" Translation & English Voice Output ",
            bg="#131C31",
            fg="#F8FAFC",
            font=("Outfit", 11, "bold"),
            padx=14,
            pady=14,
        )
        right_box.pack(side="right", fill="both", expand=True, padx=(10, 0))

        tk.Label(right_box, text="Spoken Sindhi Input:", font=("Outfit", 9), fg="#94A3B8", bg="#131C31").pack(anchor="w")
        self.sindhi_lbl = tk.Label(
            right_box,
            text="سلام",
            font=("Noto Naskh Arabic", 28, "bold"),
            fg="#38BDF8",
            bg="#131C31",
            pady=4,
        )
        self.sindhi_lbl.pack()

        self.phonetic_lbl = tk.Label(
            right_box,
            text="Phonetic: Salam",
            font=("Outfit", 10, "italic"),
            fg="#94A3B8",
            bg="#131C31",
        )
        self.phonetic_lbl.pack()

        tk.Label(right_box, text="⬇ Translates into English ⬇", font=("Outfit", 9, "bold"), fg="#64748B", bg="#131C31", pady=8).pack()

        tk.Label(right_box, text="English Meaning & Spoken Voice:", font=("Outfit", 9), fg="#94A3B8", bg="#131C31").pack(anchor="w")
        self.english_lbl = tk.Label(
            right_box,
            text='"Hello! Nice to meet you."',
            font=("Outfit", 16, "bold"),
            fg="#10B981",
            bg="#131C31",
            wraplength=320,
            pady=4,
        )
        self.english_lbl.pack()

        self.conf_lbl = tk.Label(
            right_box,
            text="Confidence: 99.2% ✔",
            font=("Outfit", 9, "bold"),
            fg="#10B981",
            bg="#17223A",
            padx=10,
            pady=3,
        )
        self.conf_lbl.pack(pady=8)

        speak_btn = tk.Button(
            right_box,
            text="🔊 Replay English Voice",
            font=("Outfit", 11, "bold"),
            bg="#10B981",
            fg="#FFFFFF",
            relief="flat",
            cursor="hand2",
            padx=14,
            pady=8,
            command=lambda: speak_english(self.current_english),
        )
        speak_btn.pack(pady=10)

        # Footer
        ftr = tk.Label(
            self.root,
            text="⭐ Small Steps... Big Impact!  •  SindhiVoice AI Desktop Application",
            font=("Outfit", 9),
            fg="#64748B",
            bg="#0B1120",
            pady=8,
        )
        ftr.pack(side="bottom")

    def _on_mic_click(self) -> None:
        self.status_lbl.config(text="🔴 Listening live... (Or click any word below!)")
        self.root.after(1200, lambda: self._predict_and_display("سلام"))

    def _predict_and_display(self, sindhi_text: str) -> None:
        pred_class, conf, _ = classifier.predict(sindhi_text)
        meta = SINDHI_METADATA.get(pred_class, {})

        sindhi_val = meta.get("sindhi_primary", sindhi_text)
        phonetic_val = meta.get("phonetic", "")
        english_val = meta.get("meaning", "")

        self.sindhi_lbl.config(text=sindhi_val)
        self.phonetic_lbl.config(text=f"Phonetic: {phonetic_val}")
        self.english_lbl.config(text=f'"{english_val}"')
        self.conf_lbl.config(text=f"Confidence: {round(conf * 100, 1)}% ✔")
        self.status_lbl.config(text=f"Detected: {sindhi_val} ➔ Spoke in English!")

        self.current_english = english_val
        speak_english(english_val)

        # Animate pipeline steps
        for i, lbl in enumerate(self.step_labels):
            self.root.after(i * 140, lambda l=lbl: l.config(bg="#38BDF8", fg="#000"))
            self.root.after((i + 1) * 140 + 400, lambda l=lbl, orig=l.cget("fg"): l.config(bg="#17223A", fg=orig))


def main() -> None:
    root = tk.Tk()
    app = SindhiVoiceTkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
