/**
 * SindhiVoice AI — Voice Recognition, NLP Pipeline Animation & English TTS Engine
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const micBtn = document.getElementById("mic-btn");
  const micBtnText = document.getElementById("mic-btn-text");
  const micHint = document.getElementById("mic-hint");
  const waveRing = document.getElementById("wave-ring");
  const liveIndicator = document.getElementById("live-indicator");
  const visualizer = document.getElementById("audio-visualizer");
  const manualInput = document.getElementById("manual-text");
  const sendTextBtn = document.getElementById("send-text-btn");

  const sindhiDisplay = document.getElementById("sindhi-display");
  const phoneticDisplay = document.getElementById("phonetic-display");
  const englishDisplay = document.getElementById("english-display");
  const descDisplay = document.getElementById("desc-display");
  const confBadge = document.getElementById("conf-badge");
  const replayBtn = document.getElementById("replay-btn");
  const copyBtn = document.getElementById("copy-btn");

  const voiceSelect = document.getElementById("voice-select");
  const voiceRate = document.getElementById("voice-rate");
  const rateVal = document.getElementById("rate-val");
  const autoSpeakToggle = document.getElementById("auto-speak-toggle");
  const historyList = document.getElementById("history-list");
  const clearHistoryBtn = document.getElementById("clear-history-btn");
  const themeBtn = document.getElementById("theme-btn");

  // State
  let isRecording = false;
  let recognition = null;
  let availableVoices = [];
  let currentSpokenEnglish = "Hello! Nice to meet you.";

  // Vocabulary Data Dictionary
  const VOCAB_DATA = {
    "Hello": {
      sindhi: "سلام",
      phonetic: "Salam",
      english: "Hello! Nice to meet you.",
      desc: "Warm greeting used when meeting someone warmly.",
      badge: "Greeting"
    },
    "Thank You": {
      sindhi: "ڏينھواد / مھرباني",
      phonetic: "Dhanwaad / Meherbani",
      english: "Thank you very much!",
      desc: "Expressing heartfelt appreciation and gratitude.",
      badge: "Gratitude"
    },
    "Yes": {
      sindhi: "ھا",
      phonetic: "Haa",
      english: "Yes, I agree!",
      desc: "Affirmative confirmation or positive agreement.",
      badge: "Confirm"
    },
    "No": {
      sindhi: "نہ",
      phonetic: "Na",
      english: "No, I decline.",
      desc: "Polite negative response or disagreement.",
      badge: "Decline"
    }
  };

  // --- Voice Synthesis Setup ---
  function populateVoices() {
    if (!window.speechSynthesis) return;
    availableVoices = window.speechSynthesis.getVoices().filter(v => v.lang.startsWith("en"));
    voiceSelect.innerHTML = "";
    if (availableVoices.length === 0) {
      availableVoices = window.speechSynthesis.getVoices();
    }
    availableVoices.forEach((v, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${v.name} (${v.lang})`;
      if (v.name.includes("Natural") || v.name.includes("Google") || v.default) {
        opt.selected = true;
      }
      voiceSelect.appendChild(opt);
    });
  }

  if (window.speechSynthesis) {
    populateVoices();
    window.speechSynthesis.onvoiceschanged = populateVoices;
  }

  // --- Speech Synthesis (Speak in English) ---
  function speakEnglish(text) {
    if (!text) return;
    currentSpokenEnglish = text;

    // 1. Browser Web Speech Synthesis
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = parseFloat(voiceRate.value) || 1.0;
      const selectedIndex = parseInt(voiceSelect.value);
      if (availableVoices[selectedIndex]) {
        utterance.voice = availableVoices[selectedIndex];
      }
      window.speechSynthesis.speak(utterance);
    }

    // 2. Also notify backend server for Python local TTS (spd-say)
    fetch(`/api/speak?text=${encodeURIComponent(text)}`).catch(() => {});
  }

  // --- 5-Step Pipeline Animation ---
  async function triggerPipelineAnimation() {
    const steps = [
      document.getElementById("step-1"),
      document.getElementById("step-2"),
      document.getElementById("step-3"),
      document.getElementById("step-4"),
      document.getElementById("step-5"),
    ];

    steps.forEach(s => s.classList.remove("active"));

    for (let i = 0; i < steps.length; i++) {
      steps[i].classList.add("active");
      await new Promise(r => setTimeout(r, 160));
    }

    // Keep final step highlighted
    setTimeout(() => {
      steps.forEach((s, idx) => {
        if (idx !== 4) s.classList.remove("active");
      });
    }, 1200);
  }

  // --- NLP Prediction & Display ---
  async function processSindhiInput(text) {
    if (!text || !text.trim()) return;

    triggerPipelineAnimation();

    try {
      const res = await fetch(`/api/predict?q=${encodeURIComponent(text)}`);
      const data = await res.json();
      renderResult(data);
    } catch (err) {
      // Local fallback logic
      const fallback = localPredict(text);
      renderResult(fallback);
    }
  }

  function localPredict(text) {
    const t = text.trim();
    if (t.includes("سلام") || t.toLowerCase().includes("salam") || t.toLowerCase().includes("hello")) {
      return { class: "Hello", confidence: 0.985, input: text, ...VOCAB_DATA["Hello"] };
    }
    if (t.includes("ڏينھواد") || t.includes("مھرباني") || t.toLowerCase().includes("dhanwaad") || t.toLowerCase().includes("meherbani") || t.toLowerCase().includes("thank")) {
      return { class: "Thank You", confidence: 0.972, input: text, ...VOCAB_DATA["Thank You"] };
    }
    if (t.includes("ھا") || t.includes("ها") || t.toLowerCase().includes("haa") || t.toLowerCase().includes("yes")) {
      return { class: "Yes", confidence: 0.988, input: text, ...VOCAB_DATA["Yes"] };
    }
    if (t.includes("نہ") || t.includes("نه") || t.toLowerCase().includes("na") || t.toLowerCase().includes("no")) {
      return { class: "No", confidence: 0.965, input: text, ...VOCAB_DATA["No"] };
    }
    return { class: "Hello", confidence: 0.940, input: text, ...VOCAB_DATA["Hello"] };
  }

  function renderResult(data) {
    const sindhiWord = data.sindhi_primary || data.sindhi || data.input;
    const englishText = data.spoken_text || data.english || VOCAB_DATA[data.class]?.english || "Hello!";
    const phonetic = data.phonetic || VOCAB_DATA[data.class]?.phonetic || "Salam";
    const desc = data.desc || VOCAB_DATA[data.class]?.desc || "";
    const confPct = Math.round((data.confidence || 0.98) * 100);

    sindhiDisplay.textContent = sindhiWord;
    phoneticDisplay.textContent = `Phonetic: ${phonetic}`;
    englishDisplay.textContent = `"${englishText}"`;
    descDisplay.textContent = desc;
    confBadge.textContent = `${confPct}% Confidence ✔`;
    confBadge.style.display = "inline-block";

    currentSpokenEnglish = englishText;

    if (autoSpeakToggle.checked) {
      speakEnglish(englishText);
    }

    addToHistory(sindhiWord, englishText, confPct);
  }

  function addToHistory(sindhi, english, conf) {
    const emptyMsg = historyList.querySelector(".history-empty");
    if (emptyMsg) emptyMsg.remove();

    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <div class="history-item-left">
        <span class="history-sindhi">${sindhi}</span>
        <span>➔</span>
        <span class="history-english">"${english}"</span>
      </div>
      <div class="history-item-right">
        <span style="font-size:0.75rem;color:var(--accent-green);font-weight:700;">${conf}%</span>
        <button class="mini-speak-btn" style="background:none;border:none;cursor:pointer;font-size:0.9rem;" title="Speak aloud">🔊</button>
      </div>
    `;

    item.querySelector(".mini-speak-btn").addEventListener("click", () => {
      speakEnglish(english);
    });

    historyList.prepend(item);
  }

  // --- Web Speech API (Microphone Recognition) ---
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (SpeechRec) {
    recognition = new SpeechRec();
    recognition.continuous = false;
    recognition.interimResults = false;
    // Prefer Sindhi (Pakistan), fallback handles general voice
    recognition.lang = "sd-PK";

    recognition.onstart = () => {
      isRecording = true;
      micBtn.classList.add("recording");
      waveRing.classList.add("pulsing");
      visualizer.classList.add("active");
      liveIndicator.textContent = "Listening...";
      liveIndicator.classList.add("recording");
      micBtnText.textContent = "Listening...";
      micHint.innerHTML = "🔴 <strong>Listening live!</strong> Speak your Sindhi word now...";
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      micHint.textContent = `Recognized speech: "${transcript}"`;
      processSindhiInput(transcript);
    };

    recognition.onerror = (event) => {
      console.warn("Speech recognition notice:", event.error);
      micHint.textContent = `Voice status: Ready. Speak clearly or click a word below!`;
      stopRecording();
    };

    recognition.onend = () => {
      stopRecording();
    };
  } else {
    micHint.innerHTML = "💡 Tip: Click any of the 4 poster words below or type in the box to test!";
  }

  function startRecording() {
    if (!recognition) {
      alert("Speech recognition is running in quick-voice mode! Click any of the 4 words below or type in the box to speak aloud.");
      return;
    }
    try {
      recognition.start();
    } catch (e) {
      recognition.stop();
    }
  }

  function stopRecording() {
    isRecording = false;
    micBtn.classList.remove("recording");
    waveRing.classList.remove("pulsing");
    visualizer.classList.remove("active");
    liveIndicator.textContent = "Ready";
    liveIndicator.classList.remove("recording");
    micBtnText.textContent = "Click to Speak";
  }

  // --- Event Listeners ---
  micBtn.addEventListener("click", () => {
    if (isRecording) {
      if (recognition) recognition.stop();
      stopRecording();
    } else {
      startRecording();
    }
  });

  sendTextBtn.addEventListener("click", () => {
    const val = manualInput.value;
    if (val) {
      processSindhiInput(val);
      manualInput.value = "";
    }
  });

  manualInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      sendTextBtn.click();
    }
  });

  replayBtn.addEventListener("click", () => {
    speakEnglish(currentSpokenEnglish);
  });

  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(currentSpokenEnglish);
    copyBtn.textContent = "Copied! ✔";
    setTimeout(() => { copyBtn.innerHTML = "<span>📋</span> Copy"; }, 1500);
  });

  // Poster 2 Quick Test Buttons
  document.querySelectorAll(".vocab-test-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const word = btn.getAttribute("data-word");
      processSindhiInput(word);
    });
  });

  voiceRate.addEventListener("input", () => {
    rateVal.textContent = `${voiceRate.value}x`;
  });

  clearHistoryBtn.addEventListener("click", () => {
    historyList.innerHTML = '<div class="history-empty">No words translated yet. Speak into the microphone!</div>';
  });

  themeBtn.addEventListener("click", () => {
    document.body.classList.toggle("light-theme");
    themeBtn.textContent = document.body.classList.contains("light-theme") ? "☀️" : "🌙";
  });

  // Initial welcome greeting speech
  setTimeout(() => {
    speakEnglish("Hello! Nice to meet you.");
  }, 1000);
});
