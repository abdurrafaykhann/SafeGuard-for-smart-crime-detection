"""Narration script for the demo video and offline text-to-speech with the Windows voices (no downloads, no network)."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

VOICE = "Microsoft Zira Desktop"

# key -> spoken text. Keep sentences short: they are also shown as captions.
LINES: dict[str, str] = {
    "hook": "Three cameras. One of them has an incident. Can you spot it before the AI does?",
    "hook_reveal": "The gym camera. SafeGuard flagged it automatically, with nobody watching.",
    "intro": "This is SafeGuard, a real-time AI surveillance console I built for my final year project at the University of Hertfordshire. Here is everything it does.",
    "live": "Every camera is captured once and analysed once, however many people are watching. A trained classifier scores five second windows of motion, and a separate model hunts for weapons.",
    "feed": "When something fires, the incident feed updates instantly, and the alert shows why it fired, not just a score.",
    "alert": "Every alert keeps a snapshot, a short video from before and after, and the cues behind the decision.",
    "ack": "An operator acknowledges it, and one click exports an evidence report: the snapshot hash, the reasoning, and who handled it.",
    "weapon": "The weapon detector watches the same streams. Here it catches a rifle, then a knife.",
    "analytics": "Analytics show alert trends, busiest hours, hot spot cameras and how quickly operators respond.",
    "video": "You can also upload a recorded video and let SafeGuard scan it.",
    "video_result": "It returns an incident timeline, the peak risk and evidence frames.",
    "faces": "Face search ranks a photo against the enrolled watchlist and opens the full case file. Faces are blurred in this demo, and every record is fictional.",
    "faces_acc": "On a fifty four person test, the right person ranked first ninety eight percent of the time, with each photo held out of the gallery.",
    "quiz_fraud": "Quick quiz. Is this website safe to pay?",
    "fraud_reveal": "Not safe. Thirteen complaints are on record. Anyone can check a business or website here, with no login.",
    "fraud_caution": "A near miss spelling gets a careful warning. An unknown name says no complaints found, which is not the same as safe.",
    "fraud_report": "The public can report a business too, but a tip only counts after a staff member approves it.",
    "fraud_after": "Approved, and the check now flags it.",
    "tech": "Under the hood: hashed passwords, CSRF protection, rate limits, audit logging, automatic data retention, and eighty three automated tests.",
    "honest": "And I measure honestly. The violence model scores seventy nine percent recall and eighty one percent precision on four hundred held out clips, and the README documents where it fails.",
    "outro": "That is SafeGuard. The code and documentation are on GitHub. Which feature would you build next? Tell me in the comments.",
}


def synth(text: str, wav: Path, voice: str = VOICE, rate: int = 0) -> float:
    """Render text to a WAV with the built-in Windows speech engine; returns the duration in seconds."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    txt = wav.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); $s.Rate = {rate}; $s.SetOutputToWaveFile('{wav}'); "
          f"$s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8)); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
    txt.unlink(missing_ok=True)
    return duration(wav)


def duration(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def synth_all(out_dir: Path, rate: int = 0) -> dict[str, float]:
    return {k: synth(t, out_dir / f"{k}.wav", rate=rate) for k, t in LINES.items()}


if __name__ == "__main__":
    import sys
    d = synth_all(Path(sys.argv[1]))
    for k, v in d.items():
        print(f"{k:14} {v:5.1f}s  {LINES[k][:60]}")
    print("total", round(sum(d.values()), 1), "s")
