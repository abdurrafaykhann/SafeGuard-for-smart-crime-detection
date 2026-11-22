# LinkedIn post: SafeGuard

Post this as a multi-image post (up to 20 images; the numbers below are the file names in `docs/linkedin/screens/`). Add the 2-minute
product video (`safeguard/static/demo/safeguard-product-demo.mp4`, captions in the `.vtt` file) as the first item if you prefer a video post.
Screenshots are git-ignored; regenerate them any time with `python scripts/demo_video/make_screenshots.py`.

---

## Post text (copy from here)

I spent my final year at the University of Hertfordshire building SafeGuard: a real-time video surveillance console with violence and weapon detection, face search with case files, and a public fraud check anyone can use.

You can try the demo yourself, no sign-in: https://safeguard-demo-puce.vercel.app/demo

What it does:
🎥 Watches several camera streams and flags violence and weapons, with a short video clip saved around every alert
🧾 One-click PDF evidence report with the snapshot's SHA-256 and the reasoning behind the alert
🔎 Face search against a consent-based watchlist, opening the full case file
🛡️ A public fraud check: type a business or website and see whether complaints are on record
👤 A human decision on every alert. Nothing happens automatically on a match.

What I'm proudest of is the measuring, not just the building:
• My clip-level violence classifier (trained on RWF-2000): 79% recall and 81% precision on 400 held-out clips
• Face matching tested beyond the easy case: with 1,000 people enrolled, my first threshold wrongly matched about 20% of strangers. I re-tuned it, and it now matches 0.3% of strangers
• I also tested for bias and found women were wrongly matched about 3x as often as men, so the README says so and recommends human review
• Failure cases are documented too: phones read as pistols, low-light misses, false alarms on some calm street footage

A note on the demo: it replays the real detectors' recorded output on five free-licence stock clips, picked because the system works on them. It's a showcase, not a benchmark. Faces are blurred and every record is fictional.

Built with Flask, OpenCV, YOLO, scikit-learn and SQLite, with automated tests. I used pretrained community models for weapon and face recognition (credited in the README) and trained the violence classifier myself.

Code and full evaluation: https://github.com/abdurrafaykhann/SafeGuard-for-smart-crime-detection

Which feature would you add next?

#ComputerVision #AI #Python #Flask #MachineLearning #FinalYearProject #UniversityOfHertfordshire #SoftwareEngineering #ResponsibleAI

---

## Image order and alt text

| # | File | Alt text (paste into LinkedIn) |
|---|---|---|
| 1 | `01-home-hero` | SafeGuard landing page: "Eyes on every frame. Judgement stays human." |
| 2 | `04-demo-fight` | Interactive demo replaying a sparring clip: risk chart rising and a violence alert in the incident feed |
| 3 | `05-demo-knife` | Weapon detector boxing a knife with 86% confidence and raising a critical alert |
| 4 | `06-demo-gun` | Pistol detected in a car park, with a weapon alert followed by a violence alert |
| 5 | `07-demo-rifle` | Rifle detected and tracked in the demo |
| 6 | `08-demo-normal` | Ordinary activity: no weapon and no violence, the feed stays quiet |
| 7 | `10-console-live-wall` | Operator console live wall with several camera tiles and the incident feed |
| 8 | `11-console-alerts` | Alerts list with severity, camera and acknowledge action |
| 9 | `12-console-analytics` | Analytics: alerts per day, by type, busiest hours and hot-spot cameras |
| 10 | `13-console-video-analysis` | Uploaded video analysed into an incident timeline |
| 11 | `14-console-watchlist` | Consent-based watchlist (faces blurred, fictional records) |
| 12 | `15-console-case-file` | Case file for one enrolled person (face blurred, fictional record) |
| 13 | `09-fraud-check-unsafe` | Public fraud check flagging a business as unsafe |
| 14 | `16-console-complaints` | Staff complaints register that feeds the fraud check |
| 15 | `17-console-cameras` | Admin cameras page |
| 16 | `03-home-responsible-ai` | Responsible-by-design principles: human in the loop, consent, data minimisation, honest limits |
| 17 | `19-mobile-demo` | The demo on a phone |
| 18 | `18-mobile-home` | The landing page on a phone |
| 19 | `02-home-capabilities` | Capabilities overview |

Tip: LinkedIn shows the first 3 to 4 images in the feed, so lead with 1, 2, 3 (the landing page, a violence alert, a weapon alert).

## First comment (paste after posting)
Live demo (no sign-in): https://safeguard-demo-puce.vercel.app/demo
2-minute narrated tour is embedded at the bottom of that page. Measured results, including the failures, are in the repo README.

## Before you post
- Re-check every number against `docs/evaluation_*.json` if you change the wording.
- Don't call it "built entirely from scratch": the violence classifier is yours, the weapon and face models are pretrained and credited.
- The live site is a replay plus the fraud check; real-time cameras only run when you host it yourself.
