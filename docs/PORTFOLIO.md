# SafeGuard: portfolio and CV wording

Everything below is checked against the repository. Items in [square brackets] are details only you know: fill them in or delete the sentence.

## One-line summary
AI surveillance console with violence and weapon detection, face search with case files, and a public fraud-check page. Flask, OpenCV, YOLO, scikit-learn.

## What I trained, and what I integrated
| Component | Status | How to say it |
|---|---|---|
| Clip-level violence classifier | Trained by me | "Trained on RWF-2000 (group split, held-out validation of 400 clips): 79% recall, 81% precision under the live streaming rule." |
| YOLO violence detector (`Yolo_nano_weights.pt`) | Third-party pretrained model by Musawer1214, **not trained by me** | "Uses a pretrained community YOLOv8n violence detector (credited in the README) as one cue; the decision is made by my own clip-level classifier." |
| Closed-set face CNN (FYP) | Trained by me | "Custom CNN for 56-person closed-set face classification in my FYP (89.3% accuracy on 302 samples, from the report)." |
| Weapon detector | Pretrained community model (ONNX) | "Integrated and evaluated a pretrained weapon detector: works at normal distance, weak on close-up webcam knives (about 9% recall), documented with measurements." |
| Face detection and recognition | Pretrained OpenCV models (YuNet, SFace) | "Built the identification pipeline and measured it, including the failure that matters: 97.7% top-1 on a 54-person test, but with 1,000 people enrolled the original threshold matched about 20% of strangers, so I re-tuned it (0.31% at 0.50)." |

## CV bullets (pick 4 to 6)
- Built SafeGuard, a Flask surveillance console with live multi-camera analysis, alerts with before/after video evidence, and one-click PDF evidence reports (snapshot SHA-256, cue breakdown, handling log).
- Trained a clip-level violence classifier on RWF-2000 from optical-flow, YOLO and colour features; evaluated on 400 held-out clips (79% recall, 81% precision) and documented its failure modes instead of reporting only the best number.
- Integrated and evaluated pretrained weapon and face-recognition models; built face search that returns a person's full case file. Tested it beyond the easy case: on 1,680 people from LFW, strangers wrongly matched rose from 2% (100 enrolled) to 20% (1,000 enrolled) at the original threshold, so I raised the default to cut that to 0.3%, and showed brute-force search (0.02 ms at 1,000 faces) beats FAISS until the gallery is far bigger.
- Cut server CPU from about 1000% to about 200% (thread caps, frame cadence, background model warm-up); added a corrupted-frame guard after a glitching webcam caused false weapon alerts.
- Hardened the app: scrypt password hashing, CSRF protection, rate limiting, security headers, audit log, automatic data retention, consent-based watchlist enrolment, ONNX-over-pickle model loading.
- Built a public fraud-check page (business/website name against a complaints register) with staff-reviewed public reports; 83 automated tests.

## Portfolio paragraph
SafeGuard started as my final-year project and became a full surveillance console. It watches several camera streams, flags violent incidents and weapons, and keeps a short video, a snapshot and the reasoning behind every alert. Operators can acknowledge alerts and export an evidence report. Face search ranks a photo against an enrolled watchlist and opens the case file. A separate public page lets anyone check a business or website against complaints on record. I trained the clip-level violence classifier myself, integrated pretrained YOLO (violence, weapons) and face models with credit to their authors, and measured all of them on held-out data. The README documents where they fail, for example false alarms on close-range webcams and on some stock street footage, because I wanted the numbers to be honest.

## Questions to be ready for
1. **Why a clip-level classifier, not frame-by-frame?** Violence is about motion over time; a single frame cannot tell a fight from a handshake. It scores 5-second windows sampled at 4 per second and needs two consecutive high scores to alert.
2. **Why is the threshold 0.46?** Chosen on training/cross-validation data, then validation scored once, with the extra looks disclosed. A threshold of 0.50 trades recall for fewer false alarms; it is configurable per site.
3. **How did you avoid leakage?** Splits are by group (source video/photo), so near-duplicate frames never sit on both sides.
4. **Where does it fail?** Close-up webcams, camera shake, and footage unlike RWF-2000. On 23 stock clips it fired on 5 of 6 calm street clips; the demo video shows examples, not an accuracy claim.
5. **Did you train the weapon model?** No, it is a pretrained community model that I evaluated and integrated; my Colab training attempt was blocked. [If you later fine-tune it, update this answer.]
6. **How do you handle privacy?** Consent basis required for every watchlist entry, tight face crops only, retention policy, audit log, fictional demo data, faces blurred in the video.
7. **What happens when someone who is not in the database walks past?** I measured it. At the original 0.40 threshold, 3% of strangers were matched against 54 people and about 20% against 1,000. I raised the default to 0.50 (0.31% at 1,000 people, at the cost of rejecting about 6% of enrolled faces) and documented that the right threshold depends on watchlist size.
8. **Did you need FAISS?** No. Brute-force numpy takes 0.02 ms at 1,000 vectors and 27 ms at a million; FAISS flat was not faster. An approximate index only pays off well beyond that.
9. **What would you do next?** Fine-tune the weapon detector on your own data; retrain the violence model on CCTV-style footage; deploy a hosted demo.
