# LinkedIn post (draft)

I spent my final year building SafeGuard: an AI surveillance console plus a public fraud check, from scratch.

In 4 minutes you can watch it:
- spot violence and weapons live across several cameras (and save a video clip around every alert)
- open a one-click PDF evidence report with a SHA-256 of the snapshot
- search a face against a watchlist and open the full case file
- check any business or website against a complaints register, no login needed

Two quizzes in the video: can you beat the AI to the incident, and can you tell which website is unsafe?

I try to measure instead of boast. On 400 held-out clips the violence model scores 79% recall and 81% precision; the README lists where it fails. Footage is free-licence stock, faces are blurred, every record is fictional.

Built with Flask, OpenCV, YOLO, scikit-learn and SQLite, with 83 automated tests.
Code: https://github.com/abdurrafaykhann/SafeGuard-for-smart-crime-detection

Which feature would you build next?

#ComputerVision #AI #Python #Flask #FinalYearProject #MachineLearning #Cybersecurity #UniversityOfHertfordshire

## Chapters (paste as the first comment)
0:00 Spot the incident (quiz)
0:30 Live wall + incident feed
0:55 Alerts with video evidence + PDF report
1:34 Weapon detection
1:55 Analytics
2:03 Video analysis
2:15 Face search + case files
2:35 Public fraud check
3:27 Security and measured accuracy

## Upload notes
- Video: `safeguard-demo.mp4` (1280x720, H.264, about 4 minutes, captions are burned in; `safeguard-demo.srt` is also provided for LinkedIn's caption upload).
- Thumbnail: `thumbnail.png`.
- Most people watch muted, so the on-screen captions carry the story.
- Keep the honesty line in the post: the footage is staged stock, and the numbers quoted are measured on validation data.
