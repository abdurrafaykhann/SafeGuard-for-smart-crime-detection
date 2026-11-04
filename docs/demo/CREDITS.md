# Demo video: sources and notes

**Footage** (free to use under the [Pexels License](https://www.pexels.com/license/); no attribution required, listed for transparency):

| Used for | Clip page |
|---|---|
| Calm cameras (Lobby, Car park) and the calm lead-in | https://www.pexels.com/video/couple-walking-together-on-street-6307658/ |
| Gym camera incident and the video-analysis scene | https://www.pexels.com/video/boxers-sparring-in-the-gym-6296278/ |
| Entrance camera, rifle | https://www.pexels.com/video/a-man-holding-a-gun-6092110/ |
| Entrance camera, knife | https://www.pexels.com/video/knife-held-in-yellow-raincoat-outdoors-30166089/ |
| Entrance camera, knife | https://www.pexels.com/video/camera-focus-video-of-a-man-holding-a-knife-blade-5735079/ |

The clips are staged or stock footage (boxing practice, prop weapons). They are shown only to demonstrate the software. Nothing in the video says or implies that any person shown did anything wrong.

**Faces:** every face image in the face-search scene is blurred during recording. All names, case numbers and offences in the demo are invented.

**Honest note on accuracy:** the clips in the video were chosen from 23 trial clips. On those 23 free-licence clips the violence classifier fired on 7 of 8 sparring clips but also on 5 of 6 calm street clips, and the weapon model raised alerts on 7 of 9 weapon clips, 5 of 8 sparring clips and 1 of 6 calm clips (raw numbers: `docs/evaluation_stock_clips.json`). The video therefore shows examples of the system working, not a measure of its accuracy; the validation numbers on the closing card (79% recall, 81% precision on 400 held-out RWF-2000 clips) are the measured ones.

**Voice:** Microsoft "Zira" text-to-speech (offline). Captions are burned in; `safeguard-demo.srt` has them as a separate file.

**Reproduce:** `python scripts/demo_video/make_demo.py` (needs ffmpeg, Chrome and `playwright`; see the script header).
