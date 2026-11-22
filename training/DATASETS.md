# Near-eye datasets for pre-training (checked 2026-10-15)

Licences decide what we may do: everything here is for PRIVATE experiments only (personal use).
Check each dataset's own licence before using it; no dataset files or models trained on them are included here.

| dataset | what | size | access | licence | fit to our cameras |
|---|---|---|---|---|---|
| **VRGaze** (GazeShift, CVPR 2026) | 2.1 M off-axis near-eye IR images, 68 people, gaze labels; per-subject `train/idNNN.tar.gz` (webdataset) + metadata CSVs | **68.5 GB train (61 subjects) + 8.3 GB val (7)** | open on Hugging Face `gazeshift/VRGaze`, not gated | **none stated** (= all rights reserved by default; private research use only) | **best**: off-axis, IR, LED glints, VR geometry. Camera below the eye (ours: temple side) |
| OpenEDS2020 (Meta) | 550 k eye images with gaze vectors (gaze-prediction track) + a segmentation subset; 80-87 people; HMD IR | not checked | on request via Facebook Research's OpenEDS challenge page | research terms (text not read yet) | good IR + HMD, but ON-axis camera |
| OpenEDS2019 (Meta) | 200 Hz HMD IR eye images; eyelid/iris/pupil segmentation labels | not checked | on request | research terms (not read) | lids/segmentation; on-axis |
| NVGaze (NVIDIA, CHI 2019) | ~2 M synthetic (1280×960) + 2.5 M real (640×480) near-eye IR, 35 real subjects; 2.06° reported | not checked | project page sites.google.com/nvidia.com/nvgaze needs a Google login | not read yet | synthetic part can be rendered to our angle? |
| TEyeD (Tübingen, 2021) | 20 M+ head-mounted eye-tracker images: pupil/iris/eyelid segmentation, landmarks, 3D eyeball, gaze, movement types | `TEyeDSComplete.7z.001…` in 4.29 GB parts (very large) | Uni Tübingen cloud (es-cloud.cs.uni-tuebingen.de, share `8e2ab8c3fdd444e1a135`), TUM mirror hctlsrva.edu.sot.tum.de/TEyeDS | **none stated**: checked 2026-10-22, no licence/terms file on either server, none in the paper (same status as VRGaze: private use only) | eye-tracker glasses, not VR; useful for lids + robustness |
| **GIW + TEyeD labels** (`jpena-173/sm-eyes-v2`) | Gaze-in-Wild Pupil Labs IR eye videos (640×480, 120 Hz) with TEyeD's per-frame labels: 34 eyelid landmarks, pupil + iris ellipses and landmarks, 3D eyeball, gaze vector, eye-movement type (blink/fixation/saccade/pursuit), validity flags. Every 3rd frame, 202,833 rows, 6 subjects × 2 eyes | **4.1 GB** (12 parquet shards) | open on Hugging Face, not gated. `prep/fetch_giw.py` → `data/giw/` | HF tag "MIT" is the UPLOADER's, not authoritative. Originals: GIW project page tags CC0/MIT, "shared in good faith for academic researchers" (page now 404; read via search 2026-10-22); TEyeD labels: none stated → private use only | glasses camera from below-front, no VR lens/glints; **fills the gap VRGaze leaves: lid labels + blinks** |

### 🔴 GIW (`sm-eyes-v2`): only 3 of 12 streams have the RIGHT labels (checked 2026-10-22)
Downloaded to `data/giw/` (4.1 GB). `prep/explore_giw.py` draws labels on frames (`data/giw_explore/sheet_*.jpg`);
`prep/giw_align_check.py` measures a dark-blob pupil detector against the label pupil on 60 random fixation frames per
stream. Median distance in px (640×480):

| shard | subject/eye | own labels | other eye's labels | within 15 px | verdict |
|---|---|---|---|---|---|
| 04 | 2 / R | 9.1 | 142.8 | 90% | ✅ use |
| 06 | 3 / R | 4.3 | 111.2 | 88% | ✅ use |
| 12 | 6 / R | 7.4 | 58.5 | 77% | ✅ use |
| 08 | 4 / R | 13.5 (7 → 16 drifting) | 91.8 | 57% | 🟡 maybe, re-check |
| 01 02 03 05 07 09 10 11 | rest | 93-400 | 58-340 | 0-2% | ❌ labels belong to another recording |

- The subject mapping itself is right (TEyeD's `file_assignementsGIW.txt`, saved in `data/giw/teyed/`: GIW 2 = GWNC_12
  etc.). The bad streams are the wrong video of that subject, or the wrong session, paired with the labels. Swapping
  eyes doesn't fix it, and the error doesn't grow over time, so it isn't frame drift either.
- Good streams look excellent on the contact sheet: lid curves and pupil ellipses sit right on the eye. Closed eye ≈ 0.20
  openness (corner-based gap/width), open ≈ 0.40-0.48.
- `eye_movement == blink` covers whole blink EPISODES (some frames are open), so take closure from the lid landmarks.
- Pupil w/h = −1 when there is no pupil (29% of blink frames).
- Usable = ~46 k frames, 3 people, right eye only: small. Good for a LID/openness head + pupil-ellipse experiments,
  not for gaze. Repairing the other 8 would need the matching labels from the full TEyeD archive (many 4.29 GB 7z parts).
- (Side note: Shinra Meisin's eye model trains on this set as-is, i.e. mostly on wrong labels.)

Skipped on purpose (2026-10-22): `jpena-173/sm-eyes-v1` (200 k synthetic UnityEyes-style renders, 5.1 GB, no licence,
visible-light eye-region renders, not near-eye IR). The Shinra Meisin training code that uses both sets
(github.com/Porudachi/shinra-meisin) is "All rights reserved, source-available" → read for ideas only, never copied here.

## Next checks
- Read the actual licence/terms for OpenEDS, NVGaze, TEyeD before downloading them.
- VRGaze metadata: which CSV holds the gaze labels and in what units (`metadata_train.csv` 76 MB,
  `metadata_train_features.csv` 531 MB, `*_static_points.csv`). Read the GazeShift code (`repos/gazeshift/config.py`)
  for the loader.
