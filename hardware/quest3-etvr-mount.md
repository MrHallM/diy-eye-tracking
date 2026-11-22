# Quest 3 EyeTrackVR mount (chosen 2026-10-15)

Source: thingiverse **6439510**, "Quest 3 eyetrackvr Mount for v4 hardware" by BasilAldwin (+ Dentohn's ring),
**CC BY-SA**. The only ★ Quest 3 mount in the EyeTrackVR docs. Downloaded 10-15 via Chrome (Thingiverse 403s
curl/WebFetch) → `repos/quest3-etvr-mount/` (not in git; the zip is in Downloads).

## Files (author's description, read 10-15)
| file | what | mesh check (trimesh) |
|---|---|---|
| `Quest_eye_track_mount_v13_Left_Repaired.stl` / `_Right_` | **BasilAldwin's ring for 130° cameras** = what we bought | 52.3 × 52.8 × 10.8 mm, watertight, 1 body, 1.91 cm³ (~2.4 g) each |
| `Eye_Tracking_Ring_V21-B.stl` | Dentohn's ring, **160° cameras**, pressure-fit | broken: 13 bodies, 34 open edges; trimesh repair didn't close it. Not needed |
| `esp_Left.stl` / `esp_Right.stl` | XIAO holders for the default facial interface (orig. SoughtJungle) | 25.1 × 21.9 × 15.9 mm, watertight |
| `V4Lite.stl` | case for the V4.1 Lite LED board | watertight, but commenters say it is warped ("chewed by a dog"), so don't trust it |

## Layout (render + photo in `images/`)
Ring sits around each lens (moves with the IPD slider). Camera on the OUTER (temple) side at mid-height, angled in;
4 LED boards around the ring, 2 upper-outer and 2 lower-outer, with wires along the ring.

## Comments worth knowing (26, read 10-15)
- v13 "has minor errors and holes" but printed fine. Our check: the `_Repaired` files are watertight.
- V21-B "removes parts when slicing" (Arachne wall generator in Bambu Studio helps); one JLC print came out undersized.
- Dentohn's V21-B puts the camera high, so eyelashes block the view. Another reason to use v13.
- Author: 40+ failed prints / 13 iterations; PLA works, PETG recommended; mirror for the other side if needed.

## Camera aim, measured from the v13 STLs (2026-10-23, trimesh)
Ring lies flat in the file: z = 0 is the lens-side face, z = 10.8 the eye-side top. Left and right are exact mirrors.
- Lens bore: Ø ≈ 40.1 mm on the lens side (z 1.5-2.5), Ø ≈ 44.9 mm on the eye side (z 6). Centre (±37.3, −40.2) in file coords.
- Camera seat: 8.9 × 8.7 mm flat (fits the 8.0 mm module), centre 22.4 mm off the lens axis, at z ≈ 6.
- Seat tilt: **36°** from the lens axis, facing straight in toward it (misses the axis by only 1.6 mm).
- ⇒ The camera's centre line crosses the lens axis **≈ 36.5 mm from the ring's lens-side face** (≈ 26 mm past its top), **≈ 38 mm from the camera**.
- Reading: the aim point is probably near the eyeball's ROTATION CENTRE (≈ 13 mm behind the cornea), which would put the cornea ≈ 20-25 mm from the ring base. That's inference: the designer didn't publish an eye relief, and the camera may not sit flush on the seat. Treat ~38 mm as the starting focus distance and find the real eye position from the image (pupil centred at straight-ahead gaze).

## IR-filter removal + focus (researched 2026-10-24, not done yet)
- ETVR docs call the 130° OV2640 "experimental": you remove the IR filter by hand, it may break the camera, the filter SHATTERS → eye protection.
- Where the filter sits varies by module: on some it's on the FRONT of the lens (cut away the front of the plastic moulding, lens stays in); on others it's cemented to the front of the SENSOR (lens out, pry it off with a jeweller's screwdriver). Look at OUR module before choosing.
- Focus: fixed-focus threaded barrel, usually glued. One user (AI-on-the-Edge docs, a different OV2640 lens): stock focus is sharp beyond ~40 cm; break the glue, and ~¼ turn moves the focus plane ~10 cm. Our target is ~38 mm (see camera aim above), so expect several turns; verify with `tools/focushelper.py` on a fixed target at that distance. Mind the ribbon connector while turning.
- No step-by-step ETVR guide found; the docs link a YouTube video that wasn't checked.
- Sources: docs.eyetrackvr.dev/how_to_build/parts_list; forum.arduino.cc threads 1113413 + 1150710; jomjol.github.io/AI-on-the-edge-device-docs/Reference-Image/.

## Desk test rig (planned 10-23, no headset yet)
Chin + forehead rest holding both v13 rings at IPD 64 on Ø40 bosses (stand-ins for the Quest 3 lenses), with eye relief ADJUSTABLE in a slot (~15-40 mm cornea-to-ring-base) instead of a guessed fixed value. No existing ETVR test stand found (web search 10-23); RestEasy (open-source psychophysics chin rest) is the closest reference.

## Plan
Print ONE `v13_Left` in PLA (≈2.4 g) now. When the 130° cameras arrive (Oct 16-21), dry-fit the camera in the pocket
and check the IR-filter-removed lens clears the ring. The Quest 3 fit itself waits for the headset.
