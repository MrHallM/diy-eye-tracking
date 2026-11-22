# IR safety: read this before you build

Eye tracking lights your eyes with infrared LEDs from a centimetre or two away. You can't see IR light, so you
**can't tell by looking** whether it is too bright. Too much IR over long sessions can damage the eye, and the damage
may be permanent.

**Use EyeTrackVR's safety guide as the reference:**
[docs.eyetrackvr.dev, Near-Eye Infrared (IR) Emitter Safety](https://docs.eyetrackvr.dev/getting_started/led_safety).
It explains the limits and how to calculate your exposure from the LED datasheet. Its key points:

- The limit is about **irradiance** (light power per area at the eye), not LED current. Calculate it as
  E = radiant intensity / distance², summed over all LEDs facing the eye.
- Design limits it cites: **10 mW/cm²** at the cornea for long exposures (ICNIRP 2013) and **4 mW/cm²** for the
  retinal hazard region (IEC 62471 / IEC 60825-1). EyeTrackVR's official kits stay well below both.
- **Don't bypass current-limiting resistors or regulators**, and don't swap LEDs for brighter ones without redoing
  the math.
- Check with an IR-sensitive camera that the brightness is what you expect, and stop if your eyes feel strained,
  dry or hot.

Nothing in this repository changes LED current. The firmware patches only touch the camera, streaming and API.
If you build your own LED board, you are responsible for its safety.

# Network privacy

The camera boards stream video of your eyes over Wi-Fi. In OpenIris (with or without the patches here) the stream
and its REST API have **no password**, so anyone on the same network can watch the cameras and send commands.

- Use the boards only on a **home network you trust**, never on public or shared Wi-Fi.
- The OpenIris patch **0009** allows several viewers at once, so an extra viewer would not cut off yours.
- The patch **0015** adds debug commands that change camera settings over the network. Leave it out unless you
  need it.
- The PC-side tools and the Baballonia control API added here listen on `127.0.0.1` only (your own PC), not
  on the network.
