# Real AR: what it takes, and what it costs

Today a lens is a colour matrix. It changes every pixel the same way and knows
nothing about what is in the frame. This is the design for lenses that are
attached to a face — glasses that stay on the eyes, something on the forehead
that follows it — and for the decisions that has to be made before writing the
format down.

## The stake

Everything about the current catalogue rests on one property: **a lens is
data, not code.** Twenty numbers, nothing to compile, nothing to execute. That
is why a lens can be downloaded from a URL and applied to somebody's camera
without asking whether it is safe.

Real AR does not have to give that up, but it is easy to give up by accident.
A lens with geometry and a texture is still data. A lens with a *script* is
not — and that is exactly what Snapchat's Lens Studio produces: lenses
containing JavaScript, which is a program running on the viewer's phone.

**Schema 2 must stay declarative.** "Attach this mesh to this anchor with this
transform, and let these blendshapes drive these parameters" is data.
"Run this function every frame" is not. The moment a lens can execute, every
published lens becomes code review, and the catalogue stops being a thing
anybody can safely publish to.

## Snapchat, specifically

Their lenses cannot be used. They are proprietary content, and Camera Kit —
the SDK that renders them — is a commercial licence: monthly-active-user
pricing, custom enterprise terms, a watermark on the staging tier, and
production access gated behind app review by Snap.

Adopting it would mean Kyron's AR is Snap's AR, on Snap's terms, revocable by
Snap. For a project whose README says *user-owned, zero gatekeepers,
MIT-licensed*, that is a contradiction rather than a trade-off. Lens Studio is
worth studying as a reference for what good feels like. It is not a source of
lenses.

## The recommendation: MediaPipe

`mediapipe_face_mesh` on pub.dev — BSD-3-Clause, Apache-2.0 models, Android
and iOS, actively maintained. It gives, on device and free:

- **478 landmarks**, including 10 iris points
- **52 ARKit-style blendshapes** — `jawOpen`, `mouthSmileLeft`, `eyeBlinkRight`
- a **4×4 facial transformation matrix**, which is the thing that places an
  object on a head with full rotation rather than approximating it

The same models run on the desktop through the Python package, which is what
makes any of this checkable here: `tools/face.py` produces a real detection
from a real photograph, and a sprite composited through the placement maths
either lands on the eyes or does not.

## Placement, and why it is measured in eyes

An attachment resolves to a **centre**, a **scale** and a **roll**.

Scale comes from the **interpupillary distance** — the gap between the irises
— not from a face bounding box. Both irises stay visible well past the angle
at which a jaw outline stops describing anything, so IPD keeps meaning the
same thing as the head turns. Measured across ±30° of roll on one face it
varied by 1.4px in 86 (1.6%), while the box around the face changed shape
completely.

So a lens says its size in faces, not pixels: `"width": 2.6` means *2.6 times
the interpupillary distance*, and it is correct at any distance from the
camera, on any face, at any resolution.

Anchors, each defined by landmark indices checked by plotting them on a real
face rather than taken from memory: `eyes`, `nose`, `mouth`, `forehead`,
`chin`, and `face` for the full 4×4 transform.

## What is proven, and what is not

**Checked, by running it:**

- MediaPipe detects a face and returns 478 landmarks, 52 blendshapes and the
  transformation matrix — on this machine, on a real photograph.
- Every landmark index above, by drawing it on the face and looking.
- The expressions are real: a smiling portrait scores `mouthSmileLeft` at 0.96.
- Glasses land on the eyes, correctly scaled and rotated.
- They stay on the eyes across ±30° of roll, with detected roll matching the
  applied rotation to within 1.4°.

**Not checked, and needing hardware:**

- Anything live. Frame rate, tracking stability, jitter between frames, and
  whether the latency between a frame arriving and the overlay drawing is
  small enough not to look detached.
- **Yaw and pitch.** Rotation was tested by rotating a photograph, which only
  produces roll. A head turning to the side is the case where a flat sprite
  stops being convincing and a real mesh starts being necessary — and it is
  unverified.
- More than one face. Everything above is one person, in good light, facing
  the camera.

That third point deserves emphasis: one face at five angles is a working
prototype, not evidence the maths generalises.

## The order to build it in

1. **A flat sprite on an anchor.** Glasses, moustaches, hats. This is most of
   what people actually make, it works with the maths already checked, and it
   is the smallest thing that is honestly AR.
2. **Blendshape drivers.** `jawOpen` scales something, `eyeBlinkLeft` swaps a
   texture. Still declarative, and it is what makes a lens feel alive rather
   than stuck on.
3. **A 3D mesh on the transform matrix.** glTF, placed with the full 4×4. This
   is where yaw stops being a problem and where the asset pipeline gets real.
4. **Occlusion.** The face mesh drawn to the depth buffer only, so glasses arms
   disappear behind the head. Cheap once there is a mesh, and the single
   biggest difference between "stuck on" and "there".

Segmentation, background replacement and hand tracking are all separate
projects and none of them are on this path.

## What stays true

Whatever schema 2 becomes, the rules the app already enforces do not change: a
lens that fails to load leaves the built-ins working, a published file cannot
replace a bundled lens, and the app never waits on the network to draw its
first frame. Adding AR must not make a camera that can fail to open.
