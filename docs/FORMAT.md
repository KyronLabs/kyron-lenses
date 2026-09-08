# The lens format

A lens is twenty numbers. That is the whole format, and it is why lenses can
be a product rather than a release: publishing one is publishing a line of
JSON, with no app store review and no version anybody has to install.

## Why it is data and not code

A lens is a 4×5 colour matrix — the form `dart:ui` and Skia take. Four rows of
`[r, g, b, a, offset]`, applied per pixel:

```
R' = round(clamp(m0·R + m1·G + m2·B  + m3·A  + m4))
G' = round(clamp(m5·R + m6·G + m7·B  + m8·A  + m9))
B' = round(clamp(m10·R + m11·G + m12·B + m13·A + m14))
A' = round(clamp(m15·R + m16·G + m17·B + m18·A + m19))
```

Channels are 0–255, unpremultiplied sRGB. Offsets are added after the multiply
and are also in 0–255.

**This constraint is what makes downloading a lens safe.** There is no shader
to compile and nothing to execute — the worst a hostile file can do is look
ugly. Had lenses been fragment shaders (which is what blur, warp or face
tracking would need), serving them from a catalogue would mean running a
stranger's program on somebody's phone. That is a different product with a
different threat model, and this is not it.

The validation below is therefore about catching nonsense, not preventing
harm. A matrix of `1e9` produces a solid white frame, which is a bad lens, not
an attack.

## One lens

```json
{
  "id": "sepia",
  "name": "Sepia",
  "author": "Kyron",
  "matrix": [
    0.393, 0.769, 0.189, 0, 0,
    0.349, 0.686, 0.168, 0, 0,
    0.272, 0.534, 0.131, 0, 0,
    0,     0,     0,     1, 0
  ]
}
```

| Field | Rules |
|:--|:--|
| `id` | Required. `^[a-z0-9][a-z0-9_-]*$`, at most 40 characters. It is a cache key and appears in log lines, so it may not look like a path. |
| `name` | Required. 1–40 characters, what the chip says. |
| `matrix` | 20 finite numbers. Coefficients within ±8, offsets (every fifth) within ±255. Omit entirely for a lens that changes nothing. |
| `author` | Optional, at most 80 characters. |

A lens that breaks any of these is **dropped**, and the app logs that it was.
The rest of the catalogue still loads: one typo costs one lens.

## Schema 2: something on a face

A lens may also hang pictures on a tracked face.

```json
{
  "id": "specs",
  "name": "Specs",
  "schema": 2,
  "attachments": [
    {
      "asset": "https://kyronlabs.github.io/kyron-lenses/assets/glasses.png",
      "anchor": "eyes",
      "width": 2.6
    }
  ]
}
```

| Field | Rules |
|:--|:--|
| `asset` | Required. **HTTPS only**, at most 500 characters. A lens is data the app fetches; `http://`, `file://` and `data:` in a published catalogue are a mistake or somebody testing what it will load. |
| `anchor` | Required. `eyes`, `nose`, `mouth`, `forehead` or `chin`. Lowercase, like an id. |
| `width` | Required. **In pupil-gaps**, not pixels — see below. Above 0, at most 12. |
| `offsetX`, `offsetY` | Optional, also in pupil-gaps, within ±8. Rotated with the head, so a hat pushed "up" stays up when somebody leans. |
| `rotation` | Optional degrees within ±360, on top of the head's own tilt. |

At most 8 attachments. **A lens with attachments must declare `"schema": 2`** —
without it an older build reads a lens with no matrix as the do-nothing lens
and shows a chip that is there and does nothing, which is worse than a chip
that is not there. Unknown schemas are dropped for the same reason.

### Width is measured in faces

`2.6` means *2.6 times the distance between the pupils*. Not pixels, not a
fraction of the frame.

That distance is the one measurement that keeps meaning the same thing as a
head turns — both irises stay visible well past the angle at which a jaw
outline stops describing anything. Measured across a 60° sweep of roll it
moved 1.6%, while the box around the face changed shape entirely. So a width
stated this way is correct at any distance from the camera, on any face, at
any resolution, with nothing to tune per device.

### Still no code

An attachment is a picture, a place and a size. There is nothing to execute,
which is the whole reason a lens can be downloaded and pointed at somebody's
camera. A lens that could run a script would end that, and a script is exactly
what a Lens Studio lens contains. See [AR_LENSES.md](AR_LENSES.md).

An image is not *nothing*, though: it is a decoder's worth of attack surface,
which the colour matrices were not. Hence HTTPS, hence the 4 MB ceiling the
app applies when fetching one.

## Schema 3: changing the face itself

Schema 2 puts a picture *on* a face. Schema 3 changes what is already there.

```json
{
  "id": "blank",
  "name": "Blank",
  "schema": 3,
  "effects": [
    { "kind": "fill", "region": "lowerFace", "feather": 0.14, "keepShading": 0.35 }
  ]
}
```

That one takes somebody's nose and mouth out, under skin the same colour as
the rest of their face. This one etches the whole picture except their eyes:

```json
{
  "id": "frosted",
  "name": "Frosted",
  "schema": 3,
  "effects": [
    { "kind": "frost", "blur": 0.16, "desaturate": 0.3, "lift": 0.16,
      "reveal": "eyes", "feather": 0.05 }
  ]
}
```

Two kinds, at most four per lens. A **region** is one of `lowerFace` (the nose
and mouth, everything below the eye line down to the chin), `eyes` (a slot
across both) or `face` (the whole thing) -- spelled exactly, so `lowerface` is
refused.

| `"kind": "fill"` | Rules |
|:--|:--|
| `region` | Required. What gets covered. |
| `feather` | Optional, default `0.14`. How far the edge is blurred, in pupil-gaps, 0 to 2. Zero is a hard edge, which reads as a decal. |
| `keepShading` | Optional, default `0.35`, 0 to 1. How much of the original shading shows through. Zero is a flat colour and looks painted on. |

| `"kind": "frost"` | Rules |
|:--|:--|
| `blur` | Optional, default `0.16`, **above 0** and at most 2, in pupil-gaps. Zero is refused: it is not frost, it is nothing. |
| `desaturate` | Optional, default `0.3`, 0 to 1. |
| `lift` | Optional, default `0.16`, 0 to 1. How far everything moves towards white. |
| `reveal` | Optional, default `eyes`. What stays sharp. |
| `feather` | Optional, default `0.05`, 0 to 2. |

**A lens with effects must declare `"schema": 3`**, for the same reason schema
2 exists: an older build would otherwise read it as the do-nothing lens.

### The colour comes off the face, not out of the lens

A fill has no colour field, and that is deliberate. A skin tone written into a
lens is one person's, and a sticker on everybody else. The app measures it
instead -- five squares on the forehead and cheekbones, positioned in
pupil-gaps from the anchor, averaged over the middle half by brightness so a
strand of hair or a highlight does not drag the answer. If it cannot read one,
the fill draws nothing rather than guessing.

### Blur, not opacity

Both kinds are the same mechanism: **a masked blur of what is already there**,
with a colour over it. That is what takes a nose out of a picture. Letting the
sharp original through at partial opacity does not -- measured, at
`keepShading` levels that looked right it changed 31,927 pixels and still read
as an unmodified face, nostrils and lips plainly visible.

Which is why `blur` and `feather` are in pupil-gaps like everything else: the
effect is the same strength relative to a face whether somebody is at arm's
length or across a room.

### Still no code

A region is a name from a list of three. There is nothing to execute here
either -- the app owns every line that does the drawing, and a lens only
chooses between things it already knows how to do. An effect the app does not
recognise is refused, not approximated.

## The catalogue

```json
{
  "schema": 1,
  "lenses": [ { … }, { … } ]
}
```

At most 120 lenses and 512 KB. Both are ceilings against a wrong URL, not
targets.

The app fetches it from `LensCatalogue.catalogueUrl`, overridable at build
time with `--dart-define=KYRON_LENS_CATALOGUE=…` so a staging build can point
elsewhere. Setting it empty disables fetching and leaves the built-ins.

### Three rules the app enforces

1. **Built-ins always win.** The seven lenses bundled with the app cannot be
   replaced or removed by a published file. A catalogue that redefined `mono`
   would otherwise change what somebody's already-taken photographs look like,
   and one that shipped an empty list would empty the strip.
2. **Cache before network.** The strip draws from disk immediately and the
   fetch refreshes it for next time. A camera that waits on a request before
   showing a lens is a slow camera.
3. **Every failure ends at the built-ins.** No network, bad JSON, a 500, a
   file over the ceiling — all of them leave seven working lenses.

## Writing one

`tools/lens.py` is the reason this is practical. Tuning twenty numbers
without seeing them is guesswork.

```
# Would the app accept these?
python3 tools/lens.py check lenses.json

# What do they look like?
python3 tools/lens.py preview lenses.json sample.png -o sheet.png
```

`check` applies **the same rules as `Lens.tryParse`** in the app. If the two
ever disagree, a lens passes locally and vanishes on the phone with nothing
saying why.

`format-vectors.json` is what stops that being a matter of memory. Forty
cases — nine that must be accepted, thirty-one that must be refused — run by
both implementations:

```
python3 tools/lens.py vectors format-vectors.json
```

It is canonical here and vendored into the app, whose CI compares its copy
against the published one and fails if the spec moved on without it. **Adding
a rule means adding a case**, in the same commit.

One thing the file cannot cover: NaN. JSON has no way to write it, so each
side tests that in its own unit tests. What JSON *can* carry is `1e400`, which
overflows to infinity in both languages, and that is a case here.

### The preview does not lie

`preview` renders with its own implementation of the matrix, which would be
worthless if it disagreed with Flutter. It does not, and that is checked
rather than claimed:

```
python3 tools/lens.py verify tools/flutter-probe.json
→ 48/48 pixels identical to Flutter
```

`flutter-probe.json` holds real output captured from Flutter's own
`ColorFilter.matrix` — eight colours through six lenses. Pinning it down found
one thing worth knowing: Skia **rounds** half away from zero. Truncating
instead is wrong on 19 of those 48 pixels, and numpy's default rounds half to
*even*, which is also wrong. The tool does `floor(x + 0.5)` for that reason.

Re-capture the probe if the lens maths ever changes.

## Publishing

Merging to `main` publishes to GitHub Pages:

```
https://kyronlabs.github.io/kyron-lenses/lenses.json
```

A stable public URL on a CDN, needing no token -- which is what an app fetching
a file at runtime requires. Actions artifacts cannot do that job: they need
authentication to download even from a public repository, they expire, and the
id changes every run. CI uses one for the contact sheet, which is a thing a
person looks at once, on a pull request.

Adding a lens is: write it, `check` it, `preview` it, open a pull request. No
release, no upload, no store review.

## What this format cannot do

Worth saying plainly, because "AR lens" suggests more:

- **No warp.** Nothing moves a pixel to a different place: no bulge, no
  stretch, no swapped faces. Blur arrived with schema 3, but only as one of
  two named effects over a region from a list of three -- not as anything a
  lens can describe for itself. That needs a fragment shader, and a shader is
  a program.
- **No animation.** A lens is a constant, not a function of time. Nothing
  responds to an expression yet, though the tracker reports 52 of them.
- **No 3D.** An attachment is a flat picture. A head turning to the side is
  where that stops being convincing, and it is the next thing to build.
- **No occlusion.** Glasses arms draw over the head rather than behind it.

Any of those would mean a new `schema` and a real look at what it means to
download one. See [AR.md](https://github.com/KyronLabs/kyron/blob/main/docs/AR.md) in the
main repository for where the line currently sits.
