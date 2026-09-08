# Kyron lenses

The lens catalogue for [Kyron](https://github.com/KyronLabs/kyron)'s AR camera:
the lenses themselves, and the tool for making them.

**A new lens does not need an app release.** Merge it here and the camera picks
it up.

| | |
|:--|:--|
| `lenses.json` | The catalogue: lenses the app does **not** already bundle. |
| `tools/lens.py` | Check a catalogue, and see what it looks like. |
| `tools/flutter-probe.json` | Real output captured from Flutter, so the tool can prove it renders identically. |
| `sample.png` | A colour chart: skin tones, nature, primaries, a step wedge and a ramp. |
| `docs/FORMAT.md` | The format, the rules, and what it deliberately cannot do. |

## This is what the app adds, not everything it has

Seven lenses ship inside the app — None, Mono, Warm, Cool, Faded, Punch, Noir.
They are not listed here, deliberately: the app always prefers its own copy, so
repeating them would be inert duplication that could quietly diverge from what
the camera actually renders.

`lenses.json` is what gets added on top.

## A lens is twenty numbers

That is the whole format — a 4×5 colour matrix — and it is why lenses can be a
product rather than a release.

It is also why downloading one is safe. **A lens is data, not code.** There is
no shader to compile and nothing to execute, so the worst a hostile file can do
is look ugly. Had lenses been fragment shaders — which is what blur, warp or
face tracking would need — serving them from a catalogue would mean running a
stranger's program on somebody's phone. That is a different product with a
different threat model, and this is deliberately not it.

## Adding one

```bash
pip install pillow numpy

# Would the app accept it?
python3 tools/lens.py check lenses.json

# What does it look like?
python3 tools/lens.py preview lenses.json sample.png -o sheet.png
```

Then open a pull request. CI runs both of those and **attaches the rendered
contact sheet to the run**, so whoever reviews it sees the lens rather than
twenty numbers.

### The preview does not lie

It renders with its own implementation of the colour matrix, which would be
worthless if it disagreed with the app. It does not, and that is checked on
every run rather than asserted:

```
python3 tools/lens.py verify tools/flutter-probe.json
→ 48/48 pixels identical to Flutter
```

If that check ever fails, the tool is lying and every lens tuned with it since
is suspect. It is a CI gate for that reason.

## How the app gets it

Merging to `main` publishes to **GitHub Pages**:

```
https://kyronlabs.github.io/kyron-lenses/lenses.json
```

A stable public URL on a CDN, needing no token — which is what an app fetching
a file at runtime requires. Actions artifacts cannot do that job: they need
authentication to download even from a public repo, they expire, and the id
changes every run. They are used here for the contact sheet, which is a thing a
person looks at once, on a pull request.

The app reads `LensCatalogue.catalogueUrl`, overridable with
`--dart-define=KYRON_LENS_CATALOGUE=…` so a staging build can point elsewhere.

**Pages must be enabled** for this to work: Settings → Pages → Source → GitHub
Actions. Until it is, the URL 404s, the app treats that as "no catalogue" and
falls back to the seven lenses bundled with it — so nothing breaks, but nothing
new appears either.

## What the app guarantees

Whatever is published here:

1. **The built-ins always win.** The seven lenses bundled with the app cannot
   be replaced or removed from here. A file that redefined `mono` would change
   what somebody's already-taken photographs look like.
2. **Cache before network.** The strip draws from disk immediately; the fetch
   refreshes it for next time.
3. **Every failure ends at the built-ins.** No network, bad JSON, a 500, a file
   over 512 KB — all of them leave seven working lenses. One unreadable entry
   costs one lens, not the catalogue.

So a mistake here degrades; it does not break a camera.
