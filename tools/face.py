#!/usr/bin/env python3
"""Face tracking on the desktop, so AR placement can be checked without a phone.

The app tracks faces with MediaPipe on the device. This runs the *same models*
here, which is the only reason the placement maths can be developed at all in
an environment with no camera: a real detection produces real landmarks, and a
sprite composited onto a real photograph either lands on the eyes or does not.

    face.py detect <image> [-o detection.json]
    face.py place  <image> <sprite.png> --anchor eyes [-o placed.png]
    face.py roll   <image> <sprite.png>       # a strip across head angles

Needs `pip install mediapipe pillow numpy` and the model:

    curl -o face_landmarker.task https://storage.googleapis.com/\\
mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
import argparse
import json
import math
import sys

import numpy as np
from PIL import Image

# Landmark indices, each checked by plotting them on a real face rather than
# taken from memory -- a wrong index puts the glasses on somebody's ear.
LANDMARKS = {
    'left_iris': 468, 'right_iris': 473,
    'left_eye_outer': 33, 'left_eye_inner': 133,
    'right_eye_outer': 263, 'right_eye_inner': 362,
    'nose_tip': 1, 'chin': 152, 'forehead': 10,
    'mouth_left': 61, 'mouth_right': 291,
    'lip_top': 13, 'lip_bottom': 14,
}

MODEL = 'face_landmarker.task'


def landmarker(model_path=MODEL):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
    )


def detect(image, model_path=MODEL):
    """Landmarks, blendshapes and the 4x4 transform, or None for no face."""
    import mediapipe as mp
    arr = np.asarray(image.convert('RGB'))
    with landmarker(model_path) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.face_landmarks:
        return None
    return {
        'landmarks': [[p.x, p.y, p.z] for p in res.face_landmarks[0]],
        'matrix': np.array(res.facial_transformation_matrixes[0]).tolist(),
        'blendshapes': {b.category_name: b.score
                        for b in res.face_blendshapes[0]},
    }


def anchor(name, landmarks, width, height):
    """Where an attachment goes: a centre, a size to scale by, and a roll.

    Scale comes from the interpupillary distance, not a face bounding box. It
    is the one measurement that stays meaningful as the head turns -- both
    irises remain visible well past the angle where a jaw outline stops
    describing anything -- and it is what a real face is measured in.
    """
    def px(index):
        p = landmarks[index]
        return p[0] * width, p[1] * height

    lx, ly = px(LANDMARKS['left_iris'])
    rx, ry = px(LANDMARKS['right_iris'])
    ipd = math.hypot(rx - lx, ry - ly)
    roll = math.degrees(math.atan2(ry - ly, rx - lx))

    centres = {
        'eyes': ((lx + rx) / 2, (ly + ry) / 2),
        'nose': px(LANDMARKS['nose_tip']),
        'forehead': px(LANDMARKS['forehead']),
        'chin': px(LANDMARKS['chin']),
        'mouth': tuple(
            (a + b) / 2 for a, b in zip(px(LANDMARKS['lip_top']),
                                        px(LANDMARKS['lip_bottom']))
        ),
    }
    if name not in centres:
        sys.exit(f'unknown anchor {name!r}; try one of {", ".join(centres)}')

    cx, cy = centres[name]
    return {'cx': cx, 'cy': cy, 'ipd': ipd, 'roll': roll}


def composite(image, sprite, spot, width_in_ipd):
    """The sprite, scaled to the face and rolled with it."""
    target = max(1, round(spot['ipd'] * width_in_ipd))
    scaled = sprite.resize(
        (target, max(1, round(target * sprite.height / sprite.width))),
        Image.LANCZOS,
    )
    # Negated: image rotation and head roll run in opposite directions.
    turned = scaled.rotate(-spot['roll'], expand=True, resample=Image.BICUBIC)
    out = image.convert('RGBA')
    out.alpha_composite(
        turned,
        (round(spot['cx'] - turned.width / 2),
         round(spot['cy'] - turned.height / 2)),
    )
    return out.convert('RGB')


def cmd_detect(args):
    found = detect(Image.open(args.image), args.model)
    if not found:
        sys.exit('no face found')
    json.dump(found, open(args.out, 'w'))
    print(f'{args.out}: {len(found["landmarks"])} landmarks, '
          f'{len(found["blendshapes"])} blendshapes')
    top = sorted(found['blendshapes'].items(), key=lambda kv: -kv[1])[:4]
    print('strongest expressions:',
          ', '.join(f'{k}={v:.2f}' for k, v in top))
    return 0


def cmd_place(args):
    image = Image.open(args.image)
    found = detect(image, args.model)
    if not found:
        sys.exit('no face found')
    spot = anchor(args.anchor, found['landmarks'], *image.size)
    print('anchor %s: centre=(%.1f, %.1f) ipd=%.1fpx roll=%+.2f deg'
          % (args.anchor, spot['cx'], spot['cy'], spot['ipd'], spot['roll']))
    composite(image, Image.open(args.sprite).convert('RGBA'), spot,
              args.width).save(args.out)
    print(f'wrote {args.out}')
    return 0


def cmd_roll(args):
    """The check that matters: one frontal face proves nothing."""
    base = Image.open(args.image).convert('RGB')
    sprite = Image.open(args.sprite).convert('RGBA')
    tiles = []
    for angle in (-30, -15, 0, 15, 30):
        turned = base.rotate(angle, expand=True, fillcolor=(18, 18, 18),
                             resample=Image.BICUBIC)
        found = detect(turned, args.model)
        if not found:
            print(f'{angle:+3d} deg: NO FACE DETECTED')
            continue
        spot = anchor(args.anchor, found['landmarks'], *turned.size)
        print(f'{angle:+3d} deg image -> roll {spot["roll"]:+6.2f} deg, '
              f'ipd {spot["ipd"]:5.1f}px')
        out = composite(turned, sprite, spot, args.width)
        tiles.append(out.resize((300, round(300 * out.height / out.width)),
                                Image.LANCZOS))

    if not tiles:
        sys.exit('no faces found at any angle')
    sheet = Image.new('RGB', (sum(t.width for t in tiles),
                              max(t.height for t in tiles)), (18, 18, 18))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    sheet.save(args.out)
    print(f'wrote {args.out}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL, help='face_landmarker.task')
    sub = parser.add_subparsers(dest='command', required=True)

    d = sub.add_parser('detect', help='landmarks and blendshapes as JSON')
    d.add_argument('image')
    d.add_argument('-o', '--out', default='detection.json')
    d.set_defaults(run=cmd_detect)

    p = sub.add_parser('place', help='composite a sprite onto a face')
    p.add_argument('image')
    p.add_argument('sprite')
    p.add_argument('--anchor', default='eyes')
    p.add_argument('--width', type=float, default=2.6,
                   help='sprite width as a multiple of interpupillary distance')
    p.add_argument('-o', '--out', default='placed.png')
    p.set_defaults(run=cmd_place)

    r = sub.add_parser('roll', help='a strip across head angles')
    r.add_argument('image')
    r.add_argument('sprite')
    r.add_argument('--anchor', default='eyes')
    r.add_argument('--width', type=float, default=2.6)
    r.add_argument('-o', '--out', default='rolls.png')
    r.set_defaults(run=cmd_roll)

    args = parser.parse_args()
    sys.exit(args.run(args))


if __name__ == '__main__':
    main()
