#!/usr/bin/env python3
"""Author and check Kyron lenses.

A lens is twenty numbers -- a 4x5 colour matrix. This applies one exactly the
way the app does, so what you see here is what the camera renders and what
gets baked into the photograph.

"Exactly" is checked rather than claimed. `verify` compares this
implementation against values captured from Flutter's own ColorFilter.matrix;
see docs/LENS_FORMAT.md.

    lens.py preview  <catalogue.json> <photo.jpg> [-o sheet.png]
    lens.py check    <catalogue.json>
    lens.py verify   <probe.json>
    lens.py vectors  [format-vectors.json]

`preview` writes a contact sheet: the original, then every lens, labelled. It
is the whole point of this file -- tuning twenty numbers without seeing them
is guesswork.
"""
import argparse
import json
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

MATRIX_LENGTH = 20
MAX_COEFFICIENT = 8.0
MAX_OFFSET = 255.0
ID = re.compile(r'^[a-z0-9][a-z0-9_-]*$')

# The newest lens shape. 1: a colour matrix. 2: attachments on a tracked face.
# 3: effects that change the face instead of hanging something on it.
# Kept in step with Lens.supportedSchema in the app -- format-vectors.json is
# what stops the two drifting.
SUPPORTED_SCHEMA = 3
ANCHORS = ('eyes', 'nose', 'mouth', 'forehead', 'chin')
MAX_ATTACHMENTS = 8
MAX_ATTACHMENT_WIDTH = 12.0
MAX_ATTACHMENT_OFFSET = 8.0

# Regions an effect may name, from FaceRegionKind in the app.
REGIONS = ('lowerFace', 'eyes', 'face')
MAX_EFFECTS = 4

# Every number an effect takes, with its default and the range it must be in.
# Ranges are nonsense limits rather than safety ones: an effect cannot do
# anything dangerous, only look wrong. Blur and feather are in pupil-gaps, so
# two of them is already a face and a half.
EFFECT_NUMBERS = {
    'fill': {'feather': (0.14, 0.0, 2.0), 'keepShading': (0.35, 0.0, 1.0)},
    'frost': {
        'blur': (0.16, 0.0, 2.0),
        'desaturate': (0.3, 0.0, 1.0),
        'lift': (0.16, 0.0, 1.0),
        'feather': (0.05, 0.0, 2.0),
    },
}


def effect_problems(item, where):
    """Everything wrong with one effect.

    Same rules as LensEffect.tryParse in the app, down to the treatment of an
    explicit null: the app reads `json['feather'] ?? 0.14`, so a null there is
    the default rather than an error, and this has to agree or a lens passes
    here and disappears on the phone.
    """
    found = []
    if not isinstance(item, dict):
        return [f'{where}: not an object']

    kind = item.get('kind')
    if kind not in EFFECT_NUMBERS:
        return [
            f'{where}: kind {kind!r} must be one of '
            f'{", ".join(sorted(EFFECT_NUMBERS))}'
        ]

    for field, (default, low, high) in EFFECT_NUMBERS[kind].items():
        value = item.get(field)
        if value is None:
            value = default
        if not _is_number(value):
            found.append(f'{where}: {field} {value!r} is not a number')
        elif not low <= value <= high:
            found.append(f'{where}: {field} {value} is outside {low} to {high}')

    if kind == 'fill':
        region = item.get('region')
        if region not in REGIONS:
            found.append(
                f'{where}: region {region!r} must be one of '
                f'{", ".join(REGIONS)}'
            )
    else:
        # A blur of zero is not frost, it is nothing. Refused rather than
        # published as a lens that appears to do something and does not.
        blur = item.get('blur')
        if blur is None:
            blur = EFFECT_NUMBERS['frost']['blur'][0]
        if _is_number(blur) and blur <= 0:
            found.append(f'{where}: blur must be above 0, not {blur}')

        reveal = item.get('reveal')
        if reveal is not None and reveal not in REGIONS:
            found.append(
                f'{where}: reveal {reveal!r} must be one of '
                f'{", ".join(REGIONS)}'
            )

    return found


def attachment_problems(item, where):
    """Everything wrong with one attachment.

    Same rules as LensAttachment.tryParse in the app.
    """
    found = []
    if not isinstance(item, dict):
        return [f'{where}: not an object']

    asset = item.get('asset')
    # HTTPS only. A lens is data the app fetches, and anything else in a
    # published catalogue is a mistake or somebody testing what it will load.
    if not isinstance(asset, str) or not asset.startswith('https://'):
        found.append(f'{where}: asset must be an https:// URL, got {asset!r}')
    elif len(asset) > 500:
        found.append(f'{where}: asset URL is over 500 characters')

    anchor = item.get('anchor')
    if anchor not in ANCHORS:
        found.append(
            f'{where}: anchor {anchor!r} must be one of {", ".join(ANCHORS)}'
        )

    width = item.get('width')
    if not _is_number(width):
        found.append(f'{where}: width {width!r} is not a number')
    elif not 0 < width <= MAX_ATTACHMENT_WIDTH:
        found.append(
            f'{where}: width {width} must be above 0 and at most '
            f'{MAX_ATTACHMENT_WIDTH} (multiples of the interpupillary distance)'
        )

    for axis in ('offsetX', 'offsetY'):
        value = item.get(axis, 0)
        if not _is_number(value):
            found.append(f'{where}: {axis} {value!r} is not a number')
        elif abs(value) > MAX_ATTACHMENT_OFFSET:
            found.append(
                f'{where}: {axis} {value} is outside '
                f'+/-{MAX_ATTACHMENT_OFFSET}'
            )

    rotation = item.get('rotation', 0)
    if not _is_number(rotation):
        found.append(f'{where}: rotation {rotation!r} is not a number')
    elif abs(rotation) > 360:
        found.append(f'{where}: rotation {rotation} is outside +/-360')

    return found


def _is_number(value):
    """A real, finite number. Booleans are not numbers, whatever Python says
    about isinstance(True, int)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return value == value and value not in (float('inf'), float('-inf'))


def apply_matrix(image, matrix):
    """One lens over one image.

    Unpremultiplied 0-255 sRGB, clamped, then rounded half-up. Every one of
    those four words was checked against Flutter rather than assumed: floor
    instead of round is wrong on 19 of 48 probe pixels.
    """
    rgba = np.asarray(image.convert('RGBA'), dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 5)

    # rgba @ m[:, :4].T applies the coefficients; m[:, 4] is the offset column.
    out = rgba @ m[:, :4].T + m[:, 4]
    out = np.clip(out, 0, 255)
    # numpy rounds half to even; Skia rounds half away from zero.
    out = np.floor(out + 0.5)
    return Image.fromarray(out.astype(np.uint8), 'RGBA')


def problems(lens):
    """Everything wrong with one lens, in the words the app would use.

    Deliberately the same rules as Lens.tryParse in the app. If these two ever
    disagree, a lens passes here and vanishes on the phone with no explanation
    -- which is the failure this file exists to prevent.
    """
    found = []
    if not isinstance(lens, dict):
        return ['not an object']

    lens_id = lens.get('id')
    if not isinstance(lens_id, str) or not ID.match(lens_id or '') or len(lens_id) > 40:
        found.append(f'id {lens_id!r} must be lowercase letters, digits, - and _')

    name = lens.get('name')
    if not isinstance(name, str) or not name.strip() or len(name) > 40:
        found.append(f'name {name!r} must be 1-40 characters')

    author = lens.get('author')
    if author is not None and (not isinstance(author, str) or len(author) > 80):
        found.append('author must be a string of at most 80 characters')

    schema = lens.get('schema', 1)
    if not isinstance(schema, int) or isinstance(schema, bool) or \
            not 1 <= schema <= SUPPORTED_SCHEMA:
        found.append(
            f'schema {schema!r} must be an integer from 1 to {SUPPORTED_SCHEMA}'
        )
        schema = 1

    attachments = lens.get('attachments')
    if attachments is not None:
        if not isinstance(attachments, list):
            found.append('attachments must be a list')
        elif len(attachments) > MAX_ATTACHMENTS:
            found.append(f'at most {MAX_ATTACHMENTS} attachments')
        else:
            for index, item in enumerate(attachments):
                found.extend(attachment_problems(item, f'attachment {index}'))
            if attachments and schema < 2:
                found.append(
                    'attachments need schema 2; a lens claiming schema 1 with '
                    'attachments is lying about what it needs to be drawn'
                )

    effects = lens.get('effects')
    if effects is not None:
        if not isinstance(effects, list):
            found.append('effects must be a list')
        elif len(effects) > MAX_EFFECTS:
            found.append(f'at most {MAX_EFFECTS} effects')
        else:
            for index, item in enumerate(effects):
                found.extend(effect_problems(item, f'effect {index}'))
            if effects and schema < 3:
                found.append(
                    'effects need schema 3; a lens claiming less with effects '
                    'is lying about what it needs to be drawn'
                )

    matrix = lens.get('matrix')
    if matrix is None:
        return found  # The identity lens, or an attachments-only lens.

    if not isinstance(matrix, list) or len(matrix) != MATRIX_LENGTH:
        found.append(
            f'matrix must be {MATRIX_LENGTH} numbers, got '
            f'{len(matrix) if isinstance(matrix, list) else type(matrix).__name__}'
        )
        return found

    for i, value in enumerate(matrix):
        row, col = divmod(i, 5)
        where = f'row {row}, {"offset" if col == 4 else "column " + str(col)}'
        if not _is_number(value):
            found.append(
                f'{where}: {value!r} is not a usable number '
                '(NaN and infinity poison every pixel)'
            )
        else:
            limit = MAX_OFFSET if col == 4 else MAX_COEFFICIENT
            if abs(value) > limit:
                found.append(f'{where}: {value} is outside +/-{limit}')
    return found


def read_catalogue(path):
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get('lenses'), list):
        sys.exit(f'{path}: needs a top-level object with a "lenses" list')
    return data['lenses']


def cmd_check(args):
    lenses = read_catalogue(args.catalogue)
    seen, failed = set(), 0

    for index, lens in enumerate(lenses):
        found = problems(lens)
        name = lens.get('id', f'#{index}') if isinstance(lens, dict) else f'#{index}'
        if isinstance(lens, dict) and lens.get('id') in seen:
            found.append(f'duplicate id {lens["id"]!r}')
        if isinstance(lens, dict):
            seen.add(lens.get('id'))

        if found:
            failed += 1
            print(f'FAIL {name}')
            for problem in found:
                print(f'       {problem}')
        else:
            print(f'ok   {name}')

    print()
    if failed:
        print(f'{failed} of {len(lenses)} lenses would be dropped by the app.')
        return 1
    print(f'{len(lenses)} lenses, all readable.')
    return 0


def cmd_preview(args):
    lenses = read_catalogue(args.catalogue)
    bad = [l for l in lenses if problems(l)]
    if bad:
        sys.exit('Some lenses are not readable. Run `lens.py check` first.')

    photo = Image.open(args.photo).convert('RGBA')
    photo.thumbnail((args.size, args.size))

    tiles = [('original', photo)]
    for lens in lenses:
        matrix = lens.get('matrix')
        if lens.get('attachments') and matrix is None:
            # A face lens has nothing to show on a photograph with no face in
            # it. Saying so beats an unchanged tile that reads as a bug.
            tiles.append((f'{lens["name"]} (face lens)', photo))
            continue
        tiles.append((
            lens['name'],
            photo if matrix is None else apply_matrix(photo, matrix),
        ))

    columns = args.columns
    rows = (len(tiles) + columns - 1) // columns
    width, height = photo.size
    label = 22
    pad = 8

    sheet = Image.new(
        'RGB',
        (columns * (width + pad) + pad, rows * (height + label + pad) + pad),
        (21, 21, 21),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (name, tile) in enumerate(tiles):
        column, row = index % columns, index // columns
        x = pad + column * (width + pad)
        y = pad + row * (height + label + pad)
        sheet.paste(tile.convert('RGB'), (x, y))
        draw.text((x + 2, y + height + 4), name, fill=(230, 230, 230))

    sheet.save(args.out)
    print(f'{args.out}: {len(tiles)} tiles, {sheet.size[0]}x{sheet.size[1]}')
    return 0


def cmd_vectors(args):
    """Runs the shared spec against this implementation.

    The same file runs against Lens.tryParse in the app. It is the only thing
    keeping two implementations of one set of rules honest -- without it, a
    rule tightened here and not there means a lens passes `check`, gets
    published, and silently never appears on anybody's phone.
    """
    def not_json(literal):
        # Python accepts `Infinity`, `-Infinity` and `NaN` as JSON; nothing
        # else does, and Dart refuses the whole file. Round-tripping this file
        # through json.dump reintroduces them, which would leave this passing
        # while the app cannot read the spec at all.
        raise ValueError(
            f'{args.vectors} contains the bare literal `{literal}`, which is '
            'not JSON. Write 1e400 instead -- it is valid, and overflows to '
            'infinity in every language that reads it.'
        )

    try:
        with open(args.vectors) as handle:
            spec = json.loads(handle.read(), parse_constant=not_json)
    except ValueError as error:
        # A message, not a traceback: this is a thing somebody has to fix.
        sys.exit(str(error))

    failures = 0
    for case in spec['cases']:
        found = problems(case['lens'])
        accepted = not found
        if accepted != case['accept']:
            failures += 1
            wanted = 'accepted' if case['accept'] else 'rejected'
            print(f'FAIL {case["id"]}: should be {wanted}')
            print(f'       {case["why"]}')
            if found:
                for problem in found:
                    print(f'       got: {problem}')
            else:
                print('       got: accepted with no complaint')

    total = len(spec['cases'])
    print(f'{total - failures}/{total} vectors agree with this implementation')
    return 1 if failures else 0


def cmd_verify(args):
    """Checks this file against values captured from Flutter itself."""
    with open(args.probe) as handle:
        probe = json.load(handle)

    matrices = probe['matrices']
    mismatches = 0
    for lens_id, rows in probe['results'].items():
        for i, expected in enumerate(rows):
            swatch = probe['swatches'][i]
            pixel = Image.new('RGBA', (1, 1), tuple(swatch) + (255,))
            got = list(apply_matrix(pixel, matrices[lens_id]).getpixel((0, 0))[:3])
            if got != expected:
                mismatches += 1
                print(f'FAIL {lens_id} on {swatch}: Flutter {expected}, here {got}')

    total = sum(len(r) for r in probe['results'].values())
    print(f'{total - mismatches}/{total} pixels identical to Flutter')
    return 1 if mismatches else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    check = sub.add_parser('check', help='validate a catalogue')
    check.add_argument('catalogue')
    check.set_defaults(run=cmd_check)

    preview = sub.add_parser('preview', help='contact sheet of every lens')
    preview.add_argument('catalogue')
    preview.add_argument('photo')
    preview.add_argument('-o', '--out', default='lens-sheet.png')
    preview.add_argument('--columns', type=int, default=4)
    preview.add_argument('--size', type=int, default=320)
    preview.set_defaults(run=cmd_preview)

    verify = sub.add_parser('verify', help='check against captured Flutter output')
    verify.add_argument('probe')
    verify.set_defaults(run=cmd_verify)

    vectors = sub.add_parser('vectors', help='run the shared format spec')
    vectors.add_argument('vectors', nargs='?', default='format-vectors.json')
    vectors.set_defaults(run=cmd_vectors)

    args = parser.parse_args()
    sys.exit(args.run(args))


if __name__ == '__main__':
    main()
