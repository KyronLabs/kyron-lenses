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

    matrix = lens.get('matrix')
    if matrix is None:
        return found  # The identity lens. Legal.

    if not isinstance(matrix, list) or len(matrix) != MATRIX_LENGTH:
        found.append(
            f'matrix must be {MATRIX_LENGTH} numbers, got '
            f'{len(matrix) if isinstance(matrix, list) else type(matrix).__name__}'
        )
        return found

    for i, value in enumerate(matrix):
        row, col = divmod(i, 5)
        where = f'row {row}, {"offset" if col == 4 else "column " + str(col)}'
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            found.append(f'{where}: {value!r} is not a number')
        elif value != value or value in (float('inf'), float('-inf')):
            found.append(f'{where}: NaN and infinity poison every pixel')
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

    args = parser.parse_args()
    sys.exit(args.run(args))


if __name__ == '__main__':
    main()
