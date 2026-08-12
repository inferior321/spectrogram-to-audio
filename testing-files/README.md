# Reference spectrograms

This folder is where reference images go. It ships empty on purpose.

The images used to build this project were spectrograms of a real recording,
and turning such a picture back into audio is exactly what this program does —
so publishing them would publish the sound. What is committed instead is
`spectro/audacity.npz`, the colour tables *derived* from them. A table is 256
RGB triples describing a gradient; it carries nothing of the recording.

Everything still passes without them: the ground-truth test reports

    SKIP - reference exports not present (8 missing)

and the other 150-odd checks run as normal, because they build their own test
images from synthesised signals.

## Supplying your own

Two of the tools here can be reproduced if you put files in this folder.

**Audacity colour tables** — `tools/derive_audacity_luts.py` expects eight
exports of the same audio, one per scheme per frequency axis:

| file | Scheme | Scale |
|---|---|---|
| `linear.png`  / `mel.png`  | Color (default) | Linear / Mel |
| `linear2.png` / `mel2.png` | Color (classic) | Linear / Mel |
| `linear3.png` / `mel3.png` | Grayscale | Linear / Mel |
| `linear4.png` / `mel4.png` | Inverse Grayscale | Linear / Mel |

Export any clip eight times, changing only Scheme and Scale between them, with
"Draw legend/axes" style chrome kept to a minimum. The script derives each
colour table by pairing a colour export against the greyscale export of the
*same axis*, then checks the table derived from the linear pair against the one
derived from the mel pair — they share nothing but the scheme, so agreement
means the derivation is right.

**ffmpeg, SoX and Spek tables** — `tools/extract_tool_palettes.py` needs no
files at all. It runs those tools directly and reads the palette out of the
colour bar each one draws in its legend. It needs `ffmpeg` and `sox` on PATH,
plus `spek`, `xdotool` and `Xvfb` for the Spek entry.
