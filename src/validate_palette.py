"""Palette gate checks in Python (no node available).

Implements the six checks: lightness band, chroma floor, CVD separation on the
requested pairlist, normal-vision floor, and contrast against the surface.
Distances are OKLab x100; CVD simulation is Machado, Oliveira & Fernandes (2009)
at severity 1.0 over protanopia and deuteranopia -- the model the dataviz
thresholds are calibrated to, so it is part of the standard rather than an
implementation detail.
"""
import itertools
import numpy as np


# ---------------------------------------------------------------- colour maths
def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(hexstr):
    lin = srgb2lin(hex2rgb(hexstr))
    lms = np.cbrt(M1 @ lin)
    return M2 @ lms


def oklch(hexstr):
    L, a, b = oklab(hexstr)
    return L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360


def dE(h1, h2):
    return float(np.linalg.norm(oklab(h1) - oklab(h2)) * 100)


def relative_luminance(hexstr):
    r, g, b = srgb2lin(hex2rgb(hexstr))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(h1, h2):
    a, b = relative_luminance(h1), relative_luminance(h2)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


# ------------------------------------------------------- dichromat simulation
# Machado, Oliveira & Fernandes (2009), severity 1.0, applied in linear RGB.
# The dataviz CVD thresholds are calibrated to this model, and gate protanopia
# and deuteranopia only.
CVD_M = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
}


def simulate(hexstr, kind):
    rgb = lin2srgb(CVD_M[kind] @ srgb2lin(hex2rgb(hexstr)))
    return "#" + "".join(f"{int(round(c * 255)):02x}" for c in rgb)


def dE_cvd(h1, h2):
    return min(dE(simulate(h1, k), simulate(h2, k)) for k in CVD_M)


# ------------------------------------------------------------------ the gate
def validate(palette, surface="#fcfcfb", mode="light", pairs="adjacent",
             L_band=(0.43, 0.77), chroma_floor=0.10, name=""):
    print(f"\n{'=' * 74}\n{name or 'palette'}  [{mode}, pairs={pairs}, surface={surface}]\n{'=' * 74}")
    ok = True

    print(f"{'slot':<6}{'hex':<10}{'L':>7}{'C':>7}{'h':>7}   {'contrast':>9}")
    for i, c in enumerate(palette, 1):
        L, C, h = oklch(c)
        cr = contrast(c, surface)
        flags = []
        if not (L_band[0] <= L <= L_band[1]):
            flags.append(f"L out of band {L_band}")
        if C < chroma_floor:
            flags.append(f"C below floor {chroma_floor}")
        print(f"{i:<6}{c:<10}{L:>7.3f}{C:>7.3f}{h:>7.1f}   {cr:>9.2f}"
              + ("   <-- " + "; ".join(flags) if flags else ""))
        if flags:
            ok = False
        if cr < 3.0:
            print(f"       WARN slot {i}: contrast {cr:.2f} < 3.0 -> relief rule "
                  f"(visible direct labels or table view required)")

    plist = (list(zip(range(len(palette) - 1), range(1, len(palette))))
             if pairs == "adjacent" else list(itertools.combinations(range(len(palette)), 2)))
    print(f"\n{'pair':<10}{'normal dE':>11}{'CVD dE':>9}   verdict")
    worst_n, worst_c = 1e9, 1e9
    for i, j in plist:
        dn, dc = dE(palette[i], palette[j]), dE_cvd(palette[i], palette[j])
        worst_n, worst_c = min(worst_n, dn), min(worst_c, dc)
        v = "PASS"
        if dn < 15:
            v = "FAIL (normal-vision floor 15)"; ok = False
        elif dc < 6:
            v = "FAIL (CVD floor 6)"; ok = False
        elif dc < 8:
            v = "WARN (CVD 6-8: needs secondary encoding)"
        print(f"{i+1}-{j+1:<8}{dn:>11.1f}{dc:>9.1f}   {v}")
    print(f"\nworst normal-vision dE {worst_n:.1f} (floor 15) | worst CVD dE {worst_c:.1f} (target 8)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def ordinal(ramp, surface="#fcfcfb", name=""):
    """Ramp checks for an ordinal/sequential scale: monotone L, dL>=0.06, light end >= 2:1."""
    print(f"\n{'=' * 74}\n{name or 'ramp'}  [ordinal, surface={surface}]\n{'=' * 74}")
    Ls = [oklch(c)[0] for c in ramp]
    hs = [oklch(c)[2] for c in ramp]
    ok = True
    for i, c in enumerate(ramp):
        L, C, h = oklch(c)
        print(f"{i:<4}{c:<10}L={L:.3f}  C={C:.3f}  h={h:6.1f}  contrast={contrast(c, surface):.2f}")
    mono = all(b < a for a, b in zip(Ls, Ls[1:])) or all(b > a for a, b in zip(Ls, Ls[1:]))
    dmin = min(abs(b - a) for a, b in zip(Ls, Ls[1:]))
    hue_spread = max(hs) - min(hs)
    light_end = max(contrast(ramp[0], surface), contrast(ramp[-1], surface))
    dark_end = min(contrast(ramp[0], surface), contrast(ramp[-1], surface))
    print(f"\nmonotone L: {mono} | min adjacent dL {dmin:.3f} (>=0.06) | "
          f"hue spread {hue_spread:.1f} deg | end nearest surface {dark_end:.2f}:1 (>=2.0)")
    for cond, msg in [(mono, "L not monotone"), (dmin >= 0.06, "adjacent dL below 0.06"),
                      (hue_spread <= 25, "not a single hue"), (dark_end >= 2.0, "light end below 2:1")]:
        if not cond:
            print("  FAIL:", msg); ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    # --- roles used by the REM-exit-continuity visual edition ---------------
    # REM-exit destination is the notebook's core entity and appears in scatter,
    # stacks and small multiples -> must clear the harder all-pairs list.
    DEST = ["#2a78d6", "#eb6834", "#1baf7a"]          # light / wake / deep
    validate(DEST, name="REM-exit destination (slots 1-3)", pairs="all")

    # Cognitive group: two series, bars + scatter -> all-pairs is the same as adjacent.
    GROUP = ["#4a3aa7", "#e34948"]                     # unimpaired / impaired
    validate(GROUP, name="cognitive group (slots 7-8)", pairs="all")

    # Destination trio and group pair co-occur in fig 1; check the union all-pairs.
    validate(DEST + GROUP, name="union: destination + group", pairs="all")

    # Outcome series (ACE total vs memory) in the nights-needed / primary panels.
    OUTCOME = ["#2a78d6", "#1baf7a", "#eb6834"]        # total / memory / power
    validate(OUTCOME, name="outcome series", pairs="all")

    # Sequential blue ramp for the transition-matrix heatmap (documented steps).
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    ordinal(SEQ, name="sequential blue 100-700")
