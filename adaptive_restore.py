#!/usr/bin/env python3
"""
adaptive_restore.py — Adaptive Audio Restoration Pipeline
==========================================================
Differs from simple stepped-EQ scripts by:
  1. Analyzing each file's spectrum FIRST (librosa)
  2. Computing per-band deficit relative to the mid reference
  3. Detecting codec frequency cutoff automatically
  4. Applying a harmonic exciter above the cutoff
  5. Gently expanding dynamics (reverse mastering compression)
  6. Normalising to EBU R128
  7. Optionally upsampling via SoX VHQ + TPDF dither

Install deps:
  pip install librosa soundfile numpy
  apt install sox ffmpeg          # or brew install

Usage examples:
  python adaptive_restore.py input/ output/ --hires
  python adaptive_restore.py input/ output/ --target_rate 48000 --no-exciter
  python adaptive_restore.py input/ output/ --workers 8 --no-hires
"""

import os
import subprocess
import argparse
import multiprocessing
import tempfile

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠  librosa not found — install with: pip install librosa soundfile")
    print("   Falling back to flat EQ without spectral analysis.\n")


# ─────────────────────────────────────────────
#  STAGE 1: Spectral analysis
# ─────────────────────────────────────────────

BANDS = {
    # name         : (f_low, f_high, center_for_eq)
    'sub_bass'     : (20,    80,     50),
    'bass'         : (80,    250,    120),
    'low_mid'      : (250,   500,    350),
    'mid'          : (500,   2000,   1000),
    'upper_mid'    : (2000,  5000,   3500),
    'presence'     : (5000,  10000,  7000),
    'air'          : (10000, 20000,  14000),
}

def analyze_audio(input_path):
    """
    Returns a dict with:
      band_deficit_db    — how many dB each band lags the mid reference
      effective_cutoff   — Hz where codec cut the spectrum (or sr/2 if intact)
      crest_factor_db    — peak/RMS; low → heavy mastering compression
      spectral_flatness  — 0=tonal 1=noise; low → strong tonal content
      sample_rate        — original SR
    """
    if not HAS_LIBROSA:
        return _default_analysis()

    y, sr = librosa.load(input_path, sr=None, mono=False)

    # Work in mono for analysis
    y_mono = librosa.to_mono(y) if y.ndim == 2 else y

    # Full-file FFT (large window for frequency resolution)
    n_fft = 8192
    D = np.abs(librosa.stft(y_mono, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_db = librosa.amplitude_to_db(D.mean(axis=1), ref=np.max)

    # Per-band average energy (dB)
    band_energy = {}
    for name, (f_lo, f_hi, _) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energy[name] = float(mean_db[mask].mean()) if mask.any() else -60.0

    # Deficit relative to mid band (our reference point)
    mid_ref = band_energy['mid']
    band_deficit = {k: mid_ref - v for k, v in band_energy.items()}

    # Effective cutoff: first frequency above 8 kHz where energy < -50 dB
    high_mask = freqs > 8000
    effective_cutoff = sr / 2  # default: no cutoff
    if high_mask.any():
        hi_db = mean_db[high_mask]
        hi_f  = freqs[high_mask]
        below = hi_db < -50
        if below.any():
            effective_cutoff = float(hi_f[np.argmax(below)])

    # Crest factor (dB) — measures dynamic range
    rms = float(librosa.feature.rms(y=y_mono).mean())
    peak = float(np.max(np.abs(y_mono)))
    crest_factor_db = 20 * np.log10(peak / (rms + 1e-9)) if rms > 0 else 20.0

    # Spectral flatness — proxy for how tonal vs noise-like the signal is
    flatness = float(librosa.feature.spectral_flatness(y=y_mono).mean())

    return {
        'sample_rate'       : int(sr),
        'effective_cutoff'  : effective_cutoff,
        'band_energy_db'    : band_energy,
        'band_deficit_db'   : band_deficit,
        'crest_factor_db'   : crest_factor_db,
        'spectral_flatness' : flatness,
    }


def _default_analysis():
    """Used when librosa is absent — mild flat boost profile."""
    return {
        'sample_rate'       : 44100,
        'effective_cutoff'  : 16000,
        'band_energy_db'    : {k: -20.0 for k in BANDS},
        'band_deficit_db'   : {k: 2.0 for k in BANDS},
        'crest_factor_db'   : 14.0,
        'spectral_flatness' : 0.01,
    }


# ─────────────────────────────────────────────
#  STAGE 2: Adaptive parameter computation
# ─────────────────────────────────────────────

def compute_eq_bands(analysis, max_gain=5.0):
    """
    Returns list of (freq_hz, gain_db, Q) based on spectral deficit.

    Strategy:
    - Boost only what's genuinely below the mid reference
    - Cap at max_gain to avoid distortion
    - Don't boost air if the codec already cut that range
    - Weight upper-mid boost higher (perceptually most audible for detail)
    """
    deficit = analysis['band_deficit_db']
    cutoff  = analysis['effective_cutoff']

    eq = []
    for name, (f_lo, f_hi, f_center) in BANDS.items():
        d = deficit.get(name, 0.0)

        # Skip if the codec already removed this band
        if f_lo > cutoff * 0.8:
            continue

        # How much to apply: fraction of deficit, band-weighted
        weights = {
            'sub_bass'  : 0.35,
            'bass'      : 0.50,
            'low_mid'   : 0.35,
            'mid'       : 0.40,
            'upper_mid' : 0.55,   # vocals / detail — weight higher
            'presence'  : 0.40,
            'air'       : 0.30,
        }
        w = weights.get(name, 0.4)

        gain = min(d * w, max_gain)
        if gain < 0.5:
            continue   # not worth touching

        # Q: narrower for highs (more targeted), wider for lows
        q = 1.0 + (f_center / 5000) * 0.8

        eq.append((f_center, round(gain, 1), round(q, 2)))

    return eq


def compute_dynamic_boost(analysis):
    """
    Gentle expander ratio based on crest factor deficit.
    A well-mastered track has crest_factor ≥ 15 dB.
    Below that, mastering compression has reduced transient peaks.
    """
    cf = analysis['crest_factor_db']
    target_cf = 15.0
    if cf >= target_cf:
        return 1.0   # no expansion needed
    deficit = target_cf - cf
    # Map 0–10 dB deficit → ratio 1.0–1.25
    ratio = 1.0 + min(deficit, 10.0) * 0.025
    return round(ratio, 3)


# ─────────────────────────────────────────────
#  STAGE 3: ffmpeg processing chain
# ─────────────────────────────────────────────

def build_af_chain(eq_bands, dynamic_ratio, cutoff, normalize, exciter):
    """
    Constructs the -af filter string for ffmpeg.

    Chain order:
      equalizer (N bands) → aexciter → agate/expander → loudnorm
    """
    filters = []

    # 1. Adaptive parametric EQ
    for freq, gain, q in eq_bands:
        filters.append(
            f'equalizer=f={freq}:t=q:width={q}:g={gain}'
        )

    # 2. Harmonic exciter — synthesizes overtones above the codec cutoff.
    #    Uses ffmpeg's aexciter filter (libavfilter).
    #    freq: start frequency of excitation (just below cutoff)
    #    ceil: maximum excited frequency
    #    drive: harmonic richness (higher = more saturation)
    #    blend: mix ratio excited/dry
    if exciter and cutoff < 18000:
        exciter_freq = max(int(cutoff * 0.75), 6000)
        filters.append(
            f'aexciter=level_in=1:level_out=1:amount=25:'
            f'drive=7:blend=0:freq={exciter_freq}:ceil=20000'
        )

    # 3. Gentle dynamic expansion (restores transient energy lost in mastering)
    if dynamic_ratio > 1.01:
        # agate with ratio > 1 acts as a gentle expander
        # attack very fast (preserve transient onset), release slow (natural decay)
        filters.append(
            f'agate=threshold=0.02:ratio={dynamic_ratio}:'
            f'attack=0.3:release=120:makeup=1'
        )

    # 4. EBU R128 loudness normalisation (last in chain)
    if normalize:
        filters.append('loudnorm=i=-16:tp=-1:lra=11:print_format=none')

    return ','.join(filters)


def run_ffmpeg(input_path, output_path, af_chain):
    """Run ffmpeg with the assembled filter chain, output to 24-bit WAV."""
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-af', af_chain,
        '-acodec', 'pcm_s24le',
        '-ar', '44100',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors='replace')[-300:]
        print(f"  [ffmpeg error] {os.path.basename(input_path)}: {err}")
        return False
    return True


# ─────────────────────────────────────────────
#  STAGE 4: SoX VHQ upsampling
# ─────────────────────────────────────────────

def sox_upsample(input_path, output_path, target_rate):
    """
    SoX very-high-quality resampling.

    Flags:
      rate -v   : very high quality (Kaiser window, β≈14)
      rate -s   : steep rolloff (maximises passband before Nyquist)
      dither -s : TPDF (triangular probability density function) dither
                  — makes quantisation error spectrally flat white noise
                  instead of harmonic distortion tones
    """
    cmd = [
        'sox', input_path,
        '-b', '24',
        output_path,
        'rate', '-v', '-s', str(target_rate),
        'dither', '-s'
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors='replace')[-200:]
        print(f"  [sox error] {os.path.basename(input_path)}: {err}")
        return False
    return True


# ─────────────────────────────────────────────
#  Worker (runs in each pool process)
# ─────────────────────────────────────────────

# These are set as module-level globals so pool workers can access them
# without pickling the full args object each time.
_g_args = None

def _init_worker(args):
    global _g_args
    _g_args = args


def process_one(input_path):
    """Full pipeline for a single file."""
    args = _g_args
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_name = stem + ('_hires.wav' if args.hires else '_restored.wav')
    final_out = os.path.join(args.output_dir, out_name)

    print(f"  → {os.path.basename(input_path)}")

    # Stage 1: Analysis
    analysis = analyze_audio(input_path)

    cutoff  = analysis['effective_cutoff']
    cf      = analysis['crest_factor_db']
    deficit = analysis['band_deficit_db']

    if args.verbose:
        print(f"     cutoff={cutoff:.0f} Hz  crest={cf:.1f} dB  "
              f"flatness={analysis['spectral_flatness']:.4f}")
        for k, v in deficit.items():
            print(f"       {k:12s} deficit={v:+.1f} dB")

    # Stage 2: Compute parameters
    eq_bands       = compute_eq_bands(analysis, max_gain=args.max_gain)
    dynamic_ratio  = compute_dynamic_boost(analysis)

    # Stage 3: ffmpeg
    af = build_af_chain(eq_bands, dynamic_ratio, cutoff,
                        normalize=args.normalize, exciter=args.exciter)

    if args.hires:
        # Write ffmpeg output to a temp file, then SoX upsamples it
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
            tmp = tf.name
        ok = run_ffmpeg(input_path, tmp, af)
        if not ok:
            _cleanup(tmp)
            return

        # Stage 4: SoX upsample
        ok = sox_upsample(tmp, final_out, args.target_rate)
        _cleanup(tmp)
        if not ok:
            return
    else:
        run_ffmpeg(input_path, final_out, af)


def _cleanup(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

AUDIO_EXTS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.aiff', '.opus')

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive audio restoration: analyze → EQ → excite → expand → upsample',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full restoration with hi-res output
  python adaptive_restore.py in/ out/ --hires

  # Conservative restoration, no exciter, 48 kHz
  python adaptive_restore.py in/ out/ --hires --target_rate 48000 --no-exciter --max_gain 3

  # Just EQ correction, no upsample
  python adaptive_restore.py in/ out/

  # All options, verbose, 12 workers
  python adaptive_restore.py in/ out/ --hires --target_rate 96000 --max_gain 5 --workers 12 -v
        """
    )

    parser.add_argument('input_dir',  help='Folder with source audio files')
    parser.add_argument('output_dir', help='Folder for processed output files')

    g = parser.add_argument_group('quality')
    g.add_argument('--hires', action='store_true', default=False,
                   help='Upsample output via SoX VHQ (requires sox installed)')
    g.add_argument('--target_rate', type=int, default=96000,
                   choices=[44100, 48000, 88200, 96000, 176400, 192000],
                   help='Target sample rate for hi-res output (default: 96000)')
    g.add_argument('--max_gain', type=float, default=5.0,
                   help='Maximum EQ boost in dB per band (default: 5.0)')

    g2 = parser.add_argument_group('effects')
    g2.add_argument('--normalize', action='store_true', default=True,
                    help='Apply EBU R128 loudness normalisation (default: on)')
    g2.add_argument('--no-normalize', dest='normalize', action='store_false')
    g2.add_argument('--exciter', action='store_true', default=True,
                    help='Harmonic exciter above codec cutoff (default: on)')
    g2.add_argument('--no-exciter', dest='exciter', action='store_false')

    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='Parallel workers (default: cpu count)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print per-file analysis details')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(AUDIO_EXTS)
    ]

    if not files:
        print(f"No audio files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files. Workers: {args.workers}. Hi-res: {args.hires}")
    if args.hires:
        print(f"Target: {args.target_rate} Hz, 24-bit WAV via SoX VHQ")
    print()

    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args,)
    ) as pool:
        pool.map(process_one, files)

    print("\nDone.")


if __name__ == '__main__':
    main()
