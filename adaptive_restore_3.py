#!/usr/bin/env python3
"""
adaptive_restore.py — Adaptive Audio Restoration Pipeline (improved)
====================================================================
Improvements over original:
  1. Spectral flatness is used to modulate EQ aggressiveness and exciter strength
  2. Temporal analysis: track is split into segments → percentile energy + variance
  3. Stereo analysis: separate L/R energy + mid-side balance detection
  4. Better spectral reference: target curve (approximate equal-loudness / pink-ish)
     instead of blindly assuming the mid band is "correct"
  5. Input format / sample rate / channels are detected and preserved by default
     (only changed when --hires or explicit --target_rate / --force_format)

Install deps:
  pip install librosa soundfile numpy
  apt install sox ffmpeg          # or brew install

Usage examples:
  python adaptive_restore.py input/ output/ --hires
  python adaptive_restore.py input/ output/ --target_rate 48000 --no-exciter
  python adaptive_restore.py input/ output/ --workers 8 --preserve
  python adaptive_restore.py input/ output/ --hires --target_rate 96000 --max_gain 5 -v
"""

import os

# One BLAS/OpenMP/numba thread per worker — we already parallelise across files.
# Must be set before numpy/librosa are imported.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import subprocess
import argparse
import multiprocessing
import tempfile
import json
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠  librosa/soundfile not found — install with: pip install librosa soundfile")
    print("   Falling back to flat EQ without spectral analysis.\n")


# ─────────────────────────────────────────────
#  Constants
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

# Approximate target spectral shape relative to mid (0 dB).
# Roughly inspired by equal-loudness contours + mild pink tilt.
# Positive = we expect this band to be louder than mid; negative = quieter.
TARGET_RELATIVE_DB = {
    'sub_bass'  : -4.0,
    'bass'      : -1.5,
    'low_mid'   : -0.5,
    'mid'       :  0.0,
    'upper_mid' : +0.5,
    'presence'  : -1.0,
    'air'       : -3.0,
}

AUDIO_EXTS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.aiff', '.aif', '.opus', '.wma')

# Preferred output codecs when we must re-encode
FORMAT_MAP = {
    '.wav'  : ('pcm_s24le', None),
    '.flac' : ('flac',      None),
    '.mp3'  : ('libmp3lame', '-b:a 320k'),
    '.ogg'  : ('libvorbis', '-q:a 6'),
    '.opus' : ('libopus',   '-b:a 192k'),
    '.m4a'  : ('aac',       '-b:a 256k'),
    '.aac'  : ('aac',       '-b:a 256k'),
    '.aiff' : ('pcm_s24be', None),
    '.aif'  : ('pcm_s24be', None),
}


# ─────────────────────────────────────────────
#  Probe original file parameters (ffprobe)
# ─────────────────────────────────────────────

def probe_file(path):
    """
    Return dict with: sample_rate, channels, bit_depth, codec, duration, format
    Uses ffprobe so we never lose original metadata.
    """
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(out)
    except Exception:
        return {
            'sample_rate': 44100,
            'channels': 2,
            'bit_depth': 16,
            'codec': 'pcm_s16le',
            'duration': 0.0,
            'format': Path(path).suffix.lower(),
            'bits_per_sample': 16,
        }

    stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), {})
    fmt    = data.get('format', {})

    sr = int(stream.get('sample_rate', 44100))
    ch = int(stream.get('channels', 2))
    codec = stream.get('codec_name', 'unknown')
    duration = float(fmt.get('duration', 0) or 0)

    # bit depth heuristics
    bps = stream.get('bits_per_sample') or stream.get('bits_per_raw_sample')
    if bps:
        bit_depth = int(bps)
    else:
        if codec in ('pcm_s24le', 'pcm_s24be', 'pcm_s24'):
            bit_depth = 24
        elif codec in ('pcm_s32le', 'pcm_f32le', 'flac'):
            bit_depth = 24
        elif codec in ('pcm_s16le', 'pcm_s16be'):
            bit_depth = 16
        else:
            bit_depth = 16

    return {
        'sample_rate'    : sr,
        'channels'       : ch,
        'bit_depth'      : bit_depth,
        'codec'          : codec,
        'duration'       : duration,
        'format'         : Path(path).suffix.lower(),
        'bits_per_sample': bit_depth,
    }


# ─────────────────────────────────────────────
#  STAGE 1: Spectral + temporal + stereo analysis
# ─────────────────────────────────────────────

# Analysis block geometry: STFT frames are laid out continuously across blocks,
# so a block is n_fft + hop*(FRAMES_PER_BLOCK-1) samples and consecutive blocks
# overlap by (n_fft - hop).  ~6 s per block at 44.1 kHz → a few MB of RAM.
_N_FFT  = 8192
_HOP    = _N_FFT // 4
_FRAMES_PER_BLOCK = 128
_BLOCK  = _N_FFT + _HOP * (_FRAMES_PER_BLOCK - 1)
_OVERLAP = _N_FFT - _HOP


def _read_exact(stream, nbytes):
    """Read exactly nbytes from a pipe (or fewer at EOF)."""
    chunks, got = [], 0
    while got < nbytes:
        b = stream.read(nbytes - got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b''.join(chunks)


def _stream_blocks(path, sr, n_ch):
    """
    Decode `path` through ffmpeg → raw float32 on a pipe and yield blocks of shape
    (n_ch, _BLOCK) with _OVERLAP samples shared between consecutive blocks.
    The whole file is never held in RAM (works for every format ffmpeg reads,
    not only the ones libsndfile supports).
    """
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-vn',
           '-f', 'f32le', '-acodec', 'pcm_f32le',
           '-ac', str(n_ch), '-ar', str(sr), '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fbytes = 4 * n_ch
    step = _BLOCK - _OVERLAP
    prev = None
    try:
        while True:
            want = _BLOCK if prev is None else step
            raw = _read_exact(proc.stdout, want * fbytes)
            raw = raw[: len(raw) - (len(raw) % fbytes)]
            if not raw:
                break
            new = np.frombuffer(raw, dtype=np.float32).reshape(-1, n_ch)
            blk = new if prev is None else np.concatenate((prev[-_OVERLAP:], new), axis=0)
            yield np.ascontiguousarray(blk.T)
            if new.shape[0] < want:       # EOF reached inside this read
                break
            prev = blk
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def analyze_audio(input_path, n_segments=8, probe=None):
    """
    Returns a rich analysis dict.

    Streaming implementation: the file is decoded in ~6 s blocks and only running
    statistics are kept (mean magnitude spectra, one summed spectrum per time
    segment, a few scalar accumulators).  Peak RAM is a few tens of MB per file
    regardless of track length — the old version loaded the full float32 signal and
    full-length STFTs (≈ 10 GB for a 1 h stereo MP3).

      • Temporal: energy percentiles + variance across time segments
      • Stereo: separate L/R band energy + mid-side imbalance
      • Spectral flatness used later for aggressiveness
      • Target-curve based deficit (not pure mid-reference)
    """
    if not HAS_LIBROSA:
        return _default_analysis()

    probe = probe or probe_file(input_path)
    sr   = int(probe.get('sample_rate') or 44100)
    n_ch = max(1, min(int(probe.get('channels') or 2), 2))   # analysis uses ≤ 2 channels

    n_fft, hop = _N_FFT, _HOP
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_bins = len(freqs)

    # running accumulators (float64)
    mag_sum  = np.zeros(n_bins)
    mag_cnt  = 0
    l_sum    = np.zeros(n_bins)
    r_sum    = np.zeros(n_bins)
    seg_sum  = np.zeros((n_segments, n_bins))   # per time-segment summed magnitudes
    seg_cnt  = np.zeros(n_segments)
    # Expected length → segment boundaries (frame-exact, constant memory).
    # If the container reports no duration everything lands in segment 0
    # (temporal variance then reads 0 → no penalty).
    est_total = max(int(float(probe.get('duration') or 0) * sr), 0)
    seg_len   = max(est_total // n_segments, 1) if est_total else 1 << 62
    blk_idx   = 0
    peak     = 0.0
    rms_sum, rms_cnt   = 0.0, 0
    flat_sum, flat_cnt = 0.0, 0
    l2 = r2 = m2 = s2 = 0.0
    n_samples = 0
    first = True

    for blk in _stream_blocks(input_path, sr, n_ch):
        n = blk.shape[1]
        lo = 0 if first else _OVERLAP          # count each sample once (skip shared overlap)
        first = False
        n_samples += n - lo
        mono = blk.mean(axis=0) if n_ch > 1 else blk[0]

        peak = max(peak, float(np.max(np.abs(mono))))

        if n >= 2048:
            rms_f = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512, center=False)
            rms_sum += float(rms_f.sum()); rms_cnt += rms_f.shape[1]
            fl = librosa.feature.spectral_flatness(y=mono, n_fft=2048, hop_length=512, center=False)
            flat_sum += float(fl.sum()); flat_cnt += fl.shape[1]

        if n >= n_fft:
            D = np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop, center=False))
            s = D.sum(axis=1)
            mag_sum += s
            mag_cnt += D.shape[1]
            g_start = np.arange(D.shape[1], dtype=np.int64) + blk_idx * _FRAMES_PER_BLOCK
            seg_ids = np.minimum(g_start * hop // seg_len, n_segments - 1)
            for sid in np.unique(seg_ids):
                sel = seg_ids == sid
                seg_sum[sid] += D[:, sel].sum(axis=1)
                seg_cnt[sid] += int(sel.sum())
            if n_ch >= 2:
                l_sum += np.abs(librosa.stft(blk[0], n_fft=n_fft, hop_length=hop, center=False)).sum(axis=1)
                r_sum += np.abs(librosa.stft(blk[1], n_fft=n_fft, hop_length=hop, center=False)).sum(axis=1)
            del D
        blk_idx += 1

        if n_ch >= 2:
            l = blk[0, lo:].astype(np.float64); r = blk[1, lo:].astype(np.float64)
            l2 += float(np.dot(l, l)); r2 += float(np.dot(r, r))
            m = (l + r) * 0.5;  sd = (l - r) * 0.5
            m2 += float(np.dot(m, m)); s2 += float(np.dot(sd, sd))

    if mag_cnt == 0:                       # file too short / undecodable
        return _default_analysis()

    mean_db = librosa.amplitude_to_db(mag_sum / mag_cnt, ref=np.max)

    # ── Per-band average energy ───────────────────────────────────────
    band_energy = {}
    for name, (f_lo, f_hi, _) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energy[name] = float(mean_db[mask].mean()) if mask.any() else -80.0

    # ── Target-curve deficit ──────────────────────────────────────────
    mid_energy = band_energy['mid']
    band_deficit = {}
    for name, energy in band_energy.items():
        target = mid_energy + TARGET_RELATIVE_DB.get(name, 0.0)
        band_deficit[name] = target - energy          # positive → needs boost

    # ── Effective codec cutoff ────────────────────────────────────────
    high_mask = freqs > 8000
    effective_cutoff = float(sr / 2)
    if high_mask.any():
        hi_db = mean_db[high_mask]
        hi_f  = freqs[high_mask]
        below = hi_db < -50
        if below.any():
            effective_cutoff = float(hi_f[np.argmax(below)])

    # ── Crest factor / flatness ───────────────────────────────────────
    rms  = rms_sum / rms_cnt if rms_cnt else 0.0
    crest_factor_db = 20 * np.log10(peak / (rms + 1e-12)) if rms > 0 else 20.0
    flatness = float(flat_sum / flat_cnt) if flat_cnt else 0.0

    # ── TEMPORAL ANALYSIS ─────────────────────────────────────────────
    temporal = {name: {'mean': [], 'p20': [], 'p80': []} for name in BANDS}
    for sid in range(n_segments):
        if seg_cnt[sid] == 0:
            continue
        seg_db = librosa.amplitude_to_db(seg_sum[sid] / seg_cnt[sid], ref=np.max)
        for name, (f_lo, f_hi, _) in BANDS.items():
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.any():
                vals = seg_db[mask]
                temporal[name]['mean'].append(float(vals.mean()))
                temporal[name]['p20'].append(float(np.percentile(vals, 20)))
                temporal[name]['p80'].append(float(np.percentile(vals, 80)))

    band_temporal_var = {}
    band_dynamic_range = {}          # p80 - p20
    for name in BANDS:
        means = temporal[name]['mean']
        if len(means) > 1:
            band_temporal_var[name] = float(np.var(means))
            band_dynamic_range[name] = float(np.mean(temporal[name]['p80']) -
                                             np.mean(temporal[name]['p20']))
        else:
            band_temporal_var[name] = 0.0
            band_dynamic_range[name] = 0.0

    # ── STEREO / MID-SIDE ANALYSIS ────────────────────────────────────
    stereo_imbalance_db = 0.0
    side_energy_ratio   = 0.0
    lr_band_diff        = {name: 0.0 for name in BANDS}

    if n_ch >= 2 and n_samples > 0:
        l_rms = float(np.sqrt(l2 / n_samples) + 1e-12)
        r_rms = float(np.sqrt(r2 / n_samples) + 1e-12)
        stereo_imbalance_db = 20 * np.log10(l_rms / r_rms)

        mid_rms  = float(np.sqrt(m2 / n_samples) + 1e-12)
        side_rms = float(np.sqrt(s2 / n_samples) + 1e-12)
        side_energy_ratio = side_rms / mid_rms

        left_db  = librosa.amplitude_to_db(l_sum / mag_cnt, ref=np.max)
        right_db = librosa.amplitude_to_db(r_sum / mag_cnt, ref=np.max)
        for name, (f_lo, f_hi, _) in BANDS.items():
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.any():
                lr_band_diff[name] = float(left_db[mask].mean() - right_db[mask].mean())

    return {
        'sample_rate'         : int(sr),
        'channels'            : int(probe.get('channels') or n_ch),
        'effective_cutoff'    : effective_cutoff,
        'band_energy_db'      : band_energy,
        'band_deficit_db'     : band_deficit,
        'crest_factor_db'     : crest_factor_db,
        'spectral_flatness'   : flatness,
        # temporal
        'band_temporal_var'   : band_temporal_var,
        'band_dynamic_range'  : band_dynamic_range,
        # stereo
        'stereo_imbalance_db' : stereo_imbalance_db,
        'side_energy_ratio'   : side_energy_ratio,
        'lr_band_diff'        : lr_band_diff,
    }


def _default_analysis():
    """Fallback when librosa is missing."""
    return {
        'sample_rate'         : 44100,
        'channels'            : 2,
        'effective_cutoff'    : 16000,
        'band_energy_db'      : {k: -20.0 for k in BANDS},
        'band_deficit_db'     : {k: 2.0 for k in BANDS},
        'crest_factor_db'     : 14.0,
        'spectral_flatness'   : 0.02,
        'band_temporal_var'   : {k: 0.0 for k in BANDS},
        'band_dynamic_range'  : {k: 6.0 for k in BANDS},
        'stereo_imbalance_db' : 0.0,
        'side_energy_ratio'   : 0.15,
        'lr_band_diff'        : {k: 0.0 for k in BANDS},
    }


# ─────────────────────────────────────────────
#  STAGE 2: Adaptive parameter computation
# ─────────────────────────────────────────────

def compute_eq_bands(analysis, max_gain=5.0):
    """
    Returns list of (freq_hz, gain_db, Q).

    Improvements:
      • Target-curve deficit (already computed)
      • Spectral flatness modulates overall aggressiveness
        (tonal material → more aggressive; noisy → more cautious)
      • Temporal variance reduces gain on highly dynamic / sparse bands
        (avoids pumping on transient-heavy material)
      • Upper-mid still weighted higher (perceptual detail)
    """
    deficit   = analysis['band_deficit_db']
    cutoff    = analysis['effective_cutoff']
    flatness  = analysis['spectral_flatness']
    temp_var  = analysis['band_temporal_var']

    # Flatness scaling: 0.0 (tonal) → 1.0, 0.15+ (noisy) → ~0.55
    # More tonal → allow stronger corrective EQ
    flat_scale = 1.0 - min(flatness * 3.0, 0.45)

    weights = {
        'sub_bass'  : 0.30,
        'bass'      : 0.45,
        'low_mid'   : 0.35,
        'mid'       : 0.30,
        'upper_mid' : 0.55,   # vocals / detail
        'presence'  : 0.40,
        'air'       : 0.25,
    }

    eq = []
    for name, (f_lo, f_hi, f_center) in BANDS.items():
        d = deficit.get(name, 0.0)

        # Skip bands already removed by codec
        if f_lo > cutoff * 0.8:
            continue

        # Temporal variance penalty (high variance → lower gain)
        var = temp_var.get(name, 0.0)
        var_penalty = 1.0 / (1.0 + var * 0.08)   # soft roll-off

        w = weights.get(name, 0.4)
        gain = d * w * flat_scale * var_penalty
        gain = min(max(gain, 0.0), max_gain)      # clamp

        if gain < 0.4:
            continue

        # Q: narrower for highs, wider for lows
        q = 0.9 + (f_center / 6000.0) * 1.1
        q = min(max(q, 0.7), 2.8)

        eq.append((f_center, round(gain, 1), round(q, 2)))

    return eq


def compute_dynamic_boost(analysis):
    """
    Gentle expander ratio based on crest factor.
    Target crest ≈ 14–16 dB for most modern material.
    """
    cf = analysis['crest_factor_db']
    target_cf = 15.0
    if cf >= target_cf:
        return 1.0
    deficit = target_cf - cf
    # 0–12 dB deficit → ratio 1.0–1.30
    ratio = 1.0 + min(deficit, 12.0) * 0.025
    return round(ratio, 3)


def compute_exciter_amount(analysis):
    """
    Spectral flatness + cutoff decide how strong the harmonic exciter should be.
    Tonal material with low cutoff → stronger excitation.
    Already noisy / full-bandwidth material → weaker or off.
    """
    flatness = analysis['spectral_flatness']
    cutoff   = analysis['effective_cutoff']

    if cutoff >= 18000:
        return 0.0          # no point exciting already full spectrum

    # Base amount inversely related to flatness
    base = 28.0 * (1.0 - min(flatness * 4.0, 0.7))
    # Stronger when cutoff is lower (more missing air)
    cutoff_factor = max(0.3, 1.0 - (cutoff - 8000) / 12000)
    amount = base * cutoff_factor
    return round(max(0.0, min(amount, 40.0)), 1)


# ─────────────────────────────────────────────
#  STAGE 3: ffmpeg filter chain
# ─────────────────────────────────────────────

def build_af_chain(eq_bands, dynamic_ratio, cutoff, normalize, exciter, exciter_amount,
                   stereo_imbalance_db=0.0):
    """
    Builds the -af filter string.

    Order:
      optional stereo balance → equalizers → aexciter → expander → loudnorm
    """
    filters = []

    # Optional gentle L/R balance correction (only if > 1.5 dB imbalance)
    # Use pan filter — more reliable than volume=c0/c1 across ffmpeg versions
    if abs(stereo_imbalance_db) > 1.5:
        corr = min(abs(stereo_imbalance_db) * 0.4, 3.0)   # max 3 dB correction
        if stereo_imbalance_db > 0:
            # Left louder → reduce left
            lin_l = 10 ** (-corr / 20.0)
            lin_r = 1.0
        else:
            lin_l = 1.0
            lin_r = 10 ** (-corr / 20.0)
        filters.append(
            f'pan=stereo|c0={lin_l:.4f}*c0|c1={lin_r:.4f}*c1'
        )

    # Adaptive parametric EQ
    for freq, gain, q in eq_bands:
        filters.append(f'equalizer=f={freq}:t=q:width={q}:g={gain}')

    # Harmonic exciter (only when useful)
    if exciter and exciter_amount > 1.0 and cutoff < 18000:
        exciter_freq = min(max(int(cutoff * 0.72), 2000), 12000)
        filters.append(
            f'aexciter=level_in=1:level_out=1:amount={exciter_amount}:'
            f'drive=6.5:blend=0:freq={exciter_freq}:ceil=20000'
        )

    # Gentle dynamic expansion (upward gate = expander)
    if dynamic_ratio > 1.02:
        filters.append(
            f'agate=mode=upward:threshold=0.018:ratio={dynamic_ratio}:'
            f'attack=1:release=140:makeup=1'
        )

    # EBU R128 loudness normalisation
    if normalize:
        filters.append('loudnorm=i=-16:tp=-1.5:lra=11:print_format=none')

    return ','.join(filters) if filters else 'anull'


def run_ffmpeg(input_path, output_path, af_chain, probe, force_format=None,
               target_rate=None, preserve=True):
    """
    Always render filters to a temporary 24-bit WAV, then (re)encode to the
    desired output format.
    """
    out_ext = Path(output_path).suffix.lower()
    in_ext  = probe['format']

    if target_rate:
        ar = int(target_rate)
    elif preserve:
        ar = int(probe['sample_rate'])
    else:
        ar = 44100

    if force_format:
        codec, extra = FORMAT_MAP.get(force_format, ('pcm_s24le', None))
        output_path = str(Path(output_path).with_suffix(force_format))
        out_ext = force_format
    else:
        if out_ext in FORMAT_MAP:
            codec, extra = FORMAT_MAP[out_ext]
        elif in_ext in FORMAT_MAP:
            codec, extra = FORMAT_MAP[in_ext]
            output_path = str(Path(output_path).with_suffix(in_ext))
            out_ext = in_ext
        else:
            codec, extra = ('pcm_s24le', None)
            output_path = str(Path(output_path).with_suffix('.wav'))
            out_ext = '.wav'

    channels = max(int(probe['channels']), 1)

    # Clamp sample rate for formats that don't support high rates
    MP3_RATES = {8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000}
    if out_ext in {'.mp3', '.m4a', '.aac'} and ar not in MP3_RATES:
        # pick closest valid rate
        ar = min(MP3_RATES, key=lambda r: abs(r - ar))
        if ar > 48000:
            ar = 48000

    def _run(cmd, label=''):
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr.decode(errors='replace')[-800:]
            print(f"  [ffmpeg error{' ' + label if label else ''}] {os.path.basename(input_path)}:")
            print(f"    cmd: {' '.join(cmd[:12])}...")
            print(f"    {err.strip()}")
            return False
        return True

    # Unique temp path next to output (avoids /tmp permission / race issues)
    tmp_dir = str(Path(output_path).parent)
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix='.wav', prefix='.ar_', dir=tmp_dir)
    os.close(fd)

    try:
        # Stage A: filters → PCM WAV
        cmd_pcm = [
            'ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', input_path,
            '-af', af_chain,
            '-ar', str(ar),
            '-ac', str(channels),
            '-sample_fmt', 's32',
            '-c:a', 'pcm_s24le',
            tmp,
        ]
        ok = _run(cmd_pcm, 'filter→wav')
        if not ok:
            # Retry without fragile filters
            safe = [f for f in af_chain.split(',')
                    if not f.startswith(('aexciter', 'agate', 'pan='))]
            safe_chain = ','.join(safe) if safe else 'anull'
            print(f"  ↳ retrying without aexciter/agate/pan …")
            cmd_pcm[cmd_pcm.index('-af') + 1] = safe_chain
            ok = _run(cmd_pcm, 'filter→wav-safe')
            if not ok:
                return False, output_path

        size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if size < 1024:
            print(f"  [error] temp WAV too small ({size} bytes) — filter chain produced no audio")
            return False, output_path

        # Stage B: already WAV and want WAV → just rename
        if out_ext == '.wav' and codec.startswith('pcm_'):
            try:
                os.replace(tmp, output_path)
                tmp = None  # ownership transferred
            except OSError:
                import shutil
                shutil.copy2(tmp, output_path)
            return True, output_path

        # Stage B: re-encode
        cmd_enc = [
            'ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', tmp,
            '-ar', str(ar),
            '-ac', str(channels),
            '-c:a', codec,
        ]
        if extra:
            cmd_enc.extend(extra.split())
        if codec.startswith('pcm_'):
            idx = cmd_enc.index('-c:a') + 1
            cmd_enc[idx] = 'pcm_s24le' if probe['bit_depth'] >= 24 else 'pcm_s16le'
        cmd_enc.append(output_path)

        ok = _run(cmd_enc, 'wav→out')
        return ok, output_path
    finally:
        if tmp and os.path.exists(tmp):
            _cleanup(tmp)





# ─────────────────────────────────────────────
#  STAGE 4: SoX VHQ upsampling (optional)
# ─────────────────────────────────────────────

def sox_upsample(input_path, output_path, target_rate):
    """
    SoX very-high-quality resampling + TPDF dither.
    """
    cmd = [
        'sox', input_path, '-b', '24', output_path,
        'rate', '-v', '-s', str(target_rate),
        'dither', '-s'
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors='replace')[-300:]
        print(f"  [sox error] {os.path.basename(input_path)}: {err}")
        return False
    return True


# ─────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────

_args = None

def _init_worker(args):
    global _args
    _args = args


def process_one(input_path):
    try:
        _process_one(input_path)
    except Exception as e:                       # keep the pool alive
        print(f"  [error] {os.path.basename(input_path)}: {type(e).__name__}: {e}")


def _process_one(input_path):
    args = _args
    name = os.path.basename(input_path)
    final_out = os.path.join(args.output_dir, name)

    # Probe original parameters first
    probe = probe_file(input_path)

    if args.verbose:
        print(f"→ {name}  |  {probe['sample_rate']} Hz  {probe['channels']}ch  "
              f"{probe['bit_depth']}bit  {probe['codec']}")

    # Analysis
    analysis = analyze_audio(input_path, probe=probe)

    if args.verbose:
        print(f"   cutoff={analysis['effective_cutoff']:.0f} Hz  "
              f"crest={analysis['crest_factor_db']:.1f} dB  "
              f"flatness={analysis['spectral_flatness']:.4f}  "
              f"L/R imbalance={analysis['stereo_imbalance_db']:+.2f} dB")
        deficits = analysis['band_deficit_db']
        print("   deficits:",
              "  ".join(f"{k}={v:+.1f}" for k, v in deficits.items()))

    # Adaptive parameters
    eq_bands       = compute_eq_bands(analysis, max_gain=args.max_gain)
    dynamic_ratio  = compute_dynamic_boost(analysis)
    exciter_amount = compute_exciter_amount(analysis) if args.exciter else 0.0

    if args.verbose and eq_bands:
        print("   EQ:", "  ".join(f"{f:.0f}Hz/{g:+.1f}dB/Q{q}" for f, g, q in eq_bands))
        print(f"   expander ratio={dynamic_ratio}  exciter={exciter_amount}")

    af = build_af_chain(
        eq_bands,
        dynamic_ratio,
        analysis['effective_cutoff'],
        args.normalize,
        args.exciter,
        exciter_amount,
        stereo_imbalance_db=analysis['stereo_imbalance_db'],
    )

    # Decide target rate — only change when --hires or user explicitly passed --target_rate
    # (default value of --target_rate must NOT force a rate change without --hires)
    target_rate = None
    if args.hires:
        target_rate = args.target_rate if args.target_rate else 96000
    elif getattr(args, 'target_rate_explicit', False):
        target_rate = args.target_rate

    if args.hires:
        # ffmpeg → temp (preserve rate for now) → sox upsample
        # NB: keep the temp WAV in the output dir — /tmp is often tmpfs (= RAM)
        with tempfile.NamedTemporaryFile(suffix='.wav', prefix='.ar_hr_', dir=args.output_dir,
                                         delete=False) as tf:
            tmp = tf.name
        ok, _ = run_ffmpeg(
            input_path, tmp, af, probe,
            force_format='.wav',
            target_rate=None,          # keep original until sox
            preserve=True
        )
        if not ok:
            _cleanup(tmp)
            return

        # Force final name to .wav for hi-res
        final_out = str(Path(final_out).with_suffix('.wav'))
        ok = sox_upsample(tmp, final_out, args.target_rate)
        _cleanup(tmp)
        if not ok:
            return
    else:
        ok, final_out = run_ffmpeg(
            input_path, final_out, af, probe,
            target_rate=target_rate,
            preserve=args.preserve
        )
        if not ok:
            return

    if args.verbose:
        print(f"   ✓ {os.path.basename(final_out)}")


def _cleanup(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Adaptive audio restoration: analyze → EQ → excite → expand → (optional upsample)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full restoration with hi-res output (96 kHz 24-bit)
  python adaptive_restore.py in/ out/ --hires

  # Preserve original sample rate, format and channels
  python adaptive_restore.py in/ out/ --preserve

  # Conservative, no exciter, force 48 kHz
  python adaptive_restore.py in/ out/ --target_rate 48000 --no-exciter --max_gain 3

  # Verbose, 12 workers
  python adaptive_restore.py in/ out/ --hires --workers 12 -v
        """
    )

    parser.add_argument('input_dir',  help='Folder with source audio files')
    parser.add_argument('output_dir', help='Folder for processed output files')

    g = parser.add_argument_group('quality / format')
    g.add_argument('--hires', action='store_true', default=False,
                   help='Upsample via SoX VHQ to --target_rate (24-bit WAV)')
    g.add_argument('--target_rate', type=int, default=None,
                   choices=[44100, 48000, 88200, 96000, 176400, 192000],
                   help='Target sample rate (only applied with --hires, or when set explicitly)')
    g.add_argument('--preserve', action='store_true', default=True,
                   help='Preserve original sample rate, channels and container when possible (default: on)')
    g.add_argument('--no-preserve', dest='preserve', action='store_false',
                   help='Do not try to preserve original format parameters')
    g.add_argument('--max_gain', type=float, default=5.0,
                   help='Maximum EQ boost in dB per band (default: 5.0)')

    g2 = parser.add_argument_group('effects')
    g2.add_argument('--normalize', action='store_true', default=True,
                    help='EBU R128 loudness normalisation (default: on)')
    g2.add_argument('--no-normalize', dest='normalize', action='store_false')
    g2.add_argument('--exciter', action='store_true', default=True,
                    help='Harmonic exciter above codec cutoff (default: on)')
    g2.add_argument('--no-exciter', dest='exciter', action='store_false')

    parser.add_argument('--workers', type=int, default=min(multiprocessing.cpu_count(), 4),
                        help='Parallel workers (default: min(cpu count, 4); each worker = '
                             'python+librosa ≈ 200 MB + 2 ffmpeg processes)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print per-file analysis details')

    args = parser.parse_args()

    # Track whether user explicitly passed --target_rate
    args.target_rate_explicit = args.target_rate is not None
    if args.hires and args.target_rate is None:
        args.target_rate = 96000
    if not args.hires and args.target_rate is None:
        args.target_rate = 0  # unused

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(AUDIO_EXTS)
    ])

    if not files:
        print(f"No audio files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files · workers={args.workers} · "
          f"hires={args.hires} · preserve={args.preserve}")
    if args.hires:
        print(f"Target: {args.target_rate} Hz 24-bit via SoX VHQ")
    print()

    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args,)
    ) as pool:
        for _ in pool.imap_unordered(process_one, files, chunksize=1):
            pass

    print("\nDone.")


if __name__ == '__main__':
    main()
