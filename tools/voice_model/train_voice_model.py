#!/usr/bin/env python3
"""
NeuralSafe voice model trainer (Normal / Elevated / HighDanger).

Why this exists: the shipped v2 model was trained on RAVDESS (studio-recorded
acted speech) only. Diagnosis on 2026-10-05 showed it outputs HighDanger with
~100% confidence for plain white noise at any volume, i.e. it defaults to
"danger" on anything that doesn't sound like RAVDESS. This script retrains on a
much more diverse mix and reports honest held-out metrics.

DATA SOURCES (class -> index: normal=0, elevated=1, danger=2)
  own        YOUR recordings from the app's "Voice Data Recorder", folders
             <own>/normal, <own>/elevated, <own>/danger (WAV, 16 kHz mono).
             Most important source: it is the real phone mic.
  ravdess    RAVDESS speech. neutral/calm/happy/sad -> normal;
             angry/fearful -> elevated (or danger if strong intensity and
             --ravdess-strong-as-danger); disgust/surprised skipped.
  speech     mini_speech_commands (real people, many mics/rooms) -> normal.
  esc50      ESC-50 environmental sounds. crying_baby -> elevated,
             glass_breaking -> danger, fireworks skipped, rest -> normal.
  synthetic  quiet low-level noise windows -> normal (teaches the model that
             the per-window peak-normalised log-Mel of faint noise is NOT danger).

FEATURES are bit-compatible with the Flutter app (MelSpectrogramExtractor):
n_fft 2048, hop 512, 128 mels from assets/ml/mel_filterbank.bin, Hann window,
zero padding, 10*log10 power, subtract per-window max, clip at -80 dB, then
(x+80)/80. A parity check against assets/ml/expected_log_mel.bin runs first.
Windows quieter than SILENCE_RMS are never trained on because the app's
silence gate (kSilenceRmsThreshold in voice_analysis_service.dart) never
sends them to the model.

COLAB QUICKSTART
  !pip -q install soundfile
  # upload: mel_filterbank.bin, expected_log_mel.bin, test_audio_window.bin
  #         (from assets/ml/) and a zip of the phone's voice_data folder
  !unzip -q voice_data.zip -d /content
  !python train_voice_model.py --own /content/voice_data \
        --filterbank /content/mel_filterbank.bin
  # download voice_out/neuralsafe_voice_model_v3.tflite + voice_model_report.txt

LOCAL SELF-TEST (no datasets, no network):  python train_voice_model.py --selftest
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from glob import glob
from pathlib import Path

import numpy as np

SR = 16000
N_FFT = 2048
HOP = 512
N_MELS = 128
WIN = SR
CLASSES = ["normal", "elevated", "danger"]
SILENCE_RMS = 0.005  # keep in sync with kSilenceRmsThreshold in the app

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
SPEECH_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"

ESC_ELEVATED = {"crying_baby"}
ESC_DANGER = {"glass_breaking"}
ESC_SKIP = {"fireworks"}

# (source, label) -> number of augmented copies per TRAIN window
AUG_COPIES = {
    ("own", 0): 2, ("own", 1): 4, ("own", 2): 6,
}
DEFAULT_AUG_COPIES = 1
MAX_FILES = {"speech": 5000}
OWN_WEIGHT = 2.0


# --------------------------------------------------------------------------
# Features (must match the Dart app)
# --------------------------------------------------------------------------
class Features:
    def __init__(self, filterbank_path):
        fb = np.fromfile(filterbank_path, dtype=np.float32)
        if fb.size != N_MELS * (N_FFT // 2 + 1):
            raise SystemExit(f"Unexpected filterbank size {fb.size} in {filterbank_path}")
        self.fb = fb.reshape(N_MELS, N_FFT // 2 + 1).astype(np.float64)
        n = np.arange(N_FFT)
        self.hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N_FFT - 1))
        padded = WIN + N_FFT
        self.n_frames = 1 + (padded - N_FFT) // HOP
        self.idx = n[None, :] + HOP * np.arange(self.n_frames)[:, None]

    def batch(self, waves):
        """waves: (B, 16000) float -> (B, 128, 32, 1) float32."""
        out = []
        for s in range(0, len(waves), 128):
            w = np.asarray(waves[s:s + 128], dtype=np.float64)
            p = np.pad(w, ((0, 0), (N_FFT // 2, N_FFT // 2)))
            frames = p[:, self.idx] * self.hann
            power = np.abs(np.fft.rfft(frames, axis=2)) ** 2
            mel = np.einsum("bfk,mk->bmf", power, self.fb)
            db = 10.0 * np.log10(np.maximum(mel, 1e-10))
            db -= db.max(axis=(1, 2), keepdims=True)
            db = np.maximum(db, -80.0)
            out.append(((db + 80.0) / 80.0).astype(np.float32)[..., None])
        return np.concatenate(out, axis=0)

    def parity_check(self, ml_dir):
        a, b = ml_dir / "test_audio_window.bin", ml_dir / "expected_log_mel.bin"
        if not (a.exists() and b.exists()):
            print("[parity] test_audio_window.bin / expected_log_mel.bin not found next to "
                  "the filterbank - SKIPPING parity check (upload them to be safe).")
            return
        audio = np.fromfile(a, dtype=np.float32)
        expected = np.fromfile(b, dtype=np.float32).reshape(N_MELS, -1)
        got = self.batch(audio[None, :])[0, :, :, 0] * 80.0 - 80.0
        diff = float(np.abs(got - expected).max())
        print(f"[parity] max |python - app expected| log-mel diff = {diff:.2e} dB")
        if diff > 1e-2:
            raise SystemExit("Feature parity FAILED - training on these features would not "
                             "match the app. Fix before continuing.")


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-18))


def load_audio(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def windows_of(x, hop_s):
    hop = int(hop_s * SR)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    out = []
    for s in range(0, len(x) - WIN + 1, hop):
        w = x[s:s + WIN]
        if rms(w) >= SILENCE_RMS:
            out.append(w.astype(np.float32))
    return out


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
def synth_noise(rng, kind=None):
    kind = kind or rng.choice(["white", "pink", "brown"])
    white = rng.standard_normal(WIN)
    if kind == "white":
        n = white
    elif kind == "pink":
        f = np.fft.rfft(white)
        f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
        n = np.fft.irfft(f, WIN)
    else:
        n = np.cumsum(white)
        n -= n.mean()
    return (n / (rms(n) + 1e-12)).astype(np.float32)


def augment(w, rng, noise_bank):
    from scipy.signal import butter, fftconvolve, sosfilt
    base_rms = rms(w)
    x = w.astype(np.float64)
    if rng.random() < 0.4:  # room reverb
        tau = rng.uniform(0.05, 0.4)
        t = np.arange(8000) / SR
        ir = rng.standard_normal(8000) * np.exp(-t / tau)
        ir[0] = 1.0
        y = fftconvolve(x, ir)[:WIN]
        x = y * (base_rms / (rms(y) + 1e-12))
    if rng.random() < 0.4:  # phone-mic band limiting
        lo, hi = rng.uniform(80, 300), rng.uniform(3500, 7500)
        x = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x)
    x = x * 10 ** (rng.uniform(-12, 6) / 20)  # gain
    if rng.random() < 0.7:  # background noise at random SNR
        if noise_bank and rng.random() < 0.6:
            noise = noise_bank[rng.integers(len(noise_bank))]
        else:
            noise = synth_noise(rng)
        snr = rng.uniform(3, 30)
        x = x + noise * (rms(x) / (10 ** (snr / 20)) / (rms(noise) + 1e-12))
    shift = int(rng.uniform(-0.2, 0.2) * SR)
    if shift:
        x = np.roll(x, shift)
        if shift > 0:
            x[:shift] = 0
        else:
            x[shift:] = 0
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------
def download_zip(url, dest_dir, label):
    dest_dir.mkdir(parents=True, exist_ok=True)
    zpath = dest_dir / (label + ".zip")
    if not zpath.exists():
        print(f"[download] {label} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(zpath, "wb") as f:
            shutil.copyfileobj(resp, f)
    print(f"[unzip] {label}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest_dir / label)
    return dest_dir / label


def find_own(root):
    items = []
    for li, name in enumerate(CLASSES):
        for p in sorted(glob(os.path.join(root, name, "*.wav"))):
            items.append((p, li, "own"))
    return items


def find_ravdess(root, strong_as_danger):
    items = []
    for p in sorted(glob(os.path.join(root, "**", "*.wav"), recursive=True)):
        parts = os.path.basename(p)[:-4].split("-")
        if len(parts) != 7 or parts[0] != "03" or parts[1] != "01":
            continue
        emo, inten = int(parts[2]), int(parts[3])
        if emo in (1, 2, 3, 4):
            label = 0
        elif emo in (5, 6):
            label = 2 if (strong_as_danger and inten == 2) else 1
        else:
            continue
        items.append((p, label, "ravdess"))
    return items


def find_speech(root):
    items = [(p, 0, "speech") for p in sorted(glob(os.path.join(root, "**", "*.wav"), recursive=True))
             if "background" not in p.lower()]
    return items


def find_esc50(root):
    metas = glob(os.path.join(root, "**", "esc50.csv"), recursive=True)
    if not metas:
        return []
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(metas[0])), "audio")
    items = []
    with open(metas[0], newline="") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat in ESC_SKIP:
                continue
            label = 1 if cat in ESC_ELEVATED else 2 if cat in ESC_DANGER else 0
            p = os.path.join(audio_dir, row["filename"])
            if os.path.exists(p):
                items.append((p, label, "esc50"))
    return items


def assign_splits(items, seed):
    """Stratified per (source, label), by FILE so overlapping windows never leak."""
    rng = np.random.default_rng(seed)
    groups = {}
    for it in items:
        groups.setdefault((it[2], it[1]), []).append(it)
    out = []
    for key, g in groups.items():
        g = list(g)
        rng.shuffle(g)
        n = len(g)
        n_val = max(1, round(0.15 * n)) if n >= 3 else 0
        n_test = max(1, round(0.15 * n)) if n >= 3 else 0
        for i, it in enumerate(g):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            out.append((*it, split))
    return out


# --------------------------------------------------------------------------
# Build arrays
# --------------------------------------------------------------------------
def build_arrays(items, feats, seed, max_files=None, log=print):
    rng = np.random.default_rng(seed)
    max_files = max_files or {}
    # cap very large sources deterministically
    by_src = {}
    for it in items:
        by_src.setdefault(it[2], []).append(it)
    capped = []
    for src, lst in by_src.items():
        lst = list(lst)
        rng.shuffle(lst)
        capped += lst[:max_files[src]] if src in max_files else lst
    items = capped

    # noise bank from benign ESC-50 TRAIN files
    noise_bank = []
    for p, label, src, split in items:
        if src == "esc50" and label == 0 and split == "train" and len(noise_bank) < 300:
            try:
                ws = windows_of(load_audio(p), 1.0)
                if ws:
                    noise_bank.append(ws[rng.integers(len(ws))])
            except Exception:
                pass
    log(f"[noise] bank of {len(noise_bank)} real noise windows")

    data = {s: {"X": [], "y": [], "src": [], "aug": []} for s in ("train", "val", "test")}
    pending = {s: [] for s in data}

    def flush(split):
        buf = pending[split]
        if not buf:
            return
        X = feats.batch([b[0] for b in buf]).astype(np.float16)
        data[split]["X"].append(X)
        data[split]["y"] += [b[1] for b in buf]
        data[split]["src"] += [b[2] for b in buf]
        data[split]["aug"] += [b[3] for b in buf]
        pending[split] = []

    def push(split, w, label, src, aug):
        pending[split].append((w, label, src, aug))
        if len(pending[split]) >= 256:
            flush(split)

    for n_done, (p, label, src, split) in enumerate(items):
        try:
            x = load_audio(p)
        except Exception as e:
            log(f"[skip] {p}: {e}")
            continue
        hop = 0.25 if (label > 0 and src == "own") else 0.5
        for w in windows_of(x, hop):
            push(split, w, label, src, False)
            if split == "train":
                for _ in range(AUG_COPIES.get((src, label), DEFAULT_AUG_COPIES)):
                    push(split, augment(w, rng, noise_bank), label, src, True)
            elif split == "test":
                # one noisy/reverberant copy of each test window for a robustness metric
                push(split, augment(w, rng, noise_bank), label, src, True)
        if (n_done + 1) % 500 == 0:
            log(f"  processed {n_done + 1}/{len(items)} files")

    # synthetic faint-noise windows -> normal
    for i in range(1500):
        w = synth_noise(rng) * float(np.exp(rng.uniform(np.log(SILENCE_RMS), np.log(0.05))))
        split = "train" if i % 100 < 70 else "val" if i % 100 < 85 else "test"
        push(split, w.astype(np.float32), 0, "synthetic", False)
    for s in data:
        flush(s)

    out = {}
    for s, d in data.items():
        out[s] = {
            "X": np.concatenate(d["X"]) if d["X"] else np.zeros((0, N_MELS, feats.n_frames, 1), np.float16),
            "y": np.array(d["y"], dtype=np.int64),
            "src": np.array(d["src"]),
            "aug": np.array(d["aug"], dtype=bool),
        }
    return out


# --------------------------------------------------------------------------
# Model, metrics, export
# --------------------------------------------------------------------------
def build_model(n_frames):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input((N_MELS, n_frames, 1))
    x = inp
    for f in (16, 32, 64):
        x = L.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.MaxPool2D(2)(x)
    x = L.Flatten()(x)
    x = L.Dropout(0.4)(x)
    out = L.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def app_tier(p):
    """Same rule as _instantaneousTier in voice_analysis_service.dart."""
    pn, pe, ph = p[:, 0], p[:, 1], p[:, 2]
    tier = np.zeros(len(p), dtype=int)
    tier[(pe > 0.5) & (pe > pn)] = 1
    tier[ph > 0.6] = 2
    return tier


def metrics_block(name, probs, y, src, mask=None):
    if mask is not None:
        probs, y, src = probs[mask], y[mask], src[mask]
    lines = [f"--- {name} (n={len(y)}) ---"]
    if len(y) == 0:
        return lines + ["(empty)"]
    pred = probs.argmax(1)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    lines.append("confusion (rows=true, cols=pred) [normal, elevated, danger]:")
    for i, c in enumerate(CLASSES):
        lines.append(f"  {c:9s} {cm[i].tolist()}")
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        rec = tp / max(cm[i].sum(), 1)
        prec = tp / max(cm[:, i].sum(), 1)
        lines.append(f"  {c:9s} recall {rec:.3f}  precision {prec:.3f}")
    tier = app_tier(probs)
    nm = y == 0
    if nm.any():
        lines.append(f"APP RULE on Normal windows: any-alert (tier>=1) = {np.mean(tier[nm] >= 1):.3%}, "
                     f"HighDanger (tier==2) = {np.mean(tier[nm] == 2):.3%}   (NFR-S1 target <= 5%)")
    dm = y == 2
    if dm.any():
        lines.append(f"APP RULE on Danger windows: HighDanger detected = {np.mean(tier[dm] == 2):.3%}, "
                     f"at least Elevated = {np.mean(tier[dm] >= 1):.3%}")
    for s in sorted(set(src.tolist())):
        m = src == s
        lines.append(f"  source {s:10s} n={m.sum():6d} acc={np.mean(pred[m] == y[m]):.3f}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--own", default="voice_data")
    ap.add_argument("--ravdess", default=None, help="folder containing RAVDESS Actor_XX/*.wav (auto-download if omitted)")
    ap.add_argument("--workdir", default="voice_work")
    ap.add_argument("--out", default="voice_out")
    ap.add_argument("--filterbank", default=None,
                    help="path to mel_filterbank.bin (default: next to this script, the current "
                         "folder, or <project>/assets/ml)")
    ap.add_argument("--sources", default="own,ravdess,speech,esc50,synthetic")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--ravdess-strong-as-danger", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.workdir)
    sources = set(args.sources.split(","))

    if args.selftest:
        work = Path(tempfile.mkdtemp(prefix="vt_selftest_"))
        make_selftest_data(work)
        args.own = str(work / "own")
        args.ravdess = str(work / "ravdess")
        args.no_download = True
        args.epochs = min(args.epochs, 12)
        MAX_FILES.clear()
        print(f"[selftest] synthetic datasets in {work}")

    if args.filterbank is None:
        candidates = [here / "mel_filterbank.bin", Path.cwd() / "mel_filterbank.bin"]
        if len(here.parents) > 1:
            candidates.append(here.parents[1] / "assets" / "ml" / "mel_filterbank.bin")
        found = [c for c in candidates if c.exists()]
        if not found:
            raise SystemExit("mel_filterbank.bin not found. Upload it next to this script or "
                             "pass --filterbank PATH.")
        args.filterbank = str(found[0])
    feats = Features(args.filterbank)
    feats.parity_check(Path(args.filterbank).parent)

    items = []
    if "own" in sources:
        own = find_own(args.own)
        print(f"[own] {len(own)} clips")
        items += own
    if "ravdess" in sources:
        root = args.ravdess
        if root is None and not args.no_download:
            try:
                root = str(download_zip(RAVDESS_URL, work, "ravdess"))
            except Exception as e:
                print(f"[ravdess] download failed ({e}); pass --ravdess DIR to use a local copy")
        if root:
            r = find_ravdess(root, args.ravdess_strong_as_danger)
            print(f"[ravdess] {len(r)} clips")
            items += r
    if "speech" in sources:
        root = work / "speech_commands" if args.selftest else None
        if root is None and not args.no_download:
            try:
                root = download_zip(SPEECH_URL, work, "speech")
            except Exception as e:
                print(f"[speech] download failed ({e})")
        if root:
            s = find_speech(str(root))
            print(f"[speech] {len(s)} clips")
            items += s
    if "esc50" in sources:
        root = work / "esc50" if args.selftest else None
        if root is None and not args.no_download:
            try:
                root = download_zip(ESC50_URL, work, "esc50")
            except Exception as e:
                print(f"[esc50] download failed ({e})")
        if root:
            e = find_esc50(str(root))
            print(f"[esc50] {len(e)} clips")
            items += e

    if not items and "synthetic" not in sources:
        raise SystemExit("No data found.")
    items = assign_splits(items, args.seed)
    print("[build] extracting features + augmenting (this is the slow part)...")
    d = build_arrays(items, feats, args.seed, MAX_FILES)

    report = ["NeuralSafe voice model training report", "=" * 40]
    report.append("windows per split/class (train includes augmented copies):")
    for s in ("train", "val", "test"):
        counts = [int((d[s]["y"] == i).sum()) for i in range(3)]
        report.append(f"  {s:5s} {dict(zip(CLASSES, counts))}")
    own_counts = {c: int(((d['train']['src'] == 'own') & (d['train']['y'] == i) & ~d['train']['aug']).sum())
                  for i, c in enumerate(CLASSES)}
    report.append(f"own-data clean train windows: {own_counts}")
    warn = [c for c in ("elevated", "danger") if own_counts[c] < 100]
    if warn and not args.selftest:
        report.append(f"WARNING: very little OWN data for {warn}. Real-world recall for those "
                      f"classes will be unreliable; record more clips in the app.")
    print("\n".join(report))
    for i, c in enumerate(CLASSES):
        if (d["train"]["y"] == i).sum() == 0:
            raise SystemExit(f"No training data for class '{c}'. Add data and re-run.")

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(feats.n_frames)
    model.summary(print_fn=lambda s: None)

    ytr = d["train"]["y"]
    counts = np.bincount(ytr, minlength=3).astype(float)
    cw = {i: float(len(ytr) / (3 * max(counts[i], 1))) for i in range(3)}
    sw = np.array([cw[y] for y in ytr], dtype=np.float32)
    sw[d["train"]["src"] == "own"] *= OWN_WEIGHT
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    oh = lambda y: tf.keras.utils.to_categorical(y, 3)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    model.fit(
        d["train"]["X"].astype(np.float32), oh(ytr), sample_weight=sw,
        validation_data=(d["val"]["X"].astype(np.float32), oh(d["val"]["y"])),
        epochs=args.epochs, batch_size=64, callbacks=cb, verbose=2,
    )

    report += ["", f"class weights: {cw}", f"own-data sample weight multiplier: {OWN_WEIGHT}", ""]
    for split in ("val", "test"):
        probs = model.predict(d[split]["X"].astype(np.float32), batch_size=256, verbose=0)
        y, src, aug = d[split]["y"], d[split]["src"], d[split]["aug"]
        report += metrics_block(f"{split} - clean windows", probs, y, src, ~aug)
        if aug.any():
            report += metrics_block(f"{split} - noisy/reverb augmented windows", probs, y, src, aug)
        own = (src == "own") & ~aug
        if own.any():
            report += metrics_block(f"{split} - OWN phone recordings only", probs, y, src, own)
        report.append("")

    # export + verify
    tflite_path = out / "neuralsafe_voice_model_v3.tflite"
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        blob = conv.convert()
    except Exception as e:
        print(f"[export] from_keras_model failed ({e}); trying SavedModel route")
        sm = out / "saved_model"
        model.export(str(sm))
        blob = tf.lite.TFLiteConverter.from_saved_model(str(sm)).convert()
    tflite_path.write_bytes(blob)
    print(f"[export] wrote {tflite_path} ({len(blob)} bytes)")
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter as TFLInterpreter
        except Exception:
            TFLInterpreter = tf.lite.Interpreter
        interp = TFLInterpreter(model_content=blob)
        interp.allocate_tensors()
        inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
        Xt = d["test"]["X"][:64].astype(np.float32)
        ref = model.predict(Xt, verbose=0)
        worst = 0.0
        for i in range(len(Xt)):
            interp.set_tensor(inp["index"], Xt[i:i + 1])
            interp.invoke()
            worst = max(worst, float(np.abs(interp.get_tensor(outp["index"]) - ref[i:i + 1]).max()))
        report.append(f"TFLite export: {tflite_path.name}, {len(blob)} bytes, input "
                      f"{inp['shape'].tolist()}, output {outp['shape'].tolist()}, "
                      f"max |tflite - keras| = {worst:.2e}")
    except Exception as e:  # never lose the trained model over a verification problem
        report.append(f"TFLite export: {tflite_path.name}, {len(blob)} bytes "
                      f"(verification skipped: {e})")
    # sanity: the failure that motivated this retrain
    rng = np.random.default_rng(1)
    noise_x = feats.batch([synth_noise(rng, "white") * a for a in (0.006, 0.02, 0.1, 0.3)])
    probs = model.predict(noise_x, verbose=0)
    report.append("white noise at RMS 0.006/0.02/0.1/0.3 -> probs [N,E,D]:")
    report += [f"  {np.round(p, 3).tolist()}" for p in probs]
    report.append("(v2 gave [0, 0, 1] for every one of these.)")

    (out / "voice_model_report.txt").write_text("\n".join(report))
    (out / "voice_training_meta.json").write_text(json.dumps(
        {"classes": CLASSES, "silence_rms": SILENCE_RMS, "n_frames": feats.n_frames,
         "sources": sorted(sources), "seed": args.seed}, indent=2))
    print("\n".join(report))
    print(f"\nDone. Copy {tflite_path} to assets/ml/ and set kVoiceModelAsset in voice_analysis_service.dart.")


# --------------------------------------------------------------------------
# Self-test fixtures
# --------------------------------------------------------------------------
def make_selftest_data(root):
    import soundfile as sf
    rng = np.random.default_rng(0)

    def tone_burst(sr, secs, f0, am_hz, amp, noise_amp=0.01, sweep=0.0):
        t = np.arange(int(sr * secs)) / sr
        f = f0 + sweep * t
        x = np.sin(2 * np.pi * np.cumsum(f) / sr) * (0.5 + 0.5 * np.sin(2 * np.pi * am_hz * t))
        x += 0.3 * np.sin(2 * np.pi * 2 * np.cumsum(f) / sr)
        return (amp * x + noise_amp * rng.standard_normal(len(t))).astype(np.float32)

    def write(path, x, sr):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), x, sr)

    for i in range(14):
        write(root / "own/normal" / f"n{i}.wav", tone_burst(16000, 6, 120 + 5 * i, 3, 0.08), 16000)
        write(root / "own/elevated" / f"e{i}.wav", tone_burst(16000, 6, 280 + 8 * i, 5, 0.25), 16000)
        write(root / "own/danger" / f"d{i}.wav", tone_burst(16000, 6, 900 + 20 * i, 7, 0.6, sweep=150), 16000)
    for i in range(16):
        emo = [1, 2, 3, 4, 5, 6, 7, 8][i % 8]
        write(root / "ravdess/Actor_01" / f"03-01-0{emo}-0{1 + i % 2}-01-01-{i + 1:02d}.wav",
              tone_burst(48000, 3.5, 150 + 20 * emo, 4, 0.05 + 0.04 * emo), 48000)
    for i in range(30):
        write(root / "speech_commands/yes" / f"s{i}.wav", tone_burst(16000, 1, 130 + 3 * i, 4, 0.07), 16000)
    meta = root / "esc50/ESC-50-master/meta"
    meta.mkdir(parents=True, exist_ok=True)
    with open(meta / "esc50.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "fold", "target", "category", "esc10", "src_file", "take"])
        cats = ["rain", "crying_baby", "glass_breaking", "dog", "fireworks", "keyboard_typing"]
        for i in range(36):
            cat = cats[i % len(cats)]
            name = f"1-{i}-A-{i % 50}.wav"
            n = (rng.standard_normal(44100 * 5) * 0.05).astype(np.float32)
            if cat == "glass_breaking":
                n[44100:44500] += 0.7 * rng.standard_normal(400).astype(np.float32)
            write(root / "esc50/ESC-50-master/audio" / name, n, 44100)
            w.writerow([name, 1, i % 50, cat, "False", "x", "A"])


if __name__ == "__main__":
    main()
