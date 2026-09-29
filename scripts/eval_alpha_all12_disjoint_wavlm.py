from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa

from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from scipy.stats import ttest_rel, wilcoxon

MODEL_ID = "microsoft/wavlm-base-plus-sv"

ROOT = Path("VCTK_2F2M/wavs")
ALL12 = Path("converted/all12")
MID = Path("converted/alpha_screen")

SPEAKERS = ["p236", "p239", "p259", "p263"]
ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]

device = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)

print("device:", device)
print("loading:", MODEL_ID)

processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioXVector.from_pretrained(MODEL_ID).to(device)
model.eval()


def load16k(path):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    if sr != 16000:
        wav = librosa.resample(
            wav,
            orig_sr=sr,
            target_sr=16000
        )

    return wav


@torch.no_grad()
def embed(path):
    wav = load16k(path)

    inputs = processor(
        wav,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    e = model(**inputs).embeddings
    e = torch.nn.functional.normalize(e, dim=-1)

    return e[0].cpu().numpy()


def cosine(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    )


def centroid(spk, n=20):
    files = sorted((ROOT / spk).glob("*.wav"))[40:60]

    embs = np.array([
        embed(f) for f in files
    ])

    c = embs.mean(axis=0)
    c /= np.linalg.norm(c) + 1e-8

    return c


print("\nBuilding independent WavLM centroids...")

centroids = {}

for spk in SPEAKERS:
    centroids[spk] = centroid(spk)
    print(spk, "done")


def get_path(src, tgt, s, r, alpha):

    base = f"{src}_to_{tgt}_s{s}_r{r}"

    if alpha == 0.0:
        return ALL12 / "baseline" / f"{base}.wav"

    if alpha == 1.0:
        return ALL12 / "centroid" / f"{base}.wav"

    a = int(alpha * 100)

    return MID / f"{base}_a{a}.wav"


def delta(path, src, tgt):
    e = embed(path)

    ts = cosine(e, centroids[tgt])
    ss = cosine(e, centroids[src])

    return ts - ss


# results[(src,tgt)][alpha] -> shape [5 sources, 3 refs]
results = {}

print("\nEvaluating all alpha curves...")

for src in SPEAKERS:
    for tgt in SPEAKERS:

        if src == tgt:
            continue

        key = (src, tgt)
        results[key] = {}

        print(src, "->", tgt)

        for alpha in ALPHAS:

            arr = np.zeros((5, 3))

            for s in range(1, 6):
                for r in range(1, 4):

                    path = get_path(
                        src, tgt, s, r, alpha
                    )

                    if not path.exists():
                        raise FileNotFoundError(path)

                    arr[s-1, r-1] = delta(
                        path, src, tgt
                    )

            results[key][alpha] = arr


print("\n=== DIRECTION-WISE CURVES ===")

direction_best = {}

for key in results:

    src, tgt = key
    print(f"\n{src}->{tgt}")

    means = {}

    for a in ALPHAS:

        arr = results[key][a]
        m = arr.mean()
        sd = arr.std(ddof=1)

        ref_std = np.mean(
            arr.std(axis=1, ddof=1)
        )

        means[a] = m

        print(
            f"alpha={a:.2f}  "
            f"DELTA={m:.4f}Â±{sd:.4f}  "
            f"ref_std={ref_std:.4f}"
        )

    best = max(means, key=means.get)
    direction_best[key] = best

    print(
        f"BEST={best:.2f}  "
        f"gain_vs_0={means[best]-means[0.0]:+.4f}"
    )


print("\n=== BEST ALPHA BY DIRECTION ===")

for key in direction_best:
    print(
        f"{key[0]}->{key[1]}: "
        f"{direction_best[key]:.2f}"
    )


print("\n=== GLOBAL FIXED ALPHA ===")

global_means = {}

for a in ALPHAS:

    vals = []

    for key in results:
        vals.extend(
            results[key][a].reshape(-1)
        )

    vals = np.array(vals)

    global_means[a] = vals.mean()

    print(
        f"alpha={a:.2f}: "
        f"{vals.mean():.4f}Â±{vals.std(ddof=1):.4f}"
    )

best_global = max(
    global_means,
    key=global_means.get
)

print(
    f"BEST GLOBAL alpha={best_global:.2f}"
)


print("\n=== DIRECTION-ORACLE ===")

baseline_units = []
oracle_units = []

for key in results:

    best = direction_best[key]

    for s in range(5):

        b = results[key][0.0][s].mean()
        o = results[key][best][s].mean()

        baseline_units.append(b)
        oracle_units.append(o)

baseline_units = np.array(baseline_units)
oracle_units = np.array(oracle_units)

print(
    f"baseline N=60: "
    f"{baseline_units.mean():.4f}"
)

print(
    f"direction oracle: "
    f"{oracle_units.mean():.4f}"
)

print(
    f"oracle gain: "
    f"{(oracle_units-baseline_units).mean():+.4f}"
)


print("\n=== LEAVE-ONE-SOURCE-OUT DIRECTION CALIBRATION ===")

# å¯¹æ¯ä¸ªæ¹åï¼
# ç¨4ä¸ªsourceéæ©alphaï¼
# å¨å©ä¸1ä¸ªsourceä¸æµè¯
cv_base = []
cv_adapt = []
chosen = []

for key in results:

    src, tgt = key

    for held_out in range(5):

        train_ids = [
            i for i in range(5)
            if i != held_out
        ]

        train_scores = {}

        for a in ALPHAS:

            vals = []

            for i in train_ids:
                vals.append(
                    results[key][a][i].mean()
                )

            train_scores[a] = np.mean(vals)

        selected = max(
            train_scores,
            key=train_scores.get
        )

        chosen.append(selected)

        b = results[key][0.0][held_out].mean()
        a = results[key][selected][held_out].mean()

        cv_base.append(b)
        cv_adapt.append(a)

cv_base = np.array(cv_base)
cv_adapt = np.array(cv_adapt)

cv_imp = cv_adapt - cv_base

print(
    f"Baseline : "
    f"{cv_base.mean():.4f}Â±{cv_base.std(ddof=1):.4f}"
)

print(
    f"Adaptive : "
    f"{cv_adapt.mean():.4f}Â±{cv_adapt.std(ddof=1):.4f}"
)

print(
    f"Gain     : "
    f"{cv_imp.mean():+.4f}Â±{cv_imp.std(ddof=1):.4f}"
)

print(
    f"Wins     : "
    f"{np.sum(cv_imp > 0)}/60 "
    f"({np.mean(cv_imp > 0)*100:.1f}%)"
)

print(
    f"Median   : "
    f"{np.median(cv_imp):+.4f}"
)

print("\nSelected alpha counts:")

for a in ALPHAS:
    print(
        f"{a:.2f}: {chosen.count(a)}"
    )


print("\n=== CV STATISTICS ===")

t = ttest_rel(
    cv_adapt,
    cv_base
)

w = wilcoxon(
    cv_adapt,
    cv_base
)

print(
    f"paired t-test: "
    f"t={t.statistic:.4f}, "
    f"p={t.pvalue:.6g}"
)

print(
    f"Wilcoxon: "
    f"W={w.statistic:.4f}, "
    f"p={w.pvalue:.6g}"
)


print("\n=== PER-DIRECTION CV ===")

for key in results:

    src, tgt = key

    bvals = []
    avals = []

    for held_out in range(5):

        train_ids = [
            i for i in range(5)
            if i != held_out
        ]

        train_scores = {}

        for a in ALPHAS:

            train_scores[a] = np.mean([
                results[key][a][i].mean()
                for i in train_ids
            ])

        selected = max(
            train_scores,
            key=train_scores.get
        )

        bvals.append(
            results[key][0.0][held_out].mean()
        )

        avals.append(
            results[key][selected][held_out].mean()
        )

    bvals = np.array(bvals)
    avals = np.array(avals)

    print(
        f"{src}->{tgt}: "
        f"{bvals.mean():.4f} -> "
        f"{avals.mean():.4f}  "
        f"gain={(avals-bvals).mean():+.4f}  "
        f"sources improved="
        f"{np.sum(avalvals > bvals) if False else np.sum(avals > bvals)}/5"
    )

np.savez_compressed("all12_disjoint_wavlm_raw.npz", **{f"{s}_to_{tg}_a{int(a*100):03d}": arr for (s,tg), by_a in results.items() for a, arr in by_a.items()})
print("saved: all12_disjoint_wavlm_raw.npz")
