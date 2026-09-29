# Layer 7: four-to-eight-speaker expansion

Layer 7 is data-blocked, not code-blocked. The private project archive currently
contains only `p236`, `p239`, `p259`, and `p263`. No second VCTK archive or
preprocessed speaker directory was found at the top level of connected Drive.

To resume without changing Layers 1--6, provide one dataset root with exactly
eight speaker IDs and matching subdirectories:

```text
DATASET/
  wavs/<speaker>/*_mic1.wav
  mels/<speaker>/*_mel.npy
  embeds/<speaker>/*_embed.npy
  txt/<speaker>/*.txt
```

Run the asset gate before training:

```bash
python research/layer7_expansion/audit_eight_speaker_assets.py \
  --data-dir /path/to/DATASET \
  --output data/processed/layer7_asset_audit.json
```

The gate requires exactly eight speakers, at least 20 continuation-training
utterances per speaker, matching Mel/embedding keys, and all ten fixed evaluation
sentence IDs. Sentence IDs `002`, `003`, `004`, `005`, `006`, `007`, `009`,
`010`, `011`, and `012` remain excluded from continuation sources, target
references, and training centroids. Speaker selection and this split must be
frozen before running the existing Layers 3--6 matrix.

The model architecture does not require an eight-class output head; it conditions
on continuous speaker embeddings. After the asset gate passes, rebuild centroids,
repeat the production-shape gradient smoke check, train the preregistered primary
and ablation variants, and evaluate all 56 directed pairs with independent
WavLM/Whisper/acoustic metrics. Historical epoch-50 exposure remains a caveat
unless the eight-speaker study is trained from scratch.
