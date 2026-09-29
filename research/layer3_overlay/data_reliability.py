"""Portable reliability-aware loader for the archived four-speaker data.

The archived loader imports ``tgt`` only for an unused TextGrid helper and has
an absolute workstation path in ``__getitem__``. This implementation preserves
its train split and collation semantics without either dependency.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

import params


class ReliabilityVCTKDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, centroid_path, seed=params.seed):
        self.mel_dir = os.path.join(data_dir, "mels")
        self.emb_dir = os.path.join(data_dir, "embeds")
        self.unseen_sentences = {
            "002", "003", "004", "005", "006", "007", "009", "010", "011", "012"
        }
        self.speakers = sorted(os.listdir(self.mel_dir))
        ordering_rng = random.Random(seed)
        ordering_rng.shuffle(self.speakers)
        self.train_info = []
        for speaker in self.speakers:
            mel_files = os.listdir(os.path.join(self.mel_dir, speaker))
            mel_files = [
                name for name in mel_files
                if name.endswith("_mel.npy") and name.split("_")[1] not in self.unseen_sentences
            ]
            self.train_info.extend((name[:-8], speaker) for name in mel_files)
        ordering_rng.shuffle(self.train_info)
        stored = np.load(centroid_path)
        self.centroids = {
            speaker: torch.from_numpy(stored[speaker]).float()
            for speaker in stored.files
        }
        missing = sorted(set(self.speakers) - set(self.centroids))
        if missing:
            raise ValueError(f"centroids missing speakers: {missing}")
        print(f"Total number of training wavs is {len(self.train_info)}.")
        print(f"Total number of training speakers is {len(self.speakers)}.")

    def get_vc_data(self, audio_info):
        audio_id, speaker = audio_info
        mel = np.load(os.path.join(self.mel_dir, speaker, audio_id + "_mel.npy"))
        embed = np.load(os.path.join(self.emb_dir, speaker, audio_id + "_embed.npy"))
        return torch.from_numpy(mel).float(), torch.from_numpy(embed).float()

    def __getitem__(self, index):
        src_audio, src_spk = self.train_info[index]
        src_mels, src_embed = self.get_vc_data((src_audio, src_spk))
        tgt_spk = random.choice([speaker for speaker in self.speakers if speaker != src_spk])
        mel_files = [
            name for name in os.listdir(os.path.join(self.mel_dir, tgt_spk))
            if (name.endswith("_mel.npy") and
                name.split("_")[1] not in self.unseen_sentences)
        ]
        tgt_audio = random.choice(mel_files)[:-8]
        tgt_mels, tgt_embed = self.get_vc_data((tgt_audio, tgt_spk))
        centroid = self.centroids[tgt_spk]
        similarity = torch.nn.functional.cosine_similarity(
            tgt_embed.reshape(1, -1), centroid.reshape(1, -1)
        ).item()
        reliability = max(0.0, min(1.0, 0.5 * (similarity + 1.0)))
        return {
            "mel": src_mels,
            "c": src_embed,
            "tgt_mel": tgt_mels,
            "tgt_c": tgt_embed,
            "tgt_spk": tgt_spk,
            "tgt_audio": tgt_audio,
            "tgt_ref_score": reliability,
        }

    def __len__(self):
        return len(self.train_info)


class ReliabilityBatchCollate:
    def __call__(self, batch):
        size = len(batch)
        frames = params.train_frames
        mels1 = torch.zeros((size, params.n_mels, frames), dtype=torch.float32)
        mels2 = torch.zeros_like(mels1)
        source_max_starts = [max(item["mel"].shape[-1] - frames, 0) for item in batch]
        starts1 = [random.randrange(limit) if limit > 0 else 0 for limit in source_max_starts]
        starts2 = [random.randrange(limit) if limit > 0 else 0 for limit in source_max_starts]
        source_lengths = []
        for item_index, item in enumerate(batch):
            mel = item["mel"]
            length = min(mel.shape[-1], frames)
            mels1[item_index, :, :length] = mel[:, starts1[item_index]:starts1[item_index] + length]
            mels2[item_index, :, :length] = mel[:, starts2[item_index]:starts2[item_index] + length]
            source_lengths.append(length)

        target_mels = torch.zeros_like(mels1)
        target_max_starts = [max(item["tgt_mel"].shape[-1] - frames, 0) for item in batch]
        target_starts = [random.randrange(limit) if limit > 0 else 0 for limit in target_max_starts]
        target_lengths = []
        for item_index, item in enumerate(batch):
            mel = item["tgt_mel"]
            length = min(mel.shape[-1], frames)
            target_mels[item_index, :, :length] = mel[
                :, target_starts[item_index]:target_starts[item_index] + length
            ]
            target_lengths.append(length)

        return {
            "mel1": mels1,
            "mel2": mels2,
            "mel_lengths": torch.tensor(source_lengths, dtype=torch.long),
            "c": torch.stack([item["c"] for item in batch], 0),
            "mel_tgt": target_mels,
            "tgt_mel_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "tgt_c": torch.stack([item["tgt_c"] for item in batch], 0),
            "tgt_ref_score": torch.tensor(
                [item["tgt_ref_score"] for item in batch], dtype=torch.float32
            ),
            "tgt_spk": [item["tgt_spk"] for item in batch],
            "tgt_audio": [item["tgt_audio"] for item in batch],
        }
