"""
RLDS Dataset Builder for Bimanual UR3 Manipulation Task.

Reads raw recorded HDF5 files directly — no preprocessing step needed.
Decodes JPEG-compressed image buffers using compress_len, resizes to
224x224, and writes a compact TFDS dataset in one pass.

Raw HDF5 layout (input):
  action:                         (T, 14)   float32
  base_action:                    (T, 2)    float32
  compress_len:                   (3, T)    float32  row 0=cam_high
                                                     row 1=cam_left_wrist
                                                     row 2=cam_right_wrist
  observations/
    qpos:                         (T, 14)   float32
    qvel:                         (T, 14)   float32  (not loaded)
    effort:                       (T, 14)   float32  (not loaded)
    images/
      cam_high:                   (T, ~N)   uint8    JPEG buffer
      cam_left_wrist:             (T, ~N)   uint8    JPEG buffer
      cam_right_wrist:            (T, ~N)   uint8    JPEG buffer

Usage
-----
1.  Place this file inside a folder named  ur3_bimanual_dataset/
2.  Set HDF5_DIR below (or export env-var HDF5_DATASET_DIR) to the
    directory containing your raw .hdf5 files.
3.  Run:
        cd ur3_bimanual_dataset
        tfds build --overwrite
    This produces a TFDS dataset in ~/tensorflow_datasets/ur3_bimanual_dataset/

OpenVLA-OFT training command (example):
    torchrun --standalone --nnodes=1 --nproc-per-node=4 \
      vla-scripts/finetune.py \
      --vla_path "openvla/openvla-7b" \
      --data_root_dir /path/to/tensorflow_datasets \
      --dataset_name ur3_bimanual_dataset \
      --run_root_dir ./runs \
      --adapter_tmp_dir ./adapter-tmp \
      --lora_rank 32 \
      --batch_size 16 \
      --learning_rate 2e-4 \
      --image_aug True \
      --use_l1_regression True \
      --wandb_project ur3_openvla_oft
"""

import io
import os
import glob
import random
from pathlib import Path
from typing import Iterator, Tuple, Any, Dict

import h5py
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

# ── Configuration ─────────────────────────────────────────────────────────────

# Directory containing your raw .hdf5 / .h5 files (searched recursively).
# Override with env-var:  export HDF5_DATASET_DIR=/your/path
HDF5_DIR = os.environ.get("HDF5_DATASET_DIR", "/path/to/your/hdf5/files")

# Language instruction attached to every step.
# Edit this to match your task.
LANGUAGE_INSTRUCTION = "perform bimanual manipulation task"

# Camera names and their corresponding compress_len row index
CAM_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CAM_ROW   = {"cam_high": 0, "cam_left_wrist": 1, "cam_right_wrist": 2}

# Output image size (OpenVLA-OFT expects 224×224)
IMAGE_SIZE = (224, 224)

# Fraction of episodes to hold out as validation (0.1 = 10%)
PERCENT_VAL = 0.1

# Random seed for reproducible train/val split
SPLIT_SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(raw_bytes: np.ndarray, length: int) -> np.ndarray:
    """
    Decode a single JPEG-compressed frame stored as a padded uint8 buffer.

    Args:
        raw_bytes:  1-D uint8 array (padded to fixed buffer size).
        length:     number of valid JPEG bytes (from compress_len).

    Returns:
        (224, 224, 3) uint8 numpy array.
    """
    jpeg = bytes(raw_bytes[:int(length)])
    img  = Image.open(io.BytesIO(jpeg)).convert("RGB")
    img  = img.resize(IMAGE_SIZE, Image.BICUBIC)
    return np.array(img, dtype=np.uint8)


def _load_episode(hdf5_path: str) -> Dict[str, Any]:
    """
    Load and decode all timesteps from a single raw HDF5 file.

    Returns a dict with decoded image arrays and action/state arrays.
    """
    with h5py.File(hdf5_path, "r") as f:
        T           = f["action"].shape[0]
        action      = f["action"][()]         # (T, 14)
        qpos        = f["observations/qpos"][()]  # (T, 14)
        compress_len = f["compress_len"][()]  # (3, T)

        # Load raw JPEG buffers for all cameras
        raw = {
            cam: f[f"observations/images/{cam}"][()]
            for cam in CAM_NAMES
        }

    # Decode every frame for every camera
    images = {}
    for cam in CAM_NAMES:
        row    = CAM_ROW[cam]
        frames = [
            _decode_image(raw[cam][t], compress_len[row, t])
            for t in range(T)
        ]
        images[cam] = np.stack(frames)  # (T, 224, 224, 3)

    return {"T": T, "action": action, "qpos": qpos, **images}


# ── Dataset Builder ───────────────────────────────────────────────────────────

class Ur3BimanualDataset(tfds.core.GeneratorBasedBuilder):
    """
    TFDS / RLDS builder for bimanual UR3 manipulation episodes.

    Reads raw ALOHA-format HDF5 files (JPEG-compressed images + compress_len)
    and converts them directly to a compact RLDS dataset compatible with
    OpenVLA-OFT. No preprocessing step required.
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release — bimanual UR3, 3 cameras, 14-DOF."
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        # Overhead camera → OpenVLA-OFT primary image key
                        "image_primary": tfds.features.Image(
                            shape=(*IMAGE_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Overhead (cam_high) RGB frame, 224x224.",
                        ),
                        "image_left_wrist": tfds.features.Image(
                            shape=(*IMAGE_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Left wrist camera RGB frame, 224x224.",
                        ),
                        "image_right_wrist": tfds.features.Image(
                            shape=(*IMAGE_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="Right wrist camera RGB frame, 224x224.",
                        ),
                        # qpos fed to OpenVLA-OFT's proprio projector
                        # when --use_proprio True is set.
                        "state": tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc="Joint positions (qpos), 7 DOF per arm.",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc="Target joint positions for both arms.",
                    ),
                    "is_first":    tf.bool,
                    "is_last":     tf.bool,
                    "is_terminal": tf.bool,
                    "reward": tfds.features.Scalar(
                        dtype=np.float32,
                        doc="Sparse reward: 0 everywhere, +1 at final step.",
                    ),
                    "language_instruction": tfds.features.Text(
                        doc="Natural-language task description.",
                    ),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(
                        doc="Source HDF5 file path.",
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """
        Find all HDF5 files under HDF5_DIR and split into train / val.
        Split is done at the episode level with a fixed random seed for
        reproducibility.
        """
        all_files = sorted(
            glob.glob(os.path.join(HDF5_DIR, "**/*.hdf5"), recursive=True) +
            glob.glob(os.path.join(HDF5_DIR, "**/*.h5"),   recursive=True)
        )

        if not all_files:
            raise FileNotFoundError(
                f"No .hdf5 / .h5 files found under '{HDF5_DIR}'.\n"
                "Set the HDF5_DATASET_DIR environment variable or edit HDF5_DIR."
            )

        # Shuffle with fixed seed for reproducible splits
        rng = random.Random(SPLIT_SEED)
        shuffled = all_files[:]
        rng.shuffle(shuffled)

        n_val   = max(1, int(len(shuffled) * PERCENT_VAL))
        n_train = len(shuffled) - n_val

        train_files = shuffled[:n_train]
        val_files   = shuffled[n_train:]

        print(f"Found {len(all_files)} episodes → "
              f"{n_train} train / {n_val} val  (seed={SPLIT_SEED})")

        return {
            "train": self._generate_examples(train_files),
            "val":   self._generate_examples(val_files),
        }

    def _generate_examples(self, file_paths) -> Iterator[Tuple[str, Any]]:
        """Yield one RLDS episode per HDF5 file."""
        for ep_idx, hdf5_path in enumerate(file_paths):
            try:
                ep = _load_episode(hdf5_path)
            except Exception as exc:
                print(f"[WARN] Skipping {hdf5_path}: {exc}")
                continue

            T     = ep["T"]
            steps = []
            for t in range(T):
                steps.append({
                    "observation": {
                        "image_primary":     ep["cam_high"][t],
                        "image_left_wrist":  ep["cam_left_wrist"][t],
                        "image_right_wrist": ep["cam_right_wrist"][t],
                        "state":             ep["qpos"][t].astype(np.float32),
                    },
                    "action":       ep["action"][t].astype(np.float32),
                    "is_first":     t == 0,
                    "is_last":      t == T - 1,
                    "is_terminal":  t == T - 1,
                    "reward":       float(t == T - 1),
                    "language_instruction": LANGUAGE_INSTRUCTION,
                })

            yield f"{ep_idx:06d}_{Path(hdf5_path).stem}", {
                "steps": steps,
                "episode_metadata": {"file_path": hdf5_path},
            }
