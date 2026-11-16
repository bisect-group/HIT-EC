from __future__ import annotations

import json
import pickle
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_DATASET_ROOT = EMULATOR_ROOT / "data" / "processed" / "datasets" / "enzyme_classification_dataset"
DEFAULT_OUT_ROOT = EMULATOR_ROOT / "outputs" / "baselines" / "HITEC"
TOKENIZER_PATH = BASELINE_ROOT / "utils" / "tokenizer.pickle"
TASK = "enzyme_classification"
DEFAULT_SEEDS = (666, 777, 888)
SPLIT_NAMES = ("train", "val", "test")
MODEL_DIMENSION = 1024
BOS_ID = 22


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_group_slug(split_group):
    return str(split_group).replace("/", "__")


def task_root(out_root):
    return Path(out_root) / TASK


def split_run_root(out_root, split_group):
    """<out_root>/<task>/<split_group>/<split_name>; a flat split group uses split_name "all"."""
    group, _, name = str(split_group).partition("/")
    return task_root(out_root) / group / (name or "all")


def vocab_path(out_root):
    return Path(out_root) / "vocab" / "ec_vocab.json"


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def load_run_metadata(split_group, out_root):
    path = split_run_root(out_root, split_group) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError("Missing run metadata: {}".format(path))
    return read_json(path)


def seed_run_root(metadata, seed):
    return Path(metadata["run_root"]) / "seed_{}".format(seed)


def seed_train_metadata_path(metadata, seed):
    return seed_run_root(metadata, seed) / "train.json"


def seed_results_root(metadata, seed):
    return seed_run_root(metadata, seed)


def load_residue_ids():
    """Residue-to-id map of HITEC's Keras tokenizer, read without importing keras."""

    class _Stub:
        pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return _Stub if module.startswith("keras") else super().find_class(module, name)

    with TOKENIZER_PATH.open("rb") as handle:
        return dict(_Unpickler(handle).load().word_index)


def tokenize_sequence(sequence, residue_ids):
    """BOS token plus residue ids, truncated to MODEL_DIMENSION and zero padded."""
    ids = [BOS_ID] + [residue_ids[char] for char in sequence.lower()]
    ids = ids[:MODEL_DIMENSION]
    return ids + [0] * (MODEL_DIMENSION - len(ids))


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = "{}{}".format(prefix, key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=name + "."))
        else:
            flat[name] = value
    return flat
