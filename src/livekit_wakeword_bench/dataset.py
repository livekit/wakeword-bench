"""Dataset format: Hugging Face parquet shards with an Audio column + per-sample metadata."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

FORMAT_VERSION = "1"

DEFAULT_DATASET = "binhpham/wakeword-bench-hey-livekit"

CATEGORIES = ("positive", "adversarial", "speech", "real_speech", "noise", "media")

# column -> default ("" / NaN / -1 when not applicable)
SCHEMA: dict[str, Any] = {
    "id": "",
    "split": "",
    "label": 0,  # 1 = contains the wake word
    "category": "",  # one of CATEGORIES
    "wake_word": "",
    "text": "",  # TTS input text or real-speech transcript
    "text_form": "",  # isolated | carrier_prefix | carrier_suffix | filler | transcript | ""
    "phrase": "",  # key phrase: wake word surface form, or the confusable phrase
    "adversarial_distance": -1,  # phoneme edit distance of the confusable to the wake word
    "provider": "",  # TTS provider, or the corpus name for real speech
    "model": "",
    "voice_id": "",
    "voice_name": "",
    "gender": "",
    "accent": "",
    "held_out_voice": False,  # voice never used in train/validation splits
    "speed": 1.0,
    "noise_bucket": "clean",  # clean | quiet | moderate | noisy | very_noisy | extreme
    "snr_db": math.nan,
    "noise_source": "",
    "noise_type": "",  # coarse: babble, music, domestic, office, street, transport, nature, ...
    "noise_category": "",  # fine: <noise_type>/<detail>
    "noise_file": "",
    "reverb": False,
    "rir_source": "",
    "rir_file": "",
    "rir_category": "",
    "interferer": "",  # competing-talker corpus, "" if none
    "sir_db": math.nan,  # target speech vs competing talker (party: the loudest one)
    "party": False,  # several overlapping talkers + crowd babble (noise_bucket "party")
    "n_talkers": 0,  # competing talkers in a party scene
    "noise2_category": "",  # second layered noise
    "eq": "",  # mic coloration description, "" if none
    "bandlimit": "none",
    "clipping": False,
    "codec": "none",  # opus_8k | opus_16k | opus_32k | mp3_32k | mulaw_8khz | none
    "augmentations": "",  # summary, e.g. "reverb+noise+codec"
    "level_dbfs": math.nan,
    "onset_s": math.nan,  # whole utterance start/end in the clip
    "offset_s": math.nan,
    "phrase_onset_s": math.nan,  # key phrase (wake word / confusable) start/end in the clip
    "phrase_offset_s": math.nan,
    "duration_s": 0.0,
    "tts_key": "",
    "phonemes": "",  # ARPAbet of the key phrase (words " . ", morphemes " | ")
    "tts_input": "",  # exact key-phrase string sent to the TTS (provider phoneme syntax)
}


def encode_flac(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def decode_audio(value: Any) -> tuple[np.ndarray, int]:
    """Decode an HF Audio cell (decoded dict, raw bytes dict, or AudioDecoder) to float32 mono."""
    if isinstance(value, dict):
        if value.get("array") is not None:
            arr = np.asarray(value["array"], dtype=np.float32)
            return arr, int(value["sampling_rate"])
        data, sr = sf.read(io.BytesIO(value["bytes"]), dtype="float32")
        return (data.mean(axis=1) if data.ndim > 1 else data), sr
    # datasets>=4 returns a torchcodec AudioDecoder
    samples = value.get_all_samples()
    arr = samples.data.numpy()
    arr = arr.mean(axis=0) if arr.ndim > 1 else arr
    return arr.astype(np.float32), int(samples.sample_rate)


@dataclass
class ShardWriter:
    root: Path
    split: str
    sample_rate: int
    shard_size: int = 1000
    _rows: list[dict[str, Any]] = field(default_factory=list)
    _shard: int = 0
    n_rows: int = 0
    seconds: float = 0.0

    def add(self, audio: np.ndarray, meta: dict[str, Any]) -> None:
        row = {**SCHEMA, **meta}
        row["split"] = self.split
        row["duration_s"] = round(len(audio) / self.sample_rate, 4)
        row["audio"] = {"bytes": encode_flac(audio, self.sample_rate), "path": f"{row['id']}.flac"}
        self._rows.append(row)
        self.n_rows += 1
        self.seconds += row["duration_s"]
        if len(self._rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        out = self.root / "data" / f"{self.split}-{self._shard:05d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._rows, schema=arrow_schema())
        pq.write_table(table, out, row_group_size=100)
        self._rows.clear()
        self._shard += 1


def arrow_schema() -> Any:
    import pyarrow as pa

    def typ(v: Any) -> Any:
        if isinstance(v, bool):
            return pa.bool_()
        if isinstance(v, int):
            return pa.int32()
        if isinstance(v, float):
            return pa.float32()
        return pa.string()

    fields = [pa.field(k, typ(v)) for k, v in SCHEMA.items()]
    fields.append(
        pa.field(
            "audio", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
        )
    )
    # HF reads this to know the struct is an Audio feature.
    hf_features = {k: _hf_value_type(v) for k, v in SCHEMA.items()}
    hf_features["audio"] = {"_type": "Audio", "sampling_rate": None}
    import json

    meta = {b"huggingface": json.dumps({"info": {"features": hf_features}}).encode()}
    return pa.schema(fields, metadata=meta)


def _hf_value_type(v: Any) -> dict[str, str]:
    if isinstance(v, bool):
        return {"dtype": "bool", "_type": "Value"}
    if isinstance(v, int):
        return {"dtype": "int32", "_type": "Value"}
    if isinstance(v, float):
        return {"dtype": "float32", "_type": "Value"}
    return {"dtype": "string", "_type": "Value"}


def load_bench(source: str | Path, split: str = "test", streaming: bool = False) -> Any:
    """Load a benchmark split from a local directory or a Hugging Face dataset repo id."""
    import datasets

    p = Path(source)
    if p.exists():
        files = sorted(str(f) for f in (p / "data").glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split} shards under {p / 'data'}")
        ds = datasets.load_dataset(
            "parquet", data_files={split: files}, split=split, streaming=streaming
        )
    else:
        ds = datasets.load_dataset(str(source), split=split, streaming=streaming)
    return ds.cast_column("audio", datasets.Audio(decode=False))
