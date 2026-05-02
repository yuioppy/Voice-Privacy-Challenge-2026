from pathlib import Path

import pyarrow.parquet as pq


DATA_ROOT = Path("data")
PARQUET_ROOT = Path("IEMOCAP/data")
SPLITS = ["IEMOCAP_dev", "IEMOCAP_test"]


def load_targets():
    targets = {}
    for split in SPLITS:
        wav_scp = DATA_ROOT / split / "wav.scp"
        with wav_scp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                utt, wav_path = line.strip().split(maxsplit=1)
                targets[utt] = Path(wav_path)
    return targets


def main():
    wav_root = DATA_ROOT / "IEMOCAP" / "wav"
    if wav_root.is_symlink():
        wav_root.unlink()
    wav_root.mkdir(parents=True, exist_ok=True)

    targets = load_targets()
    written = 0
    skipped = 0
    seen = set()

    parquet_files = sorted(PARQUET_ROOT.glob("IEMOCAP*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No IEMOCAP parquet files found in {PARQUET_ROOT}")

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(columns=["file", "audio"], batch_size=128):
            files = batch.column(0).to_pylist()
            audios = batch.column(1).to_pylist()
            for file_name, audio in zip(files, audios):
                utt = Path(file_name).stem
                out_path = targets.get(utt)
                if out_path is None:
                    continue
                seen.add(utt)
                if out_path.exists():
                    skipped += 1
                    continue
                audio_bytes = audio["bytes"]
                if audio_bytes is None:
                    raise ValueError(f"Missing audio bytes for {utt} in {parquet_path}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(audio_bytes)
                written += 1

    missing = sorted(set(targets) - seen)
    print(f"Targets: {len(targets)}")
    print(f"Written: {written}")
    print(f"Skipped existing: {skipped}")
    print(f"Missing in parquet: {len(missing)}")
    if missing:
        print("First missing:", ", ".join(missing[:10]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
