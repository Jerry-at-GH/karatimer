# SPDX-License-Identifier: Apache-2.0
"""media + lyric TSV → timing TSV."""
import argparse
import csv
import re
import unicodedata
from itertools import chain
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="align Japanese karaoke units from a TSV")
    parser.add_argument("media", type=Path, help="any supported media file; uses its first audio track")
    parser.add_argument("lyrics", type=Path, help="TSV columns: line, text, reading (romanized Japanese)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output TSV")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="default: CUDA when available")
    args = parser.parse_args()
    lines = []
    with args.lyrics.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"line", "text", "reading"} <= set(reader.fieldnames or []):
            raise ValueError("lyrics TSV needs line, text and reading columns")
        for number, row in enumerate(reader, 2):
            line = int(row["line"])
            reading = "".join(c for c in unicodedata.normalize("NFKD", (row["reading"] or "").lower().replace("’", "'")) if not unicodedata.combining(c))
            if line < 1 or not row["text"] or not re.fullmatch("[a-z']+", reading) or not re.search("[a-z]", reading):
                raise ValueError(f"invalid lyric unit on TSV row {number}: use a positive line number, display text and romanized reading")
            if lines and line < lines[-1][0]["line"]:
                raise ValueError("line numbers must be in ascending order")
            if not lines or line != lines[-1][0]["line"]:
                lines.append([])
            lines[-1].append(dict(line=line, text=row["text"], reading=reading))
    if not lines:
        raise ValueError("lyrics are empty")

    import av
    import numpy as np
    import torch
    from .alignment import align
    with av.open(str(args.media)) as media:
        if not media.streams.audio:
            raise ValueError("media has no audio track")
        stream = media.streams.audio[0]
        origin = 1000 * (float((stream.start_time or 0) * stream.time_base)
                         - (media.start_time or 0) / av.time_base)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        samples = [output.to_ndarray().reshape(-1)
                   for frame in chain(media.decode(stream), [None])
                   for output in resampler.resample(frame)]
    if sum(chunk.size for chunk in samples) < 400:
        raise ValueError("audio must contain at least 25 ms")
    wave = np.concatenate(samples).astype(np.float32) / 32768
    align(wave, lines, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["line", "text", "reading", "start_ms", "end_ms"])
        for line in lines:
            for unit in line:
                writer.writerow([unit["line"], unit["text"], unit["reading"],
                                 *(int(np.floor((unit[edge] + origin) / 10 + .5)) * 10 for edge in ["start_ms", "end_ms"])])


if __name__ == "__main__":
    main()
