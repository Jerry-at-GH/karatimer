# karatimer

models used: [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn) and [sakasegawa/japanese-wav2vec2-large-hiragana-ctc](https://huggingface.co/sakasegawa/japanese-wav2vec2-large-hiragana-ctc).

decode original audio to mono 16 kHz. supply untimed lyrics grouped into lines and romanized karaoke units. run NextFire over each complete song using waveform normalization and CTC forced alignment. generate three views: original audio, plus resampling from assumed rates of 16,800 and 15,200 Hz to 16,000 Hz. map predictions back to original duration. average boundaries when their three-view range is ≤100 ms; otherwise retain the original-view boundary. add 10 ms. internal unit ends equal following onsets.

for each line, crop raw audio using these predictions, including neighboring lyric lines and 300 ms margins. remove neighboring context when its span exceeds 27 seconds. convert romanized lyrics to kana using jaconv; require exact token-prefix correspondence for unit boundaries.

run sakasegawa’s kana and phoneme heads together. replace an onset only when both heads agree within 40 ms and displacement from NextFire is ≤160 ms. move the preceding internal end together; preserve line-final ends. reject crossings and retain unsupported boundaries.

in Japanese karaoke tests, this combination improved both ≤50 ms accuracy by ~8 pp over NextFire alone. this is an observed gain, not a guarantee for every song.

## usage

requires Python 3.12+ and a compatible PyTorch/TorchAudio installation. PyAV handles media decoding; its standard wheels bundle FFmpeg libraries, so no separate `ffmpeg` or `ffprobe` executable is needed. model weights download automatically from Hugging Face. no intermediate audio or result caches are written. CUDA is selected when available; use `--device cpu` otherwise.

```sh
pip install -e .
karatimer song.mp4 lyrics.tsv -o timing.tsv
```

media may be any audio/video format supported by the installed PyAV build, regardless of extension. the first audio track is always used, even if another track is marked as default. files without audio are rejected.

lyrics input is UTF-8 TSV: one unit per row, ascending positive line numbers, display text, and romanized Japanese reading. choose unit boundaries yourself; Japanese display text does not require automatic kanji reading. for example:

```tsv
line	text	reading
1	き	ki
1	み	mi
2	愛	ai
```

output preserves line numbers and display text, normalizes readings to lowercase unaccented romanization, and adds `start_ms` and `end_ms`, relative to the input media’s start and rounded to 10 ms. consecutive units within a line share a boundary. output can be used as input again; existing timings are ignored. use actual tabs between fields.

models run sequentially to limit VRAM use. full-song NextFire attention still grows with song length. simultaneous independent lyric tracks are unsupported.
