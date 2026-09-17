# SPDX-License-Identifier: Apache-2.0
"""three-view NextFire alignment followed by gated kana/phoneme onset refinement."""
import re
import unicodedata
from pathlib import Path

import jaconv
import numpy as np
import pyopenjtalk
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from numba import njit
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor

NEXTFIRE_MODEL = "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"
NEXTFIRE_REVISION = "2ab2b5f46539ee284703c281f286b01d2410ee12"
SAKASEGAWA_MODEL = "sakasegawa/japanese-wav2vec2-large-hiragana-ctc"
SAKASEGAWA_REVISION = "d30d246cd24a225821d03d183de7bb2e769e18df"
# sakasegawa vocabularies/conversion adapted and simplified from
# nyosegawa/hiragana-asr (Apache-2.0); see NOTICE-sakasegawa.
KANA = {c: i + 1 for i, c in enumerate("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽぁぃぅぇぉっゃゅょゎー")}
PHONES = {p: i + 1 for i, p in enumerate("A E I N O U a b by ch cl d dy e f g gy h hy i j k ky m my n ny o p py r ry s sh t ts ty u v w y z".split())}
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "エー ビー シー ディー イー エフ ジー エイチ アイ ジェー ケー エル エム エヌ オー ピー キュー アール エス ティー ユー ブイ ダブリュー エックス ワイ ゼット".split()))


def tokens(text):
    text = re.sub(r"\s+", " ", text.replace("・", " ")).strip()
    if not text:
        return [], []
    kana = unicodedata.normalize("NFKC", pyopenjtalk.g2p(text, kana=True))
    kana = "".join(LETTERS.get(c.upper(), c) for c in kana)
    kana = "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in kana)
    return ([KANA[c] for c in kana if c in KANA],
            [PHONES[p] for p in pyopenjtalk.g2p(text).split() if p in PHONES])


@njit
def ctc_starts(logp, ids):
    frames = len(logp)
    count = 2 * len(ids) + 1
    states = np.zeros(count, np.int64)
    states[1::2] = ids
    prev = np.full(count, -np.inf)
    prev[0] = 0.0
    trace = np.zeros((frames, count), np.int8)
    for t in range(frames):
        current = np.full(count, -np.inf)
        for s in range(count):
            best = prev[s]
            move = 0
            if s > 0 and prev[s - 1] > best:
                best = prev[s - 1]
                move = 1
            if (
                s > 1
                and states[s] != 0
                and states[s] != states[s - 2]
                and prev[s - 2] > best
            ):
                best = prev[s - 2]
                move = 2
            current[s] = best + logp[t, states[s]]
            trace[t, s] = move
        prev = current
    state = count - 1 if prev[count - 1] >= prev[count - 2] else count - 2
    starts = np.full(len(ids), -1, np.int64)
    for t in range(frames - 1, -1, -1):
        if state % 2:
            starts[state // 2] = t
        state -= trace[t, state]
    return starts


def align(wave, lines, device):
    """add start_ms/end_ms in place; all units share the TSV data structure."""
    torch.set_num_threads(4)
    processor = Wav2Vec2Processor.from_pretrained(NEXTFIRE_MODEL, revision=NEXTFIRE_REVISION)
    model = Wav2Vec2ForCTC.from_pretrained(NEXTFIRE_MODEL, revision=NEXTFIRE_REVISION).to(device).eval()
    feature = model.wav2vec2.feature_extractor
    native = feature.forward

    def forward(wave):
        # exact convolution blocks retain full-song transformer context.
        frames = (wave.shape[-1] - 400) // 320 + 1
        return torch.cat([native(wave[..., start * 320:(min(frames, start + 1000) - 1) * 320 + 400])
                          for start in range(0, frames, 1000)], dim=-1)

    feature.forward = forward
    groups = [processor.tokenizer.encode(unit["reading"], add_special_tokens=False) for line in lines for unit in line]
    targets = torch.tensor([[token for group in groups for token in group]], dtype=torch.int32)
    views = []
    with torch.inference_mode():
        for rate in [16000, 16800, 15200]:
            audio = wave if rate == 16000 else torchaudio.functional.resample(torch.from_numpy(wave), rate, 16000).numpy()
            inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt").to(device)
            logp = model(**inputs).logits.float().log_softmax(-1).cpu()
            path, scores = torchaudio.functional.forced_align(logp, targets, blank=processor.tokenizer.pad_token_id)
            spans = torchaudio.functional.merge_tokens(path[0], scores[0], blank=processor.tokenizer.pad_token_id)
            frame_ms = len(wave) / 16 / logp.shape[1]
            units, offset = [], 0
            for group in groups:
                units.append((spans[offset].start * frame_ms, spans[offset + len(group) - 1].end * frame_ms))
                offset += len(group)
            boundaries, offset = [], 0
            for line in lines:
                part = units[offset:offset + len(line)]
                boundaries.append([start for start, _ in part] + [part[-1][1]])
                offset += len(line)
            views.append(boundaries)
    for index, line in enumerate(lines):
        matrix = np.array([view[index] for view in views])
        boundaries = np.where(np.ptp(matrix, axis=0) <= 100, matrix.mean(0), matrix[0])
        boundaries = np.maximum.accumulate(np.maximum(0, boundaries + 10))
        for unit, start, end in zip(line, boundaries, boundaries[1:]):
            unit.update(start_ms=float(start), end_ms=float(end))
    model.cpu()
    del model, processor, inputs, logp
    if device == "cuda":
        torch.cuda.empty_cache()

    checkpoint = torch.load(hf_hub_download(SAKASEGAWA_MODEL, "best-medium-ep5-inference.pt", revision=SAKASEGAWA_REVISION),
                            map_location="cpu", weights_only=True, mmap=True)
    layer = checkpoint["inter_ctc_layer"]
    state = checkpoint["model_state_dict"]
    config = Wav2Vec2Config.from_pretrained(Path(__file__).with_name("sakasegawa-config.json"))
    config.apply_spec_augment = False
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = nn.Module()
        model.encoder = Wav2Vec2Model(config)
        model.kana_head = nn.Linear(config.hidden_size, state["kana_head.weight"].shape[0])
        model.phoneme_head = nn.Linear(config.hidden_size, state["phoneme_head.weight"].shape[0])
    model.load_state_dict(state, assign=True)
    model.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32).eval()
    parameter = next(model.parameters())
    # freeze locator boundaries: refinements must not alter later crops.
    locations = [(line[0]["start_ms"], line[-1]["end_ms"]) for line in lines]
    with torch.inference_mode():
        for center, line in enumerate(lines):
            context = list(range(max(0, center - 1), min(len(lines), center + 2)))
            while len(context) > 1 and locations[context[-1]][1] - locations[context[0]][0] > 27000:
                context.remove(context[0] if center - context[0] >= context[-1] - center else context[-1])
            left = max(0, int((locations[context[0]][0] - 300) * 16))
            right = min(len(wave), int((locations[context[-1]][1] + 300) * 16))
            if not 400 < right - left <= 480000:
                continue
            spoken, prefixes = "", []
            for index in context:
                kana = jaconv.alphabet2kana("".join(unit["reading"] for unit in lines[index]))
                prefix = ""
                for unit in lines[index]:
                    a = jaconv.alphabet2kana(prefix)
                    prefix += unit["reading"]
                    b = jaconv.alphabet2kana(prefix)
                    if index == center:
                        prefixes.append((spoken + a, kana.startswith(a) and kana.startswith(b) and len(b) > len(a)))
                spoken += kana + " "
            targets = tokens(spoken)
            cuts = [tokens(prefix) for prefix, _ in prefixes]
            clip = torch.from_numpy(wave[None, left:right].copy()).to(parameter.device)
            clip = ((clip - clip.mean(-1, keepdim=True)) / torch.sqrt(clip.var(-1, unbiased=False, keepdim=True) + 1e-7)).to(parameter.dtype)
            hidden = model.encoder(clip, attention_mask=torch.ones_like(clip, dtype=torch.long), output_hidden_states=True)
            heads = [(model.kana_head, hidden.last_hidden_state), (model.phoneme_head, hidden.hidden_states[layer])]
            onsets = []
            for index, ((head, features), ids) in enumerate(zip(heads, targets)):
                logp = head(features).float().log_softmax(-1).cpu()
                if not ids or len(logp[0]) < len(ids) + sum(a == b for a, b in zip(ids, ids[1:])):
                    onsets.append([None] * len(prefixes))
                    continue
                starts = ctc_starts(logp[0].numpy(), np.asarray(ids, dtype=np.int64))
                onsets.append([left / 16 + starts[len(cut[index])] * 20
                               if valid and len(cut[index]) < len(ids) and ids[:len(cut[index])] == cut[index] else None
                               for cut, (_, valid) in zip(cuts, prefixes)])
            old = np.array([unit["start_ms"] for unit in line] + [line[-1]["end_ms"]])
            new = old.copy()
            for index, (kana, phone) in enumerate(zip(*onsets)):
                if kana is not None and phone is not None and abs(kana - phone) <= 40 and abs(kana - old[index]) <= 160:
                    new[index] = kana
            if np.any(np.diff(new) < 0):
                continue
            for unit, start, end in zip(line, new, new[1:]):
                unit.update(start_ms=float(start), end_ms=float(end))
