import os
from faster_whisper import WhisperModel
from langdetect import detect_langs, DetectorFactory
from speechbrain.inference import EncoderClassifier
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DetectorFactory.seed = 0

sb_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp_langid"
)
vad_model = load_silero_vad()
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

LANG_ALIASES = {
    "br": "fr",
    "oc": "fr",
    "cy": "fr",
}

def normalize_lang(lang: str) -> str:
    return lang.split(":")[0].strip()

def detect_languages_in_audio(audio_path: str) -> set:
    wav = read_audio(audio_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)

    langs = set()

    for ts in speech_timestamps:
        segment = wav[ts['start']:ts['end']].unsqueeze(0)
        if segment.shape[1] < 2 * 16000:
            continue

        out_prob, score, index, text_lab = sb_classifier.classify_batch(segment)
        confidence = torch.exp(out_prob[0].max()).item()
        if confidence >= 0.85:
            langs.add(normalize_lang(text_lab[0]))

    segments, _ = whisper_model.transcribe(audio_path, language=None, beam_size=5)
    for seg in segments:
        text = seg.text.strip()
        if len(text) < 40:
            continue
        try:
            for d in detect_langs(text):
                if d.prob >= 0.99:
                    langs.add(d.lang)
        except:
            continue

    langs = {LANG_ALIASES.get(l, l) for l in langs}

    return langs

print(detect_languages_in_audio("audio_test/IMOCA_STOPOVER_4_ITAJAI.wav"))