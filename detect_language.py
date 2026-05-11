from speechbrain.inference import EncoderClassifier
import soundfile as sf
import torch

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp_langid"
)

def detect_languages_segments(audio_path: str, segment_sec: int = 10) -> list:
    waveform, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    
    waveform = waveform.mean(axis=1)
    
    segment_samples = segment_sec * sr
    total_samples = len(waveform)
    results = []

    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segment = waveform[start:end]
        
        if len(segment) < 2 * sr:
            continue
        
        segment_tensor = torch.tensor(segment).unsqueeze(0)
        
        out_prob, score, index, text_lab = classifier.classify_batch(segment_tensor)
        results.append({
            "start": round(start / sr, 1),
            "end": round(end / sr, 1),
            "language": text_lab[0],
            "confidence": round(score[0].item(), 3)
        })

    return results

segments = detect_languages_segments("PRB_Bapteme.wav")
for s in segments:
    print(f"[{s['start']}s → {s['end']}s] {s['language']} ({s['confidence']:.0%})")

langs = set(s['language'] for s in segments)
print(f"\nLangues détectées : {langs}")