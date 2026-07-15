import soundfile as sf
import os

cases = [
    ("eot_data/english/audio/en__044.wav", 7.20, 8.80, "fp_en044"),
    ("eot_data/hindi/audio/hi__022.wav", 13.90, 18.10, "fp_hi022"),
    ("eot_data/english/audio/en__007.wav", 23.00, 26.29, "fn_en007"),
    ("eot_data/english/audio/en__060.wav", 4.80, 8.20, "fn_en060"),
    ("eot_data/hindi/audio/hi__052.wav", 5.10, 8.29, "fn_hi052"),
]

os.makedirs("clips", exist_ok=True)
for path, start, end, name in cases:
    x, sr = sf.read(path)
    s, e = int(start * sr), int(min(end + 0.5, len(x)/sr) * sr)
    sf.write(f"clips/{name}.wav", x[s:e], sr)
    print(f"wrote clips/{name}.wav  ({start}s to {end}s)")
