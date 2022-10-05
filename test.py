import tgt


SILENCE = ["sil", "sp", "spn"]
path = "/mnt/d/Projects/TTS-systems/preprocessed_data/LibriTTS/TextGrid/242/242_122625_000005_000000.TextGrid"
path = "/mnt/d/Projects/FastSpeech2-Kaiwei/preprocessed/TAT-TTS/TextGrid/M2/M2-M2_1-4.TextGrid"
info = tgt.io.read_textgrid(path, include_empty_intervals=True)
tier = info.get_tier_by_name("phones")
   
phones = []
durations = []
# start_time, end_time = 0, 0
end_idx = 0
for t in tier._objects:
    s, e, p = t.start_time, t.end_time, t.text
    if p == '':
        if t.start_time == 0:
            p = 'sil'
        else:
            p = 'sp'
    print("Cur:", p)

    # Trim leading silences
    if phones == []:
        if p in SILENCE:
            continue
        else:
            pass
            # start_time = s
    phones.append(p)
    durations.append((s, e))
    if p not in SILENCE:
        # end_time = e
        end_idx = len(phones)

durations = durations[:end_idx]
phones = phones[:end_idx]
print(phones)
