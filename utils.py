import os
import soundfile as sf

test_speakers = [19, 26, 39, 40, 78, 83, 87, 89, 118, 125, 163, 196, 198, 200, 201, 250, 254, 307, 405, 446]

path = '/data/common/LibriTTS/train-clean-100'
num_files = 100

spk_dict = {}

for s in test_speakers:
    sec = 0
    if s not in spk_dict:
        spk_dict[s] = []
    spk_path = os.path.join(path, str(s))
    files = []
    for root, d , fnames in os.walk(spk_path):
        for f  in fnames:
            if 'wav' in f:
                files.append(os.path.join(root, f))

    files = files[:max(num_files, len(files))]
    sorted_files = []
    for f in files:
        if sec < 30:
            fpath = os.path.join(path, str(s), f)
            wav, fs = sf.read(fpath)
            time = len(wav)/fs
            sorted_files.append((fpath, time))
        else:
            break
    sorted_files = sorted(sorted_files, key=lambda x: x[1], reverse=True)
    spk_dict[s] = sorted_files

with open('libri_select.txt', 'w') as fo:
    for k in spk_dict:
        for f in spk_dict[k]:
            line = f"{k}\ttest\t{f[0]}\n"
            fo.write(line)