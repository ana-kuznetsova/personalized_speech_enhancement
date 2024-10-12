import json
import numpy as np
import torch
from TTS.tts.utils.synthesis import synthesis
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
import scipy


tts_model_config_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/config.json'
tts_model_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth'
#tts_spk_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/speakers.json'
#tts_lang_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/language_ids.json'
#tts_enc_ckpt = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_se.pth'
#tts_enc_conf_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/config_se.json'


tts_config = load_config(tts_model_config_path)
tts_model = setup_tts_model(config=tts_config)
tts_model.load_checkpoint(tts_config, tts_model_path, eval=True)



def synthesize_speech(average_dvector,tts_model,tts_config,wavname):
    if len(average_dvector) == 0 :
        print(f"Not enough speakers , Skipping!")
        return
    outputs = synthesis(model=tts_model,
                    text="Do I sound more like Barack Obama or Donald Trump or even both?",
                    CONFIG=tts_config, use_cuda=False, d_vector=average_dvector, language_id=0)
    waveform = outputs["wav"]
    scipy.io.wavfile.write(wavname, 16000, waveform)


with open('../interp.json', 'r') as fo:
    embeds = json.load(fo)

source_emb = torch.tensor(embeds['source'])
target_emb = torch.tensor(embeds['target'])
interp_emb = torch.tensor(embeds['z_source_close'])

out_path = '/home/anakuzne/projects/anastasia_code/interp_out'
wavname = f"{out_path}/z_close_source.wav"
synthesize_speech(interp_emb,tts_model,tts_config, wavname)