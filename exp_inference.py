from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from espnet_model_zoo.downloader import ModelDownloader

from TTS.tts.utils.synthesis import synthesis
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model

from TTS.utils.synthesizer import Synthesizer



import glob
import os
import numpy as np
import kaldiio
import sys
import json
from tqdm import tqdm
import soundfile as sf
import torch
import pandas as pd
import subprocess
import librosa

def load_xvectors_all(model_dir):
    ark_dump = f"{model_dir}/../../dump/xvector"
    tr = os.path.join(ark_dump, 'tr_no_dev', 'spk_xvector.ark ')



def init_model():
    
    lang = 'English'
    #tag = 'kan-bayashi/libritts_xvector_conformer_fastspeech2' #@param ["kan-bayashi/vctk_gst_tacotron2", "kan-bayashi/vctk_gst_transformer", "kan-bayashi/vctk_xvector_tacotron2", "kan-bayashi/vctk_xvector_transformer", "kan-bayashi/vctk_xvector_conformer_fastspeech2", "kan-bayashi/vctk_gst+xvector_tacotron2", "kan-bayashi/vctk_gst+xvector_transformer", "kan-bayashi/vctk_gst+xvector_conformer_fastspeech2", "kan-bayashi/vctk_multi_spk_vits", "kan-bayashi/vctk_full_band_multi_spk_vits", "kan-bayashi/libritts_xvector_transformer", "kan-bayashi/libritts_xvector_conformer_fastspeech2", "kan-bayashi/libritts_gst+xvector_transformer", "kan-bayashi/libritts_gst+xvector_conformer_fastspeech2", "kan-bayashi/libritts_xvector_vits"] {type:"string"}
    #vocoder_tag = "parallel_wavegan/libritts_hifigan.v1" #@param ["none", "parallel_wavegan/vctk_parallel_wavegan.v1.long", "parallel_wavegan/vctk_multi_band_melgan.v2", "parallel_wavegan/vctk_style_melgan.v1", "parallel_wavegan/vctk_hifigan.v1", "parallel_wavegan/libritts_parallel_wavegan.v1.long", "parallel_wavegan/libritts_multi_band_melgan.v2", "parallel_wavegan/libritts_hifigan.v1", "parallel_wavegan/libritts_style_melgan.v1"] {type:"string"}
    tag = 'kan-bayashi/vctk_gst_fastspeech'
    vocoder_tag = 'parallel_wavegan/vctk_hifigan.v1'
    text2speech = Text2Speech.from_pretrained(
                            model_tag=tag,
                            vocoder_tag=vocoder_tag,
                            device="cuda",
                            # Only for FastSpeech & FastSpeech2 & VITS
                            speed_control_alpha=1.0,
                            threshold=0.5,
                            # Only for Tacotron 2
                            minlenratio=0.0,
                            maxlenratio=10.0,
                            use_att_constraint=False,
                            backward_window=1,
                            forward_window=3,
                            # Only for VITS
                            noise_scale=0.333,
                            noise_scale_dur=0.333,)


    d = ModelDownloader()
    model_dir = os.path.dirname(d.download_and_unpack(tag)["train_config"])
    '''
    if text2speech.use_spembs:
        xvector_arks = [p for p in glob.glob(f"{model_dir}/../../dump/**/spk_xvector.ark", recursive=True) if "tr" in p]
    
        xvectors = {}
        for i in xvector_arks:
            for k, v in kaldiio.load_ark(i):
                xvectors[k] = v
    '''
    #return text2speech, xvectors
    return text2speech




def select_xvec(spk, xvector_dict, ignore_utters=None):
    target_duration = 5
    
    #Filter speaker utters
    speakers = list(xvector_dict.keys())
    spk_xvecs = [i for i in speakers if str(spk) in i]
    
    random_spk_idx = np.random.randint(0, len(spk_xvecs))
    spk_utter =  spk_xvecs[random_spk_idx]
    spembs = xvectors[spk_utter]
    print(spembs.shape)
    return spembs


def synthesize(text2speech, spk_id, xvector_dict, out_dir, texts):
    
    #randomly sample text until num seconds is reached 
    print(f"DEBUG spk embeds")

    spk_emb = select_xvec(spk_id, xvector_dict)

    sec = 120
    num_utter = 0
    while sec >= 0:
        text_id = np.random.randint(0, len(texts))
        text = texts[text_id]
    
        with torch.no_grad():
            wav = text2speech(text, speech=None, spembs=spk_emb, sids=None)["wav"].detach().cpu().numpy()
            time = len(wav)/text2speech.fs
            sec-=time
            path = os.path.join(out_dir, str(spk_id))
            if not os.path.exists(path):
                os.makedirs(path)
            
            fname = f"{spk_id}_{num_utter}.wav"
            sf.write(os.path.join(path, fname), wav, text2speech.fs)
            num_utter+=1


def synthesize_yourtts(spk, texts, out_dir):
    '''
    tts  --text 
     --model_name tts_models/multilingual/multi-dataset/your_tts  
     --speaker_wav /data/common/LibriTTS/train-clean-100/200/126784/200_126784_000062_000001.wav 
     --language_idx "en" 
     --out_path output/path/speech.wav
    '''
    target_duration = 5
    df = pd.read_csv('/data/common/librispeech/dataframe.csv')
    utterances = df.query("speaker_id == @spk")
    idx = (utterances.duration - target_duration).abs().idxmin()

    reference_wav  = df.iloc[idx]['filepath']
    
    lang_id = f'"en"'
    fs = 16000
    sec = 40
    num_utter = 0
    while sec >= 0:
        text_id = np.random.randint(0, len(texts))
        text = f'"{texts[text_id]}"'

        fname = f"{spk_id}_{num_utter}.wav"
        out_path = os.path.join(out_dir, str(spk))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, fname)
        
    
        cmd = ['tts',  '--text', text, '--model_name', 'tts_models/multilingual/multi-dataset/your_tts', '--speaker_wav', reference_wav,\
               '--language_idx',  lang_id, '--out_path', out_path, '--use_cuda 0']

        cmd = ' '.join(cmd)
    
        subprocess.run(cmd, shell=True,  check=True, capture_output=True)
        wav, fs = sf.read(out_path)
        time = len(wav)/fs
        sec-=time
        num_utter+=1


def synthesize_yourtts2(spk, texts,  out_dir):
    tts_model_config_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/config.json'
    tts_model_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth'
    tts_spk_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/speakers.json'
    tts_lang_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/language_ids.json'
    tts_enc_ckpt = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_se.pth'
    tts_enc_conf_path = '/home/anakuzne/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/config_se.json'

    synth = Synthesizer(tts_checkpoint=tts_model_path,
                        tts_config_path=tts_model_config_path,
                        tts_speakers_file=tts_spk_path,
                        tts_languages_file=tts_lang_path,
                        encoder_checkpoint=tts_enc_ckpt,
                        encoder_config=tts_enc_conf_path,
                        use_cuda=True)
    

    target_duration = 5

    df = pd.read_csv('/data/common/librispeech/dataframe.csv')
    utterances = df.query("speaker_id == @spk")
    idx = (utterances.duration - target_duration).abs().idxmin()

    reference_wav  = df.iloc[idx]['filepath']

    lang_id = f'"en"'
    fs = 16000
    sec = 70
    num_utter = 0
    while sec >= 0:
        text_id = np.random.randint(0, len(texts))
        text = f'"{texts[text_id]}"'

        fname = f"{spk_id}_{num_utter}.wav"
        out_path = os.path.join(out_dir, str(spk))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, fname)
        wav = synth.tts(text=text, language_name="en", speaker_wav=reference_wav)
        wav = np.array(wav)
        sf.write(out_path, wav, fs)
        time = len(wav)/fs
        sec-=time
        num_utter+=1




out_dir = sys.argv[1]
test_speakers = [19, 26, 39, 40, 78, 83, 87, 89, 118, 125, 163, 196, 198, 200, 201, 250, 254, 307, 405, 446]
#test_speakers = [4446, 1284, 1089, 4970, 3575, 121]
#test_speakers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

with open('all.json', 'r') as fo:
    data = json.load(fo)
    
texts = [data[k]['label'] for k in data]
texts_len = [(t, len(t)) for t in texts]
texts_sorted = sorted(texts_len, key=lambda x:x[1], reverse=True)
#print(texts_sorted[0], len(texts[0]))
texts = [t[0] for t in texts_sorted]
print(len(texts))
#print(len(texts[0]))

#text2speech  = init_model()

for spk_id in tqdm(test_speakers):
    synthesize_yourtts2(spk_id, texts, out_dir)