import asteroid
import exp_models as M
import torch
from tqdm import tqdm
import numpy as np
from exp_data import sdr_improvement, sdr

from speechbrain.pretrained import EncoderClassifier
import torchaudio
from torch.nn.functional import cosine_similarity
from pystoi import stoi
from pesq import pesq
import os



def load_spk_data(path, ref_path, speakers, audiolm=True):
    di = {s:{"ref":[], "pred":[]} for s in speakers}

    fpaths = []
    
    for root, d, files in os.walk(path):
        for f in files:
            if '.wav' in f:
                fpaths.append(os.path.join(root, f))

    for f in fpaths:
        if audiolm:
            if 'original' not in f:
                spk = int(f.split('/')[-2])
                wav, fs = torchaudio.load(f)
                di[spk]['pred'].append(wav)
        else:
            spk = int(f.split('/')[-2])
            wav, fs = torchaudio.load(f)
            di[spk]['pred'].append(wav)


    fpaths = []
    for root, d, files in os.walk(ref_path):
        for f in files:
            if '.wav' in f:
                fpaths.append(os.path.join(root, f))

    for s in speakers:
        for f in fpaths:
            if ('original' in f) and (f'/{s}/' in f):
                wav, fs = torchaudio.load(f)
                di[s]['ref'].append(wav)
                break

    #print(di[1284])
    return di


def calc_spk_metrics(ref, predicted, clf):
    avgs = 0
    avg_stoi = 0
    avg_pesq = 0

    fs = 16000
    classifier = clf

    ref_embed = classifier.encode_batch(ref)
    embeds = [classifier.encode_batch(i) for i in predicted]
    for i, emb in enumerate(embeds):
        ref_embed =ref_embed.squeeze(0)
        emb = emb.squeeze(0)
        avgs+=cosine_similarity(ref_embed, emb, dim=1)
        
        #pred = predicted[i].squeeze(0).numpy()
        #avg_stoi += stoi(ref, pred, fs)
        #avg_pesq += pesq(ref, pred, fs)
    avgs/=len(predicted)
    #avg_stoi/=len(predicted)
    #avg_pesq/=len(predicted)
    #return avgs, avg_stoi, avg_pesq
    return avgs
    

def eval_input_data(input_path, ref_path, audiolm):
    #test_speakers = [19, 26, 39, 40, 78, 83, 87, 89, 118, 125, 163, 196, 198, 200, 201, 250, 254, 307, 405, 446]
    test_speakers = [4446, 1284, 1089, 4970, 3575, 121]
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device":"cuda"})

    spk_d = load_spk_data(input_path, ref_path, test_speakers, audiolm)

    avg_spk_sim = 0
    avg_stoi = 0
    avg_pesq = 0

    for s in tqdm(spk_d):
        ssim = calc_spk_metrics(spk_d[s]['ref'][0], spk_d[s]['pred'], classifier)
        avg_spk_sim+=ssim
        #avg_stoi+=sstoi
        #avg_pesq+=spesq
    avg_spk_sim/=len(test_speakers)
    #avg_stoi/=len(test_speakers)
    #avg_pesq/=len(test_speakers)
    print(f"Dist: {avg_spk_sim}")





def calc_avg_metrics_val(size, data_subset, partition,split, num_outs, out_path):
    import soundfile as sf
    import os

    def eval_speaker(spk, size, data_subset, partition, split, num_outs=2, out_path='./'):
        checkpoint_path = f'/home/anakuzne/exp/pse/new_specialist_spk{spk}_{size}_1e-06_{partition}/model_best.ckpt'
        #Dec14_12-17-18_convtasnet_large.pt  Dec15_05-08-49_convtasnet_medium.pt  Dec15_14-28-33_convtasnet_small.pt  Dec19_03-47-42_convtasnet_tiny.pt
        if partition=='baseline':
            checkpoint_path = f'/data/common/pse_ckpt/'
            for f in os.listdir(checkpoint_path):
                if size in f:
                    checkpoint_path = f'{checkpoint_path}/{f}'

        if (partition=='60sec') and (size in ['small', 'tiny']):
            checkpoint_path = f'/home/anakuzne/exp/pse/new_specialist_spk{spk}_{size}_1e-06/model_best.ckpt'

        #checkpoint_path = f'/data/common/pse_ckpt/Dec19_03-47-42_convtasnet_tiny.pt'
        net, nparams, config = M.init_model(
            'convtasnet', size)

        # load weights from checkpoint
        net.load_state_dict(
            torch.load(checkpoint_path).get('model_state_dict'),
            strict=True)
        net.cuda()
        net.eval()

        avg_sisdr_imp = 0
        avg_sisdr = 0
        avg_pesq = 0
        avg_stoi = 0
        fs = 16000

        val_batch = torch.load(f"/home/anakuzne/data/{data_subset}/{split}/spk_{spk}.pt")

        x = val_batch['x'].cuda()
        t = val_batch['t'].cuda()
        stats = {"file":[], "sdri":[]}

        for i in range(x.shape[0]):
            y_mini = M.make_2d(net(x[i]))
            ##Saving data for demo
            
            y_out = y_mini.detach().cpu().numpy().reshape(-1, 1)
            x_out = x[i].detach().cpu().numpy().reshape(-1, 1)
            t_out = t[i].detach().cpu().numpy().reshape(-1, 1)
            out_p = f"{out_path}/{spk}/"
            if not os.path.exists(out_p):
                os.makedirs(out_p)
            out_fp = f"{out_p}/{spk}-{size}-{partition}-{i}-enhanced.wav"
            out_xp = f"{out_p}/{spk}-{size}-{partition}-{i}-noisy.wav"
            out_tp = f"{out_p}/{spk}-{size}-{partition}-{i}-clean.wav"
            fs = 16000
            sf.write(out_fp, y_out, fs)
            sf.write(out_xp, x_out, fs)
            sf.write(out_tp, t_out, fs)

            res_sdri = float(sdr_improvement(y_mini, t[i], x[i], 'mean'))
            stats["file"].append(out_fp)
            stats["sdri"].append(res_sdri)
            if i >= num_outs:
                log_fp = f"{out_path}/{spk}-{size}-{partition}-eval_stats.txt"
                with open(log_fp, 'w') as fo:
                    for f, val in zip(stats['file'], stats['sdri']):
                        line = f"{f} {val}"
                        fo.write(line +'\n')
                print(f"Inference stats recorded to {log_fp}")
                break
            '''
            #y_mini= y_mini.squeeze(0).detach().cpu().numpy() 
            t_i = t[i].squeeze(0).detach().cpu().numpy()
            x_i = x[i].squeeze(0).detach().cpu().numpy()
            avg_sisdr_imp+= float(sdr_improvement(y_mini, t[i], x[i], 'mean'))
            #avg_sisdr_imp+= float(sdr_improvement(y_mini, t_i, x_i, 'mean'))
            avg_sisdr+= float(sdr(y_mini, t[i]))
            #avg_sisdr+= float(sdr(y_mini, t_i))
            y_mini= y_mini.squeeze(0).detach().cpu().numpy() 
            avg_stoi += stoi(t_i, y_mini, fs, True)
            avg_pesq += pesq(fs, t_i, y_mini, 'wb')
        avg_sisdr_imp/=x.shape[0]
        avg_sisdr/=x.shape[0]
        avg_pesq/=x.shape[0]
        avg_stoi/=x.shape[0]
        
        return avg_sisdr_imp, avg_sisdr, avg_stoi, avg_pesq
        '''

    #test_speakers = [19, 26, 39, 40, 78, 83, 87, 89, 118, 125, 163, 196, 198, 200, 201, 250, 254, 307, 405, 446] #405, 446
    #test_speakers = [4446, 1284, 1089, 4970, 3575, 121]
    test_speakers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #sdri_all = 0
    #sdr_all = 0
    #avg_stoi = 0
    #avg_pesq = 0

    for s in tqdm(test_speakers):
        #imp, val, stois, pesqs =  
        eval_speaker(spk=s, size=size, data_subset=data_subset,
                    partition=partition, split=split, num_outs=num_outs, out_path=out_path) #spk, size, data_subset, partition, num_outs=2, out_path='./'
        #sdri_all+=imp
        #sdr_all+=val
        #avg_stoi+=stois
        #avg_pesq+=pesqs

    #sdri_all/=len(test_speakers)
    #sdr_all/=len(test_speakers)
    #avg_pesq/=len(test_speakers)
    #avg_stoi/=len(test_speakers)

    #print(f'SDR improvement {sdri_all:.4f}, SDR {sdr_all:.4f}\nSTOI {avg_stoi:.4f} PESQ {avg_pesq:.4f}')

if __name__ == "__main__":
    sizes = ['large', 'tiny']
    #sizes = ['large', 'tiny']
    #sizes = ['large']
    for split in ['test']:
        for size in sizes:
            print(size, split)
            print('---------------')
            calc_avg_metrics_val(size=size, data_subset='YourTTS_60sec_10spk', partition='yourtts_60sec_10spk', split=split, num_outs=5, out_path='/home/anakuzne/projects/anastasia_code/demo')
            #calc_avg_metrics_val(size, 'yourtts_60sec_set2')
            print('---------------')
