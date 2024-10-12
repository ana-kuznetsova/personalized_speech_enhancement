import asteroid
import exp_models as M
import torch
from tqdm import tqdm
import numpy as np
import os


#checkpoint_path = 'Dec14_12-17-18_convtasnet_large.pt'
#loss_sisdr = asteroid.losses.sdr.SingleSrcNegSDR('sisdr')
#checkpoint_path = '/home/anakuzne/exp/pse/specialist_spk19_large_0.0001/model_best.ckpt'
'''
# instantiates a new untrained model
net, nparams, config = M.init_model(
    'convtasnet', 'large')

# load weights from checkpoint
net.load_state_dict(
    torch.load(checkpoint_path).get('model_state_dict'),
    strict=True)
net.cuda()

# feedforward pass
x = ... # torch.FloatTensor (batch, length)
t = ... # torch.FloatTensor (batch, length)
net.train()
y = M.make_2d(net(x))
loss = loss_sisdr(y, t)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
'''
# to run the personalization test set (specifically with 200)
def clalc_avg_metrics(size, partition):
    test_speakers = [19, 26, 39, 40, 78, 83, 87, 89, 118, 125, 163, 196, 198, 200, 201, 250, 254, 307, 405, 446] #
    avg_sdri = []
    avg_sdr = []
    avg_pesq = []
    avg_stoi = []
    #sample 100 audios for each

    for s in tqdm(test_speakers):
        if partition=='baseline':
            checkpoint_path = f'/data/common/pse_ckpt/'
            for f in os.listdir(checkpoint_path):
                if size in f:
                    checkpoint_path = f'{checkpoint_path}/{f}'

        elif partition=='yourtts_60sec_set1':
            checkpoint_path = f'/home/anakuzne/exp/pse/new_specialist_spk{s}_{size}_1e-06_{partition}/model_best.ckpt'
        elif (partition=='60sec') and (size in ['small', 'tiny']):
            checkpoint_path = f'/home/anakuzne/exp/pse/specialist_spk{s}_{size}_1e-05/model_best.ckpt'
        else:
            checkpoint_path = f'/home/anakuzne/exp/pse/new_specialist_spk{s}_{size}_1e-06_{partition}/model_best.ckpt'
        
        #checkpoint_path = f'/data/common/pse_ckpt/Dec19_03-47-42_convtasnet_tiny.pt'
        #Dec14_12-17-18_convtasnet_large.pt  Dec15_05-08-49_convtasnet_medium.pt  Dec15_14-28-33_convtasnet_small.pt  Dec19_03-47-42_convtasnet_tiny.pt
        net, nparams, config = M.init_model(
            'convtasnet', size)

        # load weights from checkpoint
        net.load_state_dict(
            torch.load(checkpoint_path).get('model_state_dict'),
            strict=True)
        net.cuda()  
        results = M.test_denoiser_with_speaker(net, speaker_id=s)
        avg_sdri.append(results['sdri'])
        avg_sdr.append(results['sdr_avg'])
        avg_pesq.append(results['pesq_avg'])
        avg_stoi.append(results['stoi_avg'])
    print(f"SDR Improvement {np.array(avg_sdri).mean():.4f}, SDR {np.array(avg_sdr).mean():.4f}\nPESQ {np.array(avg_pesq).mean():.4f} STOI {np.array(avg_stoi).mean():.4f}")

if __name__ == "__main__":
    for s in ['small']:
        print(s)
        print('--------------------')
        clalc_avg_metrics(s, 'yourtts_120sec_set1')
        print('--------------------')
