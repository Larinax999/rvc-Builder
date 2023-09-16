# python train_nsf_sim_cache_sid_load_pretrain.py -e DeletedUser -sr 48k -f0 1 -bs 12 -te 10000 -se 50 -pg pretrained_v2/f0G48k.pth -pd pretrained_v2/f0D48k.pth -l 1 -c 0 -sw 1 -v v2 -li 9
import os,torch,datetime,json,random
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from time import time as ttime
from cors.data_utils import (
    TextAudioLoaderMultiNSFsid,
    # TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    # TextAudioCollate,
    DistributedBucketSampler,
)
from types import SimpleNamespace
from cors.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from cors.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from cors.process_ckpt import savee
from cors.infer_pack import commons
from cors.infer_pack.models import (
    SynthesizerTrnMs768NSFsid as RVC_Model_f0,
    MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator
)
from Settings import epoch as total_epoch,exp_dir,every_epoch,batch_size,name,PATH,num_workers
import torch.distributed as dist
import cors.utils

# hps = cors.utils.get_hparams()
global_step=0
writer=None
data_iterator=[]
configs=json.load(open("./bin/48k_v2.json"), object_hook=lambda d: SimpleNamespace(**d))
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

class EpochRecorder:
    def __init__(self):self.lt = ttime()
    def record(self,e):
        nt=ttime()
        et=nt-self.lt
        self.lt=nt
        return f"| {int((e*et)//60)} Minutes | {int(et)}/sec per Epoch"

def Time():return datetime.datetime.now().strftime('%H:%M:%S')

def train_and_evaluate(epoch, nets, optims, scaler, train_loader):
    global global_step,writer,data_iterator
    net_g, net_d = nets
    optim_g, optim_d = optims

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    # TODO: cache train_loader in memory
    if len(data_iterator) == 0:
        # Make new cache
        for _, info in enumerate(train_loader):
            # Unpack
            phone,phone_lengths,pitch,pitchf,spec,spec_lengths,wave,_,sid = info
            # Load on CUDA
            phone = phone.cuda(0, non_blocking=True)
            phone_lengths = phone_lengths.cuda(0, non_blocking=True)
            pitch = pitch.cuda(0, non_blocking=True)
            pitchf = pitchf.cuda(0, non_blocking=True)
            sid = sid.cuda(0, non_blocking=True)
            spec = spec.cuda(0, non_blocking=True)
            spec_lengths = spec_lengths.cuda(0, non_blocking=True)
            wave = wave.cuda(0, non_blocking=True)
            # Cache on list
            data_iterator.append((phone,phone_lengths,pitch,pitchf,spec,spec_lengths,wave,sid))
    else:
        # Load shuffled cache
        random.shuffle(data_iterator)

    # Run steps
    epoch_recorder = EpochRecorder()
    for phone,phone_lengths,pitch,pitchf,spec,spec_lengths,wave,sid in data_iterator:
        # Calculate
        with autocast(enabled=True):
            y_hat,ids_slice,x_mask,z_mask,(z, z_p, m_p, logs_p, m_q, logs_q) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(spec,configs.data.filter_length,configs.data.n_mel_channels,configs.data.sampling_rate,configs.data.mel_fmin,configs.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, configs.train.segment_size//configs.data.hop_length)
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(y_hat.float().squeeze(1),configs.data.filter_length,configs.data.n_mel_channels,configs.data.sampling_rate,configs.data.hop_length,configs.data.win_length,configs.data.mel_fmin,configs.data.mel_fmax).half()
            wave = commons.slice_segments(wave,ids_slice*configs.data.hop_length,configs.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=True):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * configs.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * configs.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        # Buggy
        # optim_g.step()
        # optim_d.step()

        # Amor For Tensorboard display
        if global_step % 7 == 0:
            if loss_mel>75:loss_mel=75
            if loss_kl>9:loss_kl=9
            # print(f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}")
            scalar_dict={
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc,
                "learning_rate": optim_g.param_groups[0]["lr"],
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/kl": loss_kl
            }
            scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
            scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
            scalar_dict.update({f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)})
            cors.utils.summarize(
                writer=writer,
                global_step=global_step,
                images={
                    "slice/mel_org": cors.utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": cors.utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": cors.utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                },
                scalars=scalar_dict
            )
        
        global_step+=1

    # /Run steps
    print(f"[{Time()}] Epoch {epoch}/{total_epoch} {epoch_recorder.record(total_epoch-epoch)}")
    iss=open("stop.txt","r").read()=="1"
    if epoch % every_epoch == 0 or iss:
        cors.utils.save_checkpoint(net_g,optim_g,configs.train.learning_rate,epoch,os.path.join(exp_dir, "G_6969.pth"))
        cors.utils.save_checkpoint(net_d,optim_d,configs.train.learning_rate,epoch,os.path.join(exp_dir, "D_6969.pth"))
        ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
        print(f"[{Time()}] Saving CKPT {name}_e{epoch}",savee(ckpt,"48k",True,f'{name}_e{epoch}_s{global_step}',epoch,"v2",configs))
    if epoch >= total_epoch or iss:
        print("Training is done. The program is closed.")
        open("stop.txt","w").write("0")
        os._exit(0)

def main():
    global global_step
    dist.init_process_group(backend="gloo",init_method="env://",world_size=1,rank=0)
    torch.manual_seed(1234)
    torch.cuda.set_device(0)

    train_dataset = TextAudioLoaderMultiNSFsid(f"{exp_dir}/filelist.txt",configs.data)
    train_sampler = DistributedBucketSampler(train_dataset,batch_size,[100, 200, 300, 400, 500, 600, 700, 800, 900],num_replicas=1,rank=0,shuffle=True) # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400],  # 16s
    train_loader = DataLoader(train_dataset,num_workers=num_workers,shuffle=False,pin_memory=True,collate_fn=TextAudioCollateMultiNSFsid(),batch_sampler=train_sampler,persistent_workers=True,prefetch_factor=8)

    net_g = RVC_Model_f0(configs.data.filter_length//2 + 1,configs.train.segment_size//configs.data.hop_length,**configs.model.__dict__,is_half=True,sr=configs.data.sampling_rate).cuda(0)
    optim_g = torch.optim.AdamW(net_g.parameters(),configs.train.learning_rate,betas=configs.train.betas,eps=configs.train.eps)

    net_d = MultiPeriodDiscriminator(False).cuda(0)
    optim_d = torch.optim.AdamW(net_d.parameters(),configs.train.learning_rate,betas=configs.train.betas,eps=configs.train.eps)

    net_g = DDP(net_g,device_ids=[0]) # , find_unused_parameters=True
    net_d = DDP(net_d,device_ids=[0])
    

    _A, _, _, epoch_str = cors.utils.load_checkpoint(os.path.join(exp_dir, "G_6969.pth"), net_g, optim_g)
    if _A==False: # pretrain
        print(net_g.module.load_state_dict(torch.load(f"{PATH}\\bin\\f0G48k.pth",map_location="cpu")["model"]))
        print(net_d.module.load_state_dict(torch.load(f"{PATH}\\bin\\f0D48k.pth",map_location="cpu")["model"]))
    else:
        cors.utils.load_checkpoint(os.path.join(exp_dir, "D_6969.pth"), net_d, optim_d)
        global_step = (epoch_str-1)*len(train_loader)
        

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g,gamma=configs.train.lr_decay,last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d,gamma=configs.train.lr_decay,last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=True)
    for epoch in range(epoch_str,20001):
        train_and_evaluate(epoch,[net_g, net_d],[optim_g, optim_d],scaler,train_loader) # writer
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler_g.step()
        scheduler_d.step()

if __name__ == "__main__":
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(20000, 55555))
    writer=SummaryWriter(log_dir=exp_dir)

    opt = []
    for n in set([n.split(".")[0] for n in os.listdir(f"{exp_dir}\\0_gt_wavs")]) & set([n.split(".")[0] for n in os.listdir(f"{exp_dir}\\3_feature")]) & set([n.split(".")[0] for n in os.listdir(f"{exp_dir}\\2_f0")]) & set([n.split(".")[0] for n in os.listdir(f"{exp_dir}\\2_f0nsf")]):
        opt.append(f"{exp_dir}\\0_gt_wavs\\{n}.wav|{exp_dir}\\3_feature\\{n}.npy|{exp_dir}\\2_f0\\{n}.wav.npy|{exp_dir}\\2_f0nsf\\{n}.wav.npy|0")
    for _ in range(2):opt.append(f"{PATH}\\bin\\mute\\0mute48k.wav|{PATH}\\bin\\mute\\3mute.npy|{PATH}\\bin\\mute\\2amute.npy|{PATH}\\bin\\mute\\2bmute.npy|0")
    random.shuffle(opt)
    open(f"{exp_dir}/filelist.txt", "w+").write("\n".join(opt))

    torch.multiprocessing.set_start_method("spawn")
    main()