import numpy as np, os, traceback,torch,torch.nn.functional as F,soundfile as sf
from fairseq import checkpoint_utils
from Settings import exp_dir

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "cuda"
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = f"{exp_dir}/3_feature"
model=None
os.makedirs(outPath, exist_ok=True)

# wave must be 16k, hop_size=320
def readwave(wav_path,normalize=False):
    wav, sr = sf.read(wav_path)
    # assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats

def main():
    for i, file in enumerate(sorted(list(os.listdir(wavPath)))):
        try:
            if file.endswith(".wav"):
                out_path = f"{outPath}/{file.replace('wav','npy')}"
                if os.path.exists(out_path):continue
                feats = readwave(f"{wavPath}/{file}", normalize=saved_cfg.task.normalize)
                with torch.no_grad():
                    logits = model.extract_features(source=feats.half().to(device),padding_mask=torch.BoolTensor(feats.shape).fill_(False).to(device),output_layer=12)
                    feats = logits[0]
                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    print("%s-contains nan" % file)
                print(f"now",i,file,feats.shape)
        except:
            print(traceback.format_exc())

if __name__ == "__main__":
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["./bin/hubert_base.pt"],suffix="")
    model = models[0].to(device).half()
    model.eval()
    main()
