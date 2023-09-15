import numpy as np, os,traceback,logging,multiprocessing
from cors.rmvpe import RMVPE
from Settings import thread,exp_dir,load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

class FeatureInput(object):
    def __init__(self, hop_size=160):
        self.model_rmvpe = RMVPE("./bin/rmvpe.pt")
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(),f0_coarse.min())
        return f0_coarse

    def go(self, paths):
        if len(paths) == 0:
            print("no-f0-todo")
            return
        for i, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if os.path.exists(f"{opt_path1}.npy") == True and os.path.exists(f"{opt_path2}.npy") == True:continue
                featur_pit = self.model_rmvpe.infer_from_audio(load_audio(inp_path),thred=0.03)
                np.save(opt_path2,featur_pit,allow_pickle=False)  # nsf
                np.save(opt_path1,self.coarse_f0(featur_pit),allow_pickle=False)  # ori
                print(f"f0ok-{i}-{opt_path1}")
            except:
                print(f"f0fail-{i}-{inp_path}-{traceback.format_exc()}")


if __name__ == "__main__":
    paths = []
    featureInput = FeatureInput()
    inp_root = f"{exp_dir}/0_gt_wavs/"
    opt_root1 = f"{exp_dir}/2_f0/"
    opt_root2 = f"{exp_dir}/2_f0nsf/"
    os.makedirs(opt_root1,exist_ok=True)
    os.makedirs(opt_root2,exist_ok=True)

    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}{name}"
        if "spec" in inp_path:continue
        paths.append([inp_path,f"{opt_root1}/{name}",f"{opt_root2}/{name}"])

    ps = []
    for i in range(thread):
        p=multiprocessing.Process(target=featureInput.go,args=(paths[i::thread],))
        ps.append(p)
        p.start()
    for t in ps:t.join()