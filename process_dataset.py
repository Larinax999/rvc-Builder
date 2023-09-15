import numpy as np, os,traceback,librosa,threading
from scipy import signal
from cors.slicer2 import Slicer
from scipy.io import wavfile
from Settings import sr,thread,inp_root,exp_dir,load_audio

class PreProcess:
    def __init__(self):
        self.slicer = Slicer(sr=sr,threshold=-42,min_length=1500,min_interval=400,hop_size=15,max_sil_kept=500)
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)
        self.per = 3.7
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.gt_wavs_dir=f"{exp_dir}/0_gt_wavs"
        self.wavs16k_dir=f"{exp_dir}/1_16k_wavs"
        os.makedirs(self.gt_wavs_dir,exist_ok=True)
        os.makedirs(self.wavs16k_dir,exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio/tmp_max*(self.max*self.alpha))+(1-self.alpha)*tmp_audio
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav",sr,tmp_audio.astype(np.float32))
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav",16000,librosa.resample(tmp_audio,orig_sr=sr,target_sr=16000).astype(np.float32)) # , res_type="soxr_vhq"

    def pipeline(self,files):
        for path, idx0 in files:
            try:
                idx1 = 0
                for audio in self.slicer.slice(signal.lfilter(self.bh,self.ah,load_audio(path))):
                    i=0
                    while True:
                        start=int(sr*(self.per-self.overlap)*i)
                        i+=1
                        if len(audio[start:]) > self.tail * sr:
                            tmp_audio = audio[start:start+int(self.per * sr)]
                            self.norm_write(tmp_audio,idx0,idx1)
                            idx1 += 1
                        else:
                            tmp_audio = audio[start:]
                            idx1 += 1
                            break
                    self.norm_write(tmp_audio, idx0, idx1)
                print(f"OK | {path}")
            except:
                print(f"ERROR | {path} | {traceback.format_exc()}")

    def Do(self):
        try:
            infos = [(f"{inp_root}\\{name}",idx) for idx, name in enumerate(sorted(list(os.listdir(inp_root))))]
            ps = []
            for i in range(thread):
                p = threading.Thread(target=self.pipeline, args=(infos[i::thread],))
                ps.append(p)
                p.start()
            for t in ps:t.join()
        except:print(f"Fail. {traceback.format_exc()}")

if __name__ == "__main__":
    os.makedirs(exp_dir,exist_ok=True)
    PreProcess().Do()