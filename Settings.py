import numpy as np,ffmpeg,os

name="Name"
sr=48000 # sample rate
thread=8 # thread for process
epoch=250 # target epoch
every_epoch=25 # save every epoch
batch_size=20 # GB

# It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit. / num_workers=8 -> num_workers=4
num_workers=8

# https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

inp_root = f"{os.getcwd()}\\datasets" # datasets folder
exp_dir = f"{os.getcwd()}\\logs\\{name}"
PATH=os.getcwd()

## How to Run
# - Setup
# 1. process_dataset.py
# 2. extract_f0.py
# 3. extract_pitch.py
# 4. train_index.py
# - Run
# 1. train_model.py

## needed
# https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main

def load_audio(file): # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
    try:
        out, _ = ffmpeg.input(file,threads=0).output("-",format="f32le",acodec="pcm_f32le",ac=1,ar=sr).run(cmd=["ffmpeg","-nostdin"],capture_stdout=True,capture_stderr=True)
    except Exception as e:raise RuntimeError(f"Failed to load audio: {e}")
    return np.frombuffer(out,np.float32).flatten()