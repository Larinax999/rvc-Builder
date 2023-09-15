@echo off
@title 0

echo Hi
@title 1
python process_dataset.py
@title 2
python extract_f0.py
@title 3
python extract_pitch.py
@title 4
python train_index.py

@title 0
echo
echo Run train_model.py and Done. have fun
timeout 2