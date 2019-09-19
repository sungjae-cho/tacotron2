# Tacotron 2 (without wavenet)

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf).

This implementation includes **distributed** and **automatic mixed precision** support
and uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

Visit our [website] for audio samples using our published [Tacotron 2] and
[WaveGlow] models.

![Alignment, Predicted Mel Spectrogram, Target Mel Spectrogram](tensorboard.png)


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Download and extract the [LJ Speech dataset ver. 1.1](https://keithito.com/LJ-Speech-Dataset/)
    - The extracted directory `LJSpeech-1.1` would have the `wavs` directory that contains speech wav files. The path of the `wavs` directory will be used in Step 5.
2. Clone this repo: `git clone https://github.com/sungjae-cho/tacotron2.git`
3. CD into this repo: `cd tacotron2`
4. Initialize submodule: `git submodule init; git submodule update`
5. Update .wav paths: `sed -i -- 's,DUMMY,<ljs_dataset_folder/wavs>,g' filelists/*.txt`
    - `<ljs_dataset_folder/wavs>` is the the `wavs` directory found in Step 1.
    - Alternatively, set `load_mel_from_disk=True` in `hparams.py` and update mel-spectrogram paths
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image
    - Install python requirements: `pip3 install -r requirements.txt`
9. Login `wandb login` and give API keys given in the [WandB website](https://www.wandb.com) user setting.

## Training (Single GPU)
If you want to use the second GPU device with device number 1, then enter `export CUDA_VISIBLE_DEVICES=1`. If there are 4 GPU devices, then you can use 0, 1, 2, and 3 as a device number.
1. `python3 train.py --output_directory=outdir --log_directory=logdir --run_name=<str_run_name>`
    - If you get an error message saying, for example, `RuntimeError: CUDA out of memory. Tried to allocate 106.38 MiB (GPU 0; 11.91 GiB total capacity; 10.26 GiB already allocated; 53.06 MiB free; 56.48 MiB cached)`, then reduce the batch size. You can reduce the size on the command in this way: `python3 train.py --output_directory=outdir --log_directory=logdir --run_name=<str_run_name> --hparams=batch_size=32`. Or, change `batch_size` in `hparams.py`.
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

===== What I have executed to here. =====

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. Download our published [Tacotron 2] model
2. Download our published [WaveGlow] model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation.


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
