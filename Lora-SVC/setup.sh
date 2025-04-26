
# environment
conda create -n lora-svc python=3.8
conda activate lora-svc
pip install -r requirements.txt
pip install pydub

# models for Lora-SVC (some drawn from https://github.com/MaxMax2016/lora-svc-16k)
1. download the Timbre Encoder: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), put best_model.pth.tar into speaker_pretrain/

2. download [whisper model multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt). 
Make sure to download medium.pt, put it into whisper_pretrain/

3. Download the pre-training model [maxgan_pretrain_16K_5L.pth](https://github.com/PlayVoice/lora-svc/releases/tag/v0.5.5) and put it in the storage/model_pretrain folder.

# wenet; using for calculating WER
1. ```shell
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git reset --hard dfa7b5ca2204a385607b3a8946f24d5c8f2a6ebc ## 
pip install -r requirements.txt
pre-commit install  # for clean and tidy code
cd ..
ln -s $(pwd)/wenetspeech_WER_files/* wenet/examples/wenetspeech/s0/
```
2. Download the Wenetspeech Runtime model at [Wenetspeech-Runtime Model-Conformer](https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md) 
and then untar it into Lora-SVC/20220506_u2pp_conformer_exp

# resnet18-veri for calculating identity similarity
1. Download ckpt files from [auto_speech.zip](https://drive.google.com/file/d/1Tud5eZL1YWVKXRbIOBlpnsXVjgUuJQ5C/view)
2. ```shell
unzip -d pre-trained-models/ auto_speech.zip 
```