
# Source Code for SongBsAb published at NDSS 2025
## Instructions
1.  Download OpenSinger dataset from [OpenSinger](https://github.com/Multi-Singer/Multi-Singer.github.io) and then 
```shell
tar -zxvf OpenSinger.tar.gz
```

2. Change the working directory. 
```shell
cd Lora-SVC
```

3. Build the environment by following the instructions at ```setup.sh```

4. Process the dataset. 
```shell
python preprocess_dataset_opensinger.py
```

5. Obtain the undefended output singing voices.  
```shell
python few_shot_svc.py -start 0 -end 76 --type all_singers-10_voices --source_type '' -dataset opensinger -w_start 0 -w_end 10000
```

6. Generate adversarial examples for input target singing voices.  
```shell
python protect_target.py --start 0 --end 76 lora_LSTM whisper
```

7. Generate adversarial examples for input source singing voices.
```shell
python protect_source.py --start 0 --end 3000 lora_LSTM whisper
```

8. Compute the target speaker similarity of undefended output singing voices. 
```shell
python cal_speaker_sim_undefended.py --start 0 --end 76 --attack_flag all_singers-10_voices --attack_flag_2 all_singers-10_voices resnet18_veri
```

9. Process the ground-truth lyric for computing lyric word error rate. 
```shell
cd wenet/examples/wenetspeech/s0
sh test_WER_CER_args.sh None in OpenSinger
```

10. Compute the lyric word error rate of undefended output singing voices. 
```shell
cd wenet/examples/wenetspeech/s0
sh test_WER_CER_args.sh all_singers-10_voices out inference-all_singers-10_voices
```

11. Obtain the defended output singing voices (dual prevention).  
```shell
python few_shot_svc.py -start 0 -end 76 -type adver-backtrack-lr=0_0002 -source_type backtrack-lr=0_0002 -dataset opensinger -w_start 0 -w_end 10000
```

12. Compute the target speaker similarity of defended output singing voices. 
```shell
python cal_speaker_sim_defended.py --start 0 --end 76 --attack_flag adver-backtrack-lr=0_0002 --attack_flag_2 backtrack-lr=0_0002-adver-backtrack-lr=0_0002 resnet18_veri
```

13. Compute the lyric word error rate of defended output singing voices. 
```shell
cd wenet/examples/wenetspeech/s0
sh test_WER_CER_args.sh backtrack-lr=0_0002-adver-backtrack-lr=0_0002 out inference-backtrack-lr=0_0002-adver-backtrack-lr=0_0002
```

The step 5, 6, 7, 11 may take quite long. You can reduce the number of (target_singer, source_song) pairs in ``select-target_speakers-source_speeches-des.yaml``. 

### TODO List

- [ ] High/Low Hierarchy Multi-Target Loss
- [ ] Frame-level interaction reduction-based (FL-IR) loss
- [ ] Encoder ensemble

- [ ] The English dataset NUS-48E
- [ ] More SVC models

- [ ] Metrics: success reduction rate, SNR, PESQ
- [ ] Robustness evaluation

If you find our work or code useful, please cite our paper as follows:
```bib
@inproceedings{SongBsAb,
  author       = {Guangke Chen and
                  Yedi Zhang and
                  Fu Song and
                  Ting Wang and
                  Xiaoning Du and
                  Yang Liu},
  title        = {SongBsAb: {A} Dual Prevention Approach against Singing Voice Conversion
                  based Illegal Song Covers},
  booktitle    = {32nd Annual Network and Distributed System Security Symposium, {NDSS}
                  2025, San Diego, California, USA, February 24-28, 2025},
  year         = {2025},
}
```