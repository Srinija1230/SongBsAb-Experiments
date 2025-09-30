import os

storage_dir = os.path.abspath('./storage')
os.makedirs(storage_dir, exist_ok=True)
#open_singer_root_ori = os.path.join(storage_dir, 'OpenSinger')
open_singer_root_ori = os.path.join('/data/srinija/OpenSinger')
open_singer_root = os.path.abspath(os.path.join('/data/srinija/OpenSinger'))
assert os.path.exists(open_singer_root)
if not os.path.exists(open_singer_root_ori):
    os.system(f'ln -s {open_singer_root} {open_singer_root_ori}')
data_root = 'data_svc-all_singers-all_vocies'
os.makedirs(storage_dir + '/' + data_root, exist_ok=True)
if not os.path.exists(data_root):
    os.system('ln -s ' + storage_dir + '/' + data_root + ' ./')


class Singer():

    def __init__(self, name):
        self.name = name
        self.gender, self.idx = self.name.split('_')
        assert self.gender in ['F', 'M']
        self.gender_flag = 'ManRaw' if self.gender == 'M' else 'WomanRaw'
        self.tuned = False
        
    def tune(self, force_tune=False):
        
        data_dir = os.path.join(data_root, self.name)

        wavs_raw_dir = os.path.join(data_dir, 'waves-raw')
        print(data_dir, wavs_raw_dir)
        os.makedirs(wavs_raw_dir, exist_ok=True)
        for x in os.listdir(os.path.join(open_singer_root_ori, self.gender_flag)):
            if x.split('_')[0] != self.idx:
                continue
            for name in os.listdir(os.path.join(open_singer_root_ori, self.gender_flag, x)):
                if 'wav' not in name:
                    continue
                src_path = os.path.join(open_singer_root_ori, self.gender_flag, x, name)
                des_path = os.path.join(wavs_raw_dir, name)
                if os.path.exists(des_path):
                    continue
                os.symlink(src_path, des_path)
        
        wavs_16k_dir = os.path.join(data_dir, 'waves')
        os.makedirs(wavs_16k_dir, exist_ok=True)
        command = 'python svc_preprocess_wav.py --out_dir {} --sr 16000 --in_dir {}'.format(wavs_16k_dir, wavs_raw_dir)
        os.system(command)
        
singers_str = []
for i in range(28):
    singers_str.append('M_{}'.format(i))
for i in range(48):
    singers_str.append('F_{}'.format(i))

for singer_str in singers_str:
    singer = Singer(singer_str)
    singer.tune()

# select 10 voices per singer
src_root = os.path.join(storage_dir, data_root)
des_root = os.path.join(storage_dir, 'data_svc-all_singers-10_voices')
spks_keys = singers_str
for spk_key in spks_keys:
    if spk_key not in os.listdir(src_root):
        continue
    src_path = os.path.join(src_root, spk_key)
    des_path = os.path.join(des_root, spk_key)
    os.makedirs(des_path, exist_ok=True)
    for name in os.listdir(src_path):
        if name == 'waves':
            os.makedirs(os.path.join(des_path, name), exist_ok=True)
            for wav_name in sorted(os.listdir(os.path.join(src_path, name)))[:10]:
                s = os.path.join(src_path, name, wav_name)
                d = os.path.join(des_path, name, wav_name)
                if os.path.exists(d):
                    continue
                command = 'ln -s {} {}'.format(s, d)
                print(command)
                os.system(command)