import pandas as pd
import os
import random
from scipy.io.wavfile import read as read_wav
from tqdm import tqdm
from ffmpy import FFmpeg

class MetaData:
    def __init__(self, db, use_nvidia_ljs_split=True):
        self.db = db
        self.ljs_path = '/data2/sungjaecho/data_tts/LJSpeech-1.1'
        self.emovdb_path = '/data2/sungjaecho/data_tts/EmoV-DB/EmoV-DB'
        self.metadata_path = 'metadata'
        self.df = None
        self.use_nvidia_ljs_split = use_nvidia_ljs_split

    def get_df(self, split=None):
        if split is None:
            return self.df
        elif split == 'train':
            return self.df[self.df.split == 'train']
        elif split == 'val':
            return self.df[self.df.split == 'val']
        elif split == 'test':
            return self.df[self.df.split == 'test']

    def load_from_csv(self):
        csv_path = os.path.join(self.metadata_path, '{}.csv'.format(self.db))
        self.df = pd.read_csv(csv_path)


    def load_original_db(self):
        if self.db == "ljspeech":
            csv_path = os.path.join(self.ljs_path, 'metadata.csv')

            if True:
                with open(csv_path, encoding='utf-8') as f:
                    wavpath_rawtext_text = [line.strip().split("|") for line in f]
                self.df = pd.DataFrame(columns=['id','text_raw','text'])
                print("Loading {} data...".format(self.db))
                for i in tqdm(range(len(wavpath_rawtext_text))):
                    # the length of row is 3
                    row = wavpath_rawtext_text[i]
                    self.df.loc[i] = row
            else:
                self.df = pd.read_csv(csv_path, sep='|', header=None, encoding='utf-8', quoting=3)
                self.df = self.df.rename(columns={0:"id", 1:"text_raw", 2:"text"})

        if self.db == "emovdb":
            csv_path = os.path.join(self.emovdb_path, 'emov_db.csv')
            self.df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
            self.df = self.df.rename(columns={
                'sentence_path':'wav_path',
                'script':'text'})
            self.df.speaker = self.df.speaker.apply(self.change_speaker_name)

        print("Loaded from {}".format(csv_path))

    def add_columns(self, split_ratio):
        '''
        split_ratio: dict. e.g., {'train':0.8, 'val':0.1, 'test':0.1}
        '''
        if self.db == "ljspeech":
            self.df['database'] = 'LJ-Speech-1.1'
            self.df['wav_path'] = self.df.id.apply(self.get_wav_path)
            self.df['speaker'] = ['ljs-w'] * len(self.df)
            self.df['emotion'] = ['neutral'] * len(self.df)
            self.df['sex'] = ['w'] * len(self.df)
            self.df['lang'] = 'en'
            self.df['split'] = self.get_split_labels(split_ratio)
            self.df = self.df[['database','split','id','wav_path','text_raw','text','speaker','sex','emotion','lang']]


        if self.db == "emovdb":
            self.df['wav_path'] = self.df.wav_path.apply(self.get_wav_path)
            print(self.df['wav_path'])
            self.df['sex'] = self.df.speaker.apply(self.get_sex)
            self.df['lang'] = 'en'
            self.df['split'] = self.get_split_labels(split_ratio)
            self.df = self.df[['database','split','id','wav_path','duration','text','speaker','sex','emotion','lang']]

    def get_split_labels(self, split_ratio):
        if self.use_nvidia_ljs_split and self.db == "ljspeech":
            split_labels = list()

            split_types = ['train', 'val', 'test']
            db_path = dict()
            db_path['train'] = 'filelists/ljs_audio_text_train_filelist.txt'
            db_path['val']   = 'filelists/ljs_audio_text_val_filelist.txt'
            db_path['test']  = 'filelists/ljs_audio_text_test_filelist.txt'

            db = dict()
            id_sets = dict()
            for split_type in split_types:
                id_sets[split_type] = set()
                db[split_type] = pd.read_csv(db_path[split_type], sep='|', header=None, encoding='utf-8', quoting=3)
                db[split_type] = db[split_type].rename(columns={0:"wav_path", 1:"text"})
                for i, row in db[split_type].iterrows():
                    wav_id = os.path.splitext(os.path.split(row.wav_path)[1])[0]
                    id_sets[split_type].add(wav_id)

            for i, row in self.df.iterrows():
                for split_type in split_types:
                    if row.id in id_sets[split_type]:
                        split_labels.append(split_type)

        else:
            df_len = len(self.df)
            i_val_start = int(df_len * split_ratio['train'])
            i_test_start = int(df_len * (split_ratio['train'] + split_ratio['val']))

            n_train = i_val_start
            n_val = i_test_start - i_val_start
            n_test = df_len - i_test_start

            print("split_ratio", split_ratio)
            print("(n_train, n_val, n_test)", (n_train, n_val, n_test))

            split_labels = (['train'] * n_train) + (['val'] * n_val) + (['test'] * n_test)

            random.seed(3141)
            random.shuffle(split_labels)


        return split_labels

    def get_wav_path(self, col):
        if self.db == "ljspeech":
            wav_path = os.path.join(self.ljs_path, 'wavs', "{}.wav".format(col))

            return wav_path

        if self.db == "emovdb":
            wav_path = os.path.join(self.emovdb_path, col)

            return wav_path


    def get_sex(self, speaker_name):
        if self.db == "emovdb":
            return speaker_name.split('-')[1]


    def change_speaker_name(self, src_speaker_name):
        if self.db == "emovdb":

            if src_speaker_name == 'bea':
                dst_speaker_name = '{}-w-{}'.format(self.db, src_speaker_name)
            elif src_speaker_name == 'jenie':
                dst_speaker_name = '{}-w-{}'.format(self.db, src_speaker_name)
            elif src_speaker_name == 'josh':
                dst_speaker_name = '{}-m-{}'.format(self.db, src_speaker_name)
            elif src_speaker_name == 'sam':
                dst_speaker_name = '{}-m-{}'.format(self.db, src_speaker_name)

        return dst_speaker_name

    def make_new_db(self, split_ratio={'train':0.95, 'val':0.025, 'test':0.025}):
        self.load_original_db()
        self.add_columns(split_ratio)

        df = self.get_df()
        csv_path = os.path.join(self.metadata_path, '{}.csv'.format(self.db))
        df.to_csv(csv_path, index=False)
        print("Saved! {}".format(csv_path))

        splits = ['train', 'val', 'test']
        for split in splits:
            df = self.get_df(split)
            csv_path = os.path.join(self.metadata_path, '{}_{}.csv'.format(self.db, split))
            df.to_csv(csv_path, index=False)
            print("Saved! {}".format(csv_path))

    def print_data_stat(self):
        csv_path = os.path.join(self.metadata_path, '{}.csv'.format(self.db))
        df = pd.read_csv(csv_path)
        print(self.db)

        print(df.groupby(['split']).size().to_frame('size'))
        csv_path = os.path.join(self.metadata_path, '{}_size_groupby_split.csv'.format(self.db))
        df.groupby(['split']).size().to_frame('size').to_csv(csv_path)

        print(df.groupby(['split', 'speaker', 'emotion']).size().to_frame('size'))
        csv_path = os.path.join(self.metadata_path, '{}_size_groupby_split_speaker_emotion.csv'.format(self.db))
        df.groupby(['split', 'speaker', 'emotion']).size().to_frame('size').to_csv(csv_path)


def save_csv_db():
    db = "ljspeech"
    #split_ratio = {'train':0.99, 'val':0.005, 'test':0.005}
    md = MetaData(db, use_nvidia_ljs_split=True)
    md.make_new_db()

    db = "emovdb"
    test_ratio = 0.2
    split_ratio = {'train':(1 - test_ratio)**2, 'val':(1 - test_ratio)*test_ratio, 'test':test_ratio}
    md = MetaData(db)
    md.make_new_db(split_ratio)

def print_data_stat():
    db = "ljspeech"
    md = MetaData(db)
    md.print_data_stat()

    db = "emovdb"
    md = MetaData(db)
    md.print_data_stat()

def debug():
    db = "ljspeech"
    split_ratio = {'train':0.99, 'val':0.005, 'test':0.005}
    md = MetaData(db)
    md.load_original_db()
    md.add_columns(split_ratio)
    row = md.df[md.df.wav_path == "/data2/sungjaecho/data_tts/LJSpeech-1.1/wavs/LJ005-0030.wav"]


    print(row.values.tolist())


def make_one_sample_rate(db, sample_rate=22050):
    md = MetaData(db)
    md.load_from_csv()
    df = md.get_df()

    print("Start to encode wav files of {} to have {} sample rate.".format(db, sample_rate))

    for i, row in tqdm(df.iterrows(), total=len(df)):
        dst_wav = row.wav_path
        src_wav = dst_wav.replace(os.path.join('EmoV-DB', 'EmoV-DB'), os.path.join('EmoV-DB', 'EmoV-DB-copy'))

        change_sample_rate(src_wav, dst_wav, sample_rate)


def change_sample_rate(src_wav, dst_wav, sample_rate=22050):
    frame_rate, _  = read_wav(src_wav)
    #print("Original sample rate:", frame_rate)

    if frame_rate == sample_rate:
        return

    ff = FFmpeg(
        inputs={src_wav: None},
        outputs={dst_wav: "-ar {} -y".format(sample_rate)}
    )

    ff.run()

    #frame_rate, _  = read_wav(dst_wav)
    #print("New sample rate:", frame_rate)






def main():
    save_csv_db()
    print_data_stat()
    #debug()
    #make_one_sample_rate(db = "emovdb", sample_rate=22050)

if __name__ == "__main__":
    main()
