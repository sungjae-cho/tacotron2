import pandas as pd
import os
import random

class MetaData:
    def __init__(self, db):
        self.db = db
        self.ljs_path = '/data2/sungjaecho/data_tts/LJSpeech-1.1'
        self.emovdb_path = '/data2/sungjaecho/data_tts/EmoV-DB/EmoV-DB'
        self.metadata_path = 'metadata'
        self.df = None

    def get_df(self, split=None):
        if split is None:
            return self.df
        elif split == 'train':
            return self.df[self.df.split == 'train']
        elif split == 'val':
            return self.df[self.df.split == 'val']
        elif split == 'test':
            return self.df[self.df.split == 'test']

    def load_original_db(self):
        if self.db == "ljspeech":
            csv_path = os.path.join(self.ljs_path, 'metadata.csv')
            self.df = pd.read_csv(csv_path, sep='|', header=None, encoding='utf-8')
            self.df = self.df.rename(columns={0:"id", 1:"text_raw", 2:"text"})

        if self.db == "emovdb":
            csv_path = os.path.join(self.emovdb_path, 'emov_db.csv')
            self.df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
            self.df = self.df.rename(columns={
                'sentence_path':'wav_path',
                'transcription':'text'})
            self.df.speaker = self.df.speaker.apply(self.change_speaker_name)

        print("Loaded from {}".format(csv_path))

    def add_columns(self, split_ratio):
        '''
        split_ratio: dict. e.g., {'train':0.8, 'val':0.1, 'test':0.1}
        '''
        if self.db == "ljspeech":
            self.df['wav_path'] = self.df.id.apply(self.get_wav_path)
            self.df['speaker'] = ['ljs-w'] * len(self.df)
            self.df['emotion'] = ['neutral'] * len(self.df)
            self.df['sex'] = ['w'] * len(self.df)
            self.df['split'] = self.get_split_labels(split_ratio)

        if self.db == "emovdb":
            self.df['sex'] = self.df.speaker.apply(self.get_sex)
            self.df['split'] = self.get_split_labels(split_ratio)

    def get_split_labels(self, split_ratio):
        df_len = len(self.df)
        i_val_start = int(df_len * split_ratio['train'])
        i_test_start = int(df_len * (split_ratio['train'] + split_ratio['val']))

        n_train = i_val_start
        n_val = i_test_start - i_val_start
        n_test = df_len - i_test_start

        split_labels = (['train'] * n_train) + (['val'] * n_val) + (['test'] * n_test)

        random.seed(3141)
        random.shuffle(split_labels)

        return split_labels

    def get_wav_path(self, id, speaker=None, emotion=None):
        if self.db == "ljspeech":
            wav_path = os.path.join(self.ljs_path, "{}.wav".format(id))

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

    def make_new_db(self, split_ratio):
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




def save_csv_db():
    db = "ljspeech"
    split_ratio = {'train':0.9, 'val':0.05, 'test':0.05}
    md = MetaData(db)
    md.make_new_db(split_ratio)

    db = "emovdb"
    split_ratio = {'train':0.95, 'val':0.01, 'test':0.04}
    md = MetaData(db)
    md.make_new_db(split_ratio)


def main():
    save_csv_db()

if __name__ == "__main__":
    main()
