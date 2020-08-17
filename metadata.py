import pandas as pd
import os
from os.path import splitext, join
import random
from scipy.io.wavfile import read as read_wav
from tqdm import tqdm
from ffmpy import FFmpeg
import librosa
import sys
import numpy as np
import scipy.stats
from g2p_en import G2p
from sklearn import linear_model


class MetaData:
    def __init__(self, db, use_nvidia_ljs_split=True):
        self.db = db
        self.ljs_path = '/data2/sungjaecho/data_tts/LJSpeech-1.1'
        self.emovdb_path = '/data2/sungjaecho/data_tts/EmoV-DB/EmoV-DB'
        self.bc2013_path = '/data2/sungjaecho/data_tts/BC2013'
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
        '''
        Import data offered by data provider, and create a dataframe `df`
        containing them. `df` should include columns of wav_path, text, and
        everything that can specify a data sample during importing.
        If some columns have a single value across samples, those columns and
        their values will be filled with the single value. For examples,
        if a DB contains the only one speaker, the `speaker` column will be
        added and filled with a single value in the `add_columns` method.
        '''
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
                print("Loaded from {}".format(csv_path))
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

        if self.db == 'bc2013':
            segment_dir = os.path.join(self.bc2013_path, 'segments')
            transcript_dir = os.path.join(self.bc2013_path, 'transcripts')
            book_dirs = [f for f in os.listdir(segment_dir) \
                if not os.path.isfile(os.path.join(segment_dir, f))]

            df_txt_books = list()
            for book_dir in book_dirs:
                txt_dir = os.path.join(transcript_dir, book_dir)
                txts = [f for f in os.listdir(txt_dir) if splitext(f)[1] == '.txt']
                df_txt_chapters = list()
                for txt in txts:
                    chapter = int(splitext(txt)[0])
                    txt_path = os.path.join(txt_dir, txt)
                    df_txt = pd.read_csv(txt_path, sep='|', header=None, encoding='utf-8', quoting=3)
                    df_txt = df_txt.rename(columns={0:"sentence_id", 1:"text"})
                    df_txt['chapter'] = [chapter] * len(df_txt)
                    df_txt_chapters.append(df_txt)
                df_txt_chapters = pd.concat(df_txt_chapters)
                df_txt_chapters['book'] = [book_dir] * len(df_txt_chapters)
                df_txt_books.append(df_txt_chapters)
            df_txt_books = pd.concat(df_txt_books)
            df_txt_books['segmented'] = [False] * len(df_txt_books)
            df_txt_books = df_txt_books.reset_index(drop=True)

            rm_i_list = list()
            rm_wav_paths = list()
            log_file = 'log_bc2013_nonexisting_wavs.txt'
            df_txt_books['wav_path'] = [''] * len(df_txt_books)
            for i, row in df_txt_books.iterrows():
                wav = '{:02d}-{:06d}.wav'.format(row.chapter, row.sentence_id)
                wav_path = os.path.join(segment_dir, row.book, wav)
                if not os.path.exists(wav_path):
                    rm_i_list.append(i)
                    rm_wav_paths.append(wav_path)
                else:
                    df_txt_books.at[i,'wav_path'] = wav_path

            with open(os.path.join(self.metadata_path, log_file), 'w') as f:
                for rm_wav_path in rm_wav_paths:
                    f.write(rm_wav_path)
                    f.write('\n')

            df_txt_books = df_txt_books.drop(rm_i_list)
            df_txt_books = df_txt_books.reset_index(drop=True)

            print("{} wav files do not exist. Those files are logged in {}.".format(len(rm_wav_paths), log_file))
            print("Corresponding scripts are removed.")

            segmented_wav_dir = os.path.join(self.bc2013_path, 'wav')
            segmented_txt_dir = os.path.join(self.bc2013_path, 'txt')
            segmented_wavs = [f for f in os.listdir(segmented_wav_dir) \
                if splitext(f)[1] == '.wav']
            segmented_txts = [f for f in os.listdir(segmented_txt_dir) \
                if splitext(f)[1] == '.txt']

            wav_paths = list()
            texts = list()
            for txt in segmented_txts:
                f_name = splitext(txt)[0]
                wav_path = os.path.join(segmented_wav_dir, f_name) + '.wav'
                if not os.path.exists(wav_path):
                    continue

                txt_path = os.path.join(segmented_txt_dir, txt)
                with open(txt_path, 'r') as f:
                    text = f.read()

                wav_paths.append(wav_path)
                texts.append(text)

            df = pd.DataFrame({
                'text':texts,
                'wav_path':wav_paths,
                'segmented':[True]*len(wav_paths)
            })

            df_txt_books = pd.concat([df_txt_books, df], sort=True)
            df_txt_books = df_txt_books.reset_index(drop=True)

            print("Getting duration for every audio in bc2013 DB.")
            rm_i_list = list()
            rm_wav_paths = list()
            e_list = list()
            df_txt_books['duration'] = [0.0] * len(df_txt_books)
            for i, row in tqdm(df_txt_books.iterrows(), total=len(df_txt_books)):
                try:
                    y, sr = librosa.load(row.wav_path)
                    duration = librosa.get_duration(y, sr)
                    duration = round(duration, 3)
                    df_txt_books.at[i,'duration'] = duration
                except Exception as e:
                    print(e)
                    print(row.wav_path)
                    rm_i_list.append(i)
                    e_list.append(e)
                    rm_wav_paths.append(row.wav_path)

            df_txt_books = df_txt_books.drop(rm_i_list)
            df_txt_books = df_txt_books.reset_index(drop=True)

            with open(os.path.join(self.metadata_path, 'log_get_duration_errors.txt'), 'w') as f:
                for error, wav_path in zip(e_list, rm_wav_paths):
                    f.write(str(error))
                    f.write('\n')
                    f.write(wav_path)
                    f.write('\n')

            self.df = df_txt_books

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

        if self.db == "bc2013":
            self.df['database'] = 'bc2013'
            self.df['speaker'] = ['bc2013-w'] * len(self.df)
            self.df['emotion'] = ['neutral'] * len(self.df)
            self.df['sex'] = ['w'] * len(self.df)
            self.df['lang'] = ['en'] * len(self.df)
            self.df['split'] = self.get_split_labels(split_ratio)
            self.df = self.df[['database','split','wav_path','duration','text','speaker','sex','emotion','lang','segmented','book','chapter','sentence_id']]

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
            if split_ratio['val'] < 1:
                i_val_start = int(df_len * split_ratio['train'])
                i_test_start = int(df_len * (split_ratio['train'] + split_ratio['val']))
            else:
                # This case specifies split ratio using the number of samples
                i_val_start = df_len - split_ratio['val'] - split_ratio['test']
                i_test_start = df_len - split_ratio['test']

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
        # Import data offered by the original source.
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

        if 'duration' in df.columns:
            agg_dict = {'wav_path':'size', 'duration':'sum'}
        else:
            agg_dict = {'wav_path':'size'}

        df_agg = df.groupby(['split']).agg(agg_dict)
        print(df_agg)
        csv_path = os.path.join(self.metadata_path, '{}_size_groupby_split.csv'.format(self.db))
        df_agg.to_csv(csv_path)

        df_agg = df.groupby(['split', 'speaker', 'emotion']).agg(agg_dict)
        print(df_agg)
        csv_path = os.path.join(self.metadata_path, '{}_size_groupby_split_speaker_emotion.csv'.format(self.db))
        df_agg.to_csv(csv_path)


def save_csv_db(db):
    if db == "ljspeech":
        #split_ratio = {'train':0.99, 'val':0.005, 'test':0.005}
        md = MetaData(db, use_nvidia_ljs_split=True)
        md.make_new_db()

    if db == "emovdb":
        test_ratio = 0.2
        split_ratio = {'train':(1 - test_ratio)**2, 'val':(1 - test_ratio)*test_ratio, 'test':test_ratio}
        md = MetaData(db)
        md.make_new_db(split_ratio)

    if db == "bc2013":
        val_size = 100
        test_size = 400
        split_ratio = {'val':val_size, 'test':test_size}
        md = MetaData(db)
        md.make_new_db(split_ratio)


def print_data_stat():
    db = "ljspeech"
    md = MetaData(db)
    md.print_data_stat()

    db = "emovdb"
    md = MetaData(db)
    md.print_data_stat()

    db = "bc2013"
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

        src_wav = row.wav_path + "_old"
        dst_wav = row.wav_path
        os.rename(
            row.wav_path,
            src_wav
        )
        #src_wav = dst_wav.replace(os.path.join('EmoV-DB', 'EmoV-DB'), os.path.join('EmoV-DB', 'EmoV-DB-copy'))

        change_sample_rate(src_wav, dst_wav, sample_rate)

        if os.path.exists(src_wav) and (not os.path.exists(dst_wav)):
            os.rename(
                src_wav,
                dst_wav
            )
        if os.path.exists(src_wav) and os.path.exists(dst_wav):
            os.remove(src_wav)

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

def convert_sec(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return hours, minutes, seconds




def main():
    save_csv_db()
    print_data_stat()
    #debug()
    #make_one_sample_rate(db = "emovdb", sample_rate=22050)

if __name__ == "__main__":
    main()
