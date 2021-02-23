import librosa, os, argparse
from ffmpy import FFmpeg
from tqdm import tqdm

def make_one_sample_rate_in_dir(rDirectory, sample_rate):
    wav_paths = list()
    for root, dirnames, filenames in os.walk(rDirectory):
            for filename in filenames:
                if filename[-4:] == '.wav':
                    rf = os.path.join(root, filename)
                    wav_paths.append(rf)

    for wav_path in tqdm(wav_paths, total=len(wav_paths)):
        src_wav = wav_path + "_old"
        dst_wav = wav_path
        os.rename(
            wav_path,
            src_wav
        )

        change_sample_rate(src_wav, dst_wav, sample_rate)

        if os.path.exists(src_wav) and (not os.path.exists(dst_wav)):
            os.rename(
                src_wav,
                dst_wav
            )
        if os.path.exists(src_wav) and os.path.exists(dst_wav):
            os.remove(src_wav)

    failed_file_paths = list()
    for root, dirnames, filenames in os.walk(rDirectory):
            for filename in filenames:
                if filename[-4:] == '_old':
                    rf = os.path.join(root, filename)
                    failed_file_paths.append(rf)
    print("# Failed file paths:", len(failed_file_paths))
    print("Failed file paths:", failed_file_paths)


def change_sample_rate(src_wav, dst_wav, sample_rate=22050):
    '''
    1. change sample rate
    2. multiple channels -> mono channel
    '''
    samples, frame_rate  = librosa.load(src_wav, sr=None)
    #print("Original sample rate:", frame_rate)

    '''if frame_rate == sample_rate and len(samples.shape) == 1:
        return'''

    ff = FFmpeg(
        inputs={src_wav: None},
        outputs={dst_wav: "-ar {} -ac 1 -y -loglevel quiet".format(sample_rate)}
    )

    ff.run()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Resampling script')
    parser.add_argument('--sample_rate', type=int, help='type dataset for trimming')
    parser.add_argument('--in_dir', type=str, help='type dataset for trimming')

    args = parser.parse_args()

    if not args.in_dir or not args.sample_rate:
        parser.error('--sample_rate and --in_dir should be given')

    in_dir = args.in_dir
    sample_rate = args.sample_rate

    make_one_sample_rate_in_dir(in_dir, sample_rate)
