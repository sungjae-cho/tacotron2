import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.manifold import TSNE
from utils import get_f0, get_text_durations


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, encoding_len=None, decoding_len=None, info=None):
    # alignment.size == [txt_steps, mel_steps]

    '''
    max_mel_len = alignment.size(1)
    list_xticks = sorted(list(range(0, max_mel_len+1, step=100)) + [decoding_len])
    plt.xticks(ticks=x, rotation=45)
    '''
    alignment = alignment.astype(np.float32) # casting required when fp16_run.
    if encoding_len is None:
        encoding_len = alignment.shape[0]
    if decoding_len is  None:
        decoding_len = alignment.shape[1]
    alignment = alignment[:encoding_len,:decoding_len]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)

    xticks = set(range(0, decoding_len, 100))
    xticks.add(decoding_len)
    list_xticks = sorted(list(xticks))
    plt.xticks(ticks=list_xticks, rotation=45)

    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    spectrogram = spectrogram.astype(np.float32) # casting required when fp16_run.
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data



def plot_embeddings_to_numpy(labels, label_embeddings):
    tsne = TSNE(n_components=2, n_iter=100000)
    X_2d = tsne.fit_transform(label_embeddings)

    fig, ax = plt.subplots(figsize=(4, 4))

    plt.scatter(X_2d[:,0], X_2d[:,1], c=list(range(len(labels))), cmap='rainbow')

    for i in range(len(labels)):
        label = labels[i]
        plt.annotate(
            label,
            xy=(X_2d[i,0], X_2d[i,1]), xytext=(0, 30),
            textcoords='offset points', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6),
            arrowprops=dict(arrowstyle = '-', connectionstyle='arc3,rad=0'),
            annotation_clip=None
        )
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_prosody_dims_to_numpy(spectrogram, wav, text_seq, alignment, prosody, hparams):
    '''
    PARAMS
    -----
    spectrogram: Mel Spectrogram of speech.
    - type: numpy.ndarray.
    - shape: (frames, n_mel_channels)
    : A wav sequence of speech.
    - type: numpy.ndarray.
    - shape: (sample_len,)
    text_seq: A list of graphemes or phonemes.
    - type: list.
    - dtype: str.
    alignment: A stack of attention weights at every decoding step.
    - type: numpy.ndarray.
    - dtype: float.
    - shape: (mel_steps, text_steps)
    prosody: A pack of prosody encodings at speech spectrogram frames.
    - type: numpy.ndarry.
    - shape: (frames, prosody_dim)
    sr: The sample rate of the wav.
    - type: int.

    RETURNS
    -----
    data: An image in the format of NumPy.
    - type: numpy.ndarray
    '''
    n_decoding_steps = prosody.shape[0]
    n_prosody_dims = prosody.shape[1]

    n_leading_figures = 4
    n_figures = n_leading_figures + n_prosody_dims
    fig, axes = plt.subplots(n_figures, 1, figsize=(12, 2*n_figures))

    x = np.arange(n_decoding_steps)
    colors = ['orange', 'blue', 'green', 'magenta']

    # [1] First figure: Spectrogram
    spectrogram = spectrogram.astype(np.float32) # casting required when fp16_run.
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Channels")
    im = axes[0].imshow(spectrogram, aspect="auto", origin="lower",
               interpolation='none')

    # [2] Second figure: F0
    wav = wav.astype(np.float32) # casting required when fp16_run.
    f0 = get_f0(wav, hparams.sampling_rate,
            hparams.filter_length, hparams.hop_length, hparams.f0_min,
            hparams.f0_max, hparams.harm_thresh)
    f0 = f0[:n_decoding_steps]
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("F0 (Hz)")
    axes[1].set_xlim(x[0], x[-1])
    axes[1].set_ylim(hparams.f0_min, f0.max())
    axes[1].plot(x, f0)

    # [3] Third figure: Amplitude
    # Learned from https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
    D = np.abs(
            librosa.core.stft(wav,
                n_fft=hparams.filter_length,
                hop_length=hparams.hop_length,
                win_length=hparams.win_length))
    D = D[:,:n_decoding_steps]
    axes[2].set_xlabel("Frames")
    axes[2].set_ylabel("Amplitude (ΣSTFT(f))")
    axes[2].set_xlim(x[0], x[-1])
    axes[2].plot(x, D.sum(axis=0))

    # [4] Fourth figure: Duration
    durations, x_chunks, att_text_seq = get_text_durations(alignment)
    axes[3].set_xlabel("Frames")
    axes[3].set_ylabel("Duration (frames)")
    axes[3].set_xlim(x[0], x[-1])
    axes[3].set_ylim(0, max(durations)+1)
    i_color = 0
    for x_chunk in x_chunks:
        x_start = x_chunk[0]
        x_last = x_chunk[-1]
        duration = durations[x_start]
        int_att_text = att_text_seq[x_start]
        att_text = text_seq[int_att_text]
        axes[3].plot(x_chunk, durations[x_start:x_last+1], '-', color=colors[i_color % len(colors)])
        axes[3].text(x_start, duration, "{}".format(att_text))
        i_color += 1


    # [5] The rest of figures: Prosody dimensions.
    for i_dim in range(n_prosody_dims):
        prosody_dim = prosody[:,i_dim]
        prosody_dim = prosody_dim.astype(np.float32) # casting required when fp16_run.
        axes[i_dim+n_leading_figures].plot(x, prosody_dim, '-', color=colors[i_dim % len(colors)])
        axes[i_dim+n_leading_figures].set_xlabel("Decoding_step")
        axes[i_dim+n_leading_figures].set_ylabel("Prosody_dim{}".format(i_dim))
        axes[i_dim+n_leading_figures].set_xlim(x[0], x[-1])
        #axes[i_dim+n_leading_figures].set_ylim(0, prosody_dim.max())

    # [6] Convert figures to an image
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()

    return data
