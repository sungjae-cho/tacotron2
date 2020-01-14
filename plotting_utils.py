import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, decoding_len=None, info=None):
    # alignment.size == [batch_size, mel_steps, txt_steps]

    '''
    max_mel_len = alignment.size(1)
    list_xticks = sorted(list(range(0, max_mel_len+1, step=100)) + [decoding_len])
    plt.xticks(ticks=x, rotation=45)
    '''
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')

    if decoding_len is not None:
        plt.axvline(x=decoding_len, color='r')
    fig.colorbar(im, ax=ax)

    max_mel_len = alignment.shape[1]
    if decoding_len is not None:
        list_xticks = sorted(list(range(0, max_mel_len+1, 100)) + [decoding_len])
    else:
        list_xticks = sorted(list(range(0, max_mel_len+1, 100)))
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
