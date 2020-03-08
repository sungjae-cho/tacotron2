import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost",
        dist_port=54321,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        csv_data_paths={
            'ljspeech':'metadata/ljspeech.csv',
            'emovdb':'metadata/emovdb.csv'
        },
        # All DBs to be used in the project
        all_dbs = ['ljspeech', 'emovdb'],
        # All speakers to be used in the project
        all_speakers = ['ljs-w', 'emovdb-w-bea', 'emovdb-w-jenie', 'emovdb-m-josh', 'emovdb-m-sam'],
        # All emotions to be used in the project
        all_emotions = ['neutral', 'amused', 'angry', 'disgusted', 'sleepy'],
        # DBs to use while this run
        dbs = ['ljspeech', 'emovdb'],
        # Emotions to use while this run
        emotions = ['neutral', 'amused', 'angry', 'disgusted', 'sleepy'],
        # Speakers to use while this run
        speakers = ['ljs-w', 'emovdb-w-bea', 'emovdb-w-jenie', 'emovdb-m-josh', 'emovdb-m-sam'],
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        new_param=10,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # With modules
        speaker_adversarial_training=False,
        residual_encoder=False,
        monotonic_attention=False,

        # (Text) Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # SpeakerEncoder parameters
        speaker_embedding_dim=6,

        # LanguageEncoder parameters
        max_languages=2,
        lang_embedding_dim=3,

        # EmotionEncoder parameters
        emotion_embedding_dim=3,
        neutral_zero_vector=True,

        # SpeakerClassifier parameters
        n_hidden_units=256,
        revgrad_lambda=1.0,
        revgrad_max_grad_norm=0.5,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        has_style_token_lstm_1=True,
        has_style_token_lstm_2=True,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # MontonicAttention parameters
        loss_att_means_weight=0.1,
        n_mean_units=1,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # Adversarial training with the speaker classfier
        speaker_adv_weight=0.02,
        speaker_gradrev_lambda=1,
        speaker_gradrev_grad_max_norm=0.5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=True,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0, # gradient clipping L2-norm
        batch_size=64,
        mask_padding=True,  # set model's padded outputs to padded values
        freeze_pretrained=False,
        freeze_except_for=[None]
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
