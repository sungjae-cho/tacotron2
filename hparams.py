import tensorflow as tf
import numpy as np
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
        fp16_opt_level='O1',
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost",
        dist_port=54321,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        cudnn_deterministic=False,
        ignore_layers=['embedding.weight'],
        compute_alignments=False,

        # Logging options
        log_validation=True,
        log_fr_test=False, # Control for logging free-running test
        # SecondStopPredictor control
        val_tf_stop_pred2=False,
        val_fr_stop_pred2=False,
        # Free-running test text
        test_text = "This is a synthesized audio without teacher-forcing. Any question?",

        ################################
        # Data Parameters             #
        ################################
        waveglow_path='/data2/sungjaecho/pretrained/waveglow_256channels_ljs_v2.pt',
        load_mel_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        csv_data_paths={
            'ljspeech':'metadata/ljspeech.csv',
            'emovdb':'metadata/emovdb.csv',
            'bc2013':'metadata/bc2013.csv',
            'ketts':'metadata/ketts.csv',
            'ketts2':'metadata/ketts2.csv',
            'nc':'metadata/nc.csv',
            'kss':'metadata/kss.csv',
        },
        p_arpabet=1.0,
        cmudict_path='text/cmu_dictionary',
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
        f0_min=80,
        f0_max=880,
        harm_thresh=0.25,
        new_param=10,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        txt_type='g', # ['g', 'p_cmudict', 'p_g2p']

        # With modules
        speaker_adversarial_training=False,
        emotion_adversarial_training=False,
        residual_encoder=False,
        monotonic_attention=False,
        val_tf_zero_res_en=False,
        prosody_predictor='', # ['MLP', 'LSTM']
        reference_encoder='', # ['ref1', 'ref2', 'ref3', 'ref4']
        reference_encoders_taking_mels_at_inference=['Glob2Temp', 'ReferenceEncoder', 'GST', 'LocalConvRefEncoder'],
        reference_encoders_concat_global_styles_to_text=['ReferenceEncoder', 'GST'],
        reference_encoders_with_prosody_attention=['LocalConvRefEncoder'],

        # (Text) Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # SpeakerEncoder parameters
        speaker_embedding_dim=3,

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

        # Prosody predictor Parameters
        prosody_dim=4,
        pp_lstm_hidden_dim=512,
        pp_opt_inputs = [''], # ['prev_global_prosody', 'AttRNN']

        # Reference encoder
        loss_ref_enc_weight=1.0,
        with_gst=True,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_filter_size=[1, 3], # [time_wise_stride, freq_wise_stride]
        ref_enc_strides=[1, 2], # [time_wise_stride, freq_wise_stride]
        ref_enc_pad=[0, 1], #[1, 1],
        ref_enc_gru_size=128,
        global_prosody_is_hidden=False,

        # Style Token Layer
        token_embedding_size=128, # in the paper, 256
        token_num=10,
        num_heads=1,

        # Residual encoder parameters
        res_en_out_dim=16,
        res_en_conv_kernels=512,
        res_en_conv_kernel_size=(3,3),
        res_en_lstm_dim=256,
        std_lower_bound=np.exp(-2),
        KLD_weight_scheduling='fixed', # ['fixed', 'pulse', 'cycle_linear']
        ## fixed KLD weight (KLD_weight_scheduling == 'pulse_KLD_weight') hparams
        res_en_KLD_weight=1e-3,
        ## pulse KLD weight (KLD_weight_scheduling == 'pulse_KLD_weight') hparams
        KLD_weight_warm_up_step=15000,
        init_KLD_weight=0.001,
        KLD_weight_cof=0.002,
        ## cyclic linear KLD weight (KLD_weight_scheduling == 'pulse_KLD_weight') hparams
        cycle_KLDW_period=10000,
        cycle_KLDW_ratio=0.5,
        cycle_KLDW_min=0.0,
        cycle_KLDW_max=1e-5,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        style_to_attention_rnn=False,
        style_to_decoder_rnn=False,
        style_to_decoder_linear=False,
        style_to_encoder_output=False,

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

        # Adversarial training with the emotion classfier
        emotion_adv_weight=0.02,
        emotion_gradrev_lambda=1,
        emotion_gradrev_grad_max_norm=0.5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=True,
        learning_rate=1e-3,
        adam_batas=(0.9, 0.999),
        adam_eps=1e-06,
        weight_decay=1e-6,
        lr_scheduling=False,
        lr_scheduling_start_iter=50000,
        lr_min=1e-5,
        grad_clip_thresh=1.0, # gradient clipping L2-norm
        batch_size=64,
        mask_padding=True,  # set model's padded outputs to padded values
        freeze_pretrained=False,
        freeze_except_for=['nothing']
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
