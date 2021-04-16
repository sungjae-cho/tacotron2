cd ..

cp -vr /data2/sungjaecho/data_tts/KAB/02_KAB_renamed /data2/sungjaecho/data_tts/KAB/03_KAB_sr-48000

python resample.py --sample_rate=48000 --in_dir=/data2/sungjaecho/data_tts/KAB/03_KAB_sr-48000

cp -vr /data2/sungjaecho/data_tts/KAB/03_KAB_sr-48000 /data2/sungjaecho/data_tts/KAB/04_KAB_sr-48000_trimmed

python trimmer.py --in_dir=/data2/sungjaecho/data_tts/KAB/04_KAB_sr-48000_trimmed/wav --out_dir=/data2/sungjaecho/data_tts/KAB/04_KAB_sr-48000_trimmed/wav_trimmed

rm -r /data2/sungjaecho/data_tts/KAB/04_KAB_sr-48000_trimmed/wav

mv /data2/sungjaecho/data_tts/KAB/04_KAB_sr-48000_trimmed/wav_trimmed /data2/sungjaecho/data_tts/KAB/04_KAB_sr-48000_trimmed/wav

cp -rv /data2/sungjaecho/data_tts/KAB/04_KAB_sr-48000_trimmed /data2/sungjaecho/data_tts/KAB/05_KAB_sr-22050_trimmed

python resample.py --sample_rate=22050 --in_dir=/data2/sungjaecho/data_tts/KAB/05_KAB_sr-22050_trimmed

rm -r /data2/sungjaecho/data_tts/KAB/KAB

mv /data2/sungjaecho/data_tts/KAB/05_KAB_sr-22050_trimmed /data2/sungjaecho/data_tts/KAB/KAB
