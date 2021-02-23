cd ..


#python resample.py --sample_rate=48000 --in_dir=/data2/sungjaecho/data_tts/KETTS2/03_KETTS2_sr-48000

cp -r /data2/sungjaecho/data_tts/KETTS2/03_KETTS2_sr-48000 /data2/sungjaecho/data_tts/KETTS2/04_KETTS2_sr-48000_trimmed

python trimmer.py --in_dir=/data2/sungjaecho/data_tts/KETTS2/04_KETTS2_sr-48000_trimmed/wav --out_dir=/data2/sungjaecho/data_tts/KETTS2/04_KETTS2_sr-48000_trimmed/wav_trimmed

rm -r /data2/sungjaecho/data_tts/KETTS2/04_KETTS2_sr-48000_trimmed/wav

mv /data2/sungjaecho/data_tts/KETTS2/04_KETTS2_sr-48000_trimmed/wav_trimmed /data2/sungjaecho/data_tts/KETTS2/04_KETTS2_sr-48000_trimmed/wav

cp -r /data2/sungjaecho/data_tts/KETTS2/04_KETTS2_sr-48000_trimmed /data2/sungjaecho/data_tts/KETTS2/05_KETTS2_sr-22050_trimmed

python resample.py --sample_rate=22050 --in_dir=/data2/sungjaecho/data_tts/KETTS2/05_KETTS2_sr-22050_trimmed

rm -r /data2/sungjaecho/data_tts/KETTS2/KETTS2

mv /data2/sungjaecho/data_tts/KETTS2/05_KETTS2_sr-22050_trimmed /data2/sungjaecho/data_tts/KETTS2/KETTS2
