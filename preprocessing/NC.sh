cd ..

#cp -r /data2/sungjaecho/data_tts/NC/02_NC_renamed /data2/sungjaecho/data_tts/NC/03_NC_sr-48000

#python resample.py --sample_rate=48000 --in_dir=/data2/sungjaecho/data_tts/NC/03_NC_sr-48000

#cp -vr /data2/sungjaecho/data_tts/NC/03_NC_sr-48000 /data2/sungjaecho/data_tts/NC/04_NC_sr-48000_trimmed

#python trimmer.py --in_dir=/data2/sungjaecho/data_tts/NC/04_NC_sr-48000_trimmed/wav --out_dir=/data2/sungjaecho/data_tts/NC/04_NC_sr-48000_trimmed/wav_trimmed

#rm -r /data2/sungjaecho/data_tts/NC/04_NC_sr-48000_trimmed/wav

#mv /data2/sungjaecho/data_tts/NC/04_NC_sr-48000_trimmed/wav_trimmed /data2/sungjaecho/data_tts/NC/04_NC_sr-48000_trimmed/wav

#cp -rv /data2/sungjaecho/data_tts/NC/04_NC_sr-48000_trimmed /data2/sungjaecho/data_tts/NC/05_NC_sr-22050_trimmed

#python resample.py --sample_rate=22050 --in_dir=/data2/sungjaecho/data_tts/NC/05_NC_sr-22050_trimmed

rm -r /data2/sungjaecho/data_tts/NC/NC

mv /data2/sungjaecho/data_tts/NC/05_NC_sr-22050_trimmed /data2/sungjaecho/data_tts/NC/NC
