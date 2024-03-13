import os
import subprocess
import argparse

# Funkcja do stosowania equalizera (EQ) za pomocą ffmpeg
def apply_eq_ffmpeg(input_dir, output_dir, num_bands, mid_band_gain, low_band_gain, high_band_gain):
    # Sprawdź czy katalog wyjściowy istnieje, jeśli nie, utwórz go
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Przetwarzanie wszystkich plików dźwiękowych w katalogu wejściowym
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav') or file_name.endswith('.mp3') or file_name.endswith('.ogg') or file_name.endswith('.flac'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            command = ['ffmpeg', '-i', input_path]
            for i in range(num_bands):
                if i % 2 == 0:
                    gain = mid_band_gain
                else:
                    gain = low_band_gain if i % 4 == 1 else high_band_gain
                    gain += 1 if i % 4 == 1 else 2  # Podbicie dolnego pasma o 1, górnego o 2
                command += ['-af', 'equalizer=f={}:t=h:width_type=q:width=1:g={}'.format(i, gain)]
            command += [output_path]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Parser argumentów wiersza poleceń
parser = argparse.ArgumentParser(description='Apply EQ to audio files in a directory.')
parser.add_argument('input_dir', type=str, help='Input directory containing audio files')
parser.add_argument('output_dir', type=str, help='Output directory to save processed audio files')
parser.add_argument('--num_bands', type=int, default=16, help='Number of bands in the EQ (default: 16)')
parser.add_argument('--mid_band_gain', type=int, default=3, help='Gain for the mid band (default: 3)')
parser.add_argument('--low_band_gain', type=int, default=1, help='Gain for the low band (default: 1)')
parser.add_argument('--high_band_gain', type=int, default=1, help='Gain for the high band (default: 1)')
args = parser.parse_args()

# Stosowanie EQ za pomocą ffmpeg
apply_eq_ffmpeg(args.input_dir, args.output_dir, args.num_bands, args.mid_band_gain, args.low_band_gain, args.high_band_gain)

