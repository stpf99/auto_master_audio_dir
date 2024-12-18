import os
import subprocess
import argparse
import multiprocessing

def apply_effects(input_path, output_path, num_bands, step, mid_band_gain, low_band_gain, high_band_gain, lamp_head_gain):
    command = ['ffmpeg', '-i', input_path]
    
    # Wybieranie pasm co określony krok
    selected_bands = range(0, num_bands, step)
    
    for i in selected_bands:
        # Dynamiczne przypisywanie wzmocnienia w zależności od zakresu pasma
        if i % 4 == 0:  # Pasma środkowe
            gain = mid_band_gain
        elif i % 4 == 1:  # Dolne pasma
            gain = low_band_gain + 1  # Lekkie wzmocnienie dolnych częstotliwości
        else:  # Górne pasma
            gain = high_band_gain + 2  # Nieco większe wzmocnienie górnych częstotliwości
        
        command += ['-af', f'equalizer=f={i}:t=h:width_type=q:width=1:g={gain}']
    
    # Normalizacja dźwięku i ustawienie częstotliwości próbkowania
    command += ['-af', 'loudnorm=i=-16:tp=-2:lra=11:print_format=summary', '-ar', '44100']
    command += [output_path]
    
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Funkcja pomocnicza dla przetwarzania równoległego
def process_file(file):
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)
    apply_effects(input_path, output_path, num_bands, step, mid_band_gain, low_band_gain, high_band_gain, lamp_head_gain)

# Parser argumentów wiersza poleceń
parser = argparse.ArgumentParser(description='Apply stepped EQ and "Lamp head effect" to audio files in a directory.')
parser.add_argument('input_dir', type=str, help='Input directory containing audio files')
parser.add_argument('output_dir', type=str, help='Output directory to save processed audio files')
parser.add_argument('--num_bands', type=int, default=16, help='Number of bands in the EQ (default: 16)')
parser.add_argument('--step', type=int, default=2, help='Step for selecting frequency bands (default: 2)')
parser.add_argument('--mid_band_gain', type=int, default=3, help='Gain for the mid band (default: 3)')
parser.add_argument('--low_band_gain', type=int, default=1, help='Gain for the low band (default: 1)')
parser.add_argument('--high_band_gain', type=int, default=1, help='Gain for the high band (default: 1)')
parser.add_argument('--lamp_head_gain', type=int, default=50, choices=range(0, 101, 5), help='Lamp head effect gain (0-100 with step 5, default: 50)')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
num_bands = args.num_bands
step = args.step
mid_band_gain = args.mid_band_gain
low_band_gain = args.low_band_gain
high_band_gain = args.high_band_gain
lamp_head_gain = args.lamp_head_gain

# Sprawdź czy katalog wyjściowy istnieje, jeśli nie, utwórz go
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lista plików dźwiękowych w katalogu wejściowym
files = [file_name for file_name in os.listdir(input_dir) if file_name.endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))]

# Przetwarzanie plików równolegle
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(process_file, files)
