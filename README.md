python3 auto_master_dir.py input_dir output_dir --num_bands 16 --mid_band_gain 3 --low_band_gain 1 --high_band_gain 1


python3 auto_master_dir_lamps.py in out --num_bands 64 --mid_band_gain 8 --low_band_gain 0 --high_band_gain 10 --lamp_head_gain 100

multiprocessing:

230 TRACKS 93min > 8 min 

on xeon 1.8ghz x 12c/24t

-------------------

python audio-multiprocessing-eq.py in/ out/ --step 3 --num_bands 24 --mid_band_gain 3 --low_band_gain 2 --high_band_gain 2 --lamp_head_gain 100

-------------------

Parametr --step powoduje, że niektóre zakresy częstotliwości są pomijane we wzmacnianiu.

Przykładowo:

Przy num_bands=16 i step=2:

Wzmocnione zostaną pasma: 0, 2, 4, 6, 8, 10, 12, 14

Pasma 1, 3, 5, 7, 9, 11, 13, 15 będą całkowicie pominięte


Przy step=4:

Wzmocnione zostaną pasma: 0, 4, 8, 12

Pasma 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15 będą pominięte


dla selektywnego wybierania pasm częstotliwości, aby zredukować ryzyko niepożądanych wibracji i zniekształceń w słuchawkach i głośnikach.

W realnym świecie wybór najlepszych ustawień zależy od kilku czynników, ale mogę zaproponować optymalne parametry oparte na praktyce audio:

Liczba pasm (num_bands):


16-32 pasm dla precyzyjnej kontroli
Rekomendacja: 24 pasma jako kompromis między szczegółowością a prostotą


Krok (step):


2-4 dla subtelnej, delikatnej korekcji
Rekomendacja: 3 lub 4

Step 3 = bardziej selektywne próbkowanie
Step 4 = mniej ingerencji, większe bezpieczeństwo




Wzmocnienie pasm:


Środek (mid): +2 do +4 dB
Niskie częstotliwości (low): +1 do +2 dB
Wysokie częstotliwości (high): +1 do +3 dB
