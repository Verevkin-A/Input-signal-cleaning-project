# Verevkin Aleksandr (xverev00)
# ISS project 2021/22

import numpy as np
import matplotlib.pyplot as plt
import math
import IPython
import scipy
from IPython.core.display import clear_output
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
from scipy.io import wavfile

# Plotting configuration
plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True
# plt.rcParams["axes.grid"] = True


# Function for counting DFT of given signal
def dft(frame):
    length = len(frame)
    if length != 1024:
        frame = np.append(frame, np.zeros(1024 - length))
    dft_s = []
    for i in range(length):
        appended = 0
        for j in range(length):
            appended += frame[j] * np.exp(-2j * np.pi * i * j / length)
        dft_s.append(appended)
    return dft_s


print("---------------4.1---------------")

# Opening and plotting of signal
fs, s = wavfile.read("../audio/xverev00.wav")

# FIRST SIGNAL
t = np.arange(s.size) / fs
plt.plot(t, s)
plt.title("Sound signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.savefig("4_1.pdf")
# plt.show()
plt.clf()

# Length of the signal
len_sec = s.size / fs
print("Number of signal samples:", s.size, "[samples]")
print("Length of signal:", len_sec, "[sec]")
# Found max and min values of the signal
print("Min. value:", s.min(), ", max. value:", s.max())

print("---------------4.2---------------")

# Normalized signal (-1, 1)
s = (s - np.mean(s)) / 2 ** 15

# Cut signal into frames
count_of_frames = math.floor(s.size / 512)
frame_arr = [[0 for x in range(1024)] for y in range(count_of_frames)]
frame_start = 0
frame_end = 1024
frame_counter = 0
while frame_counter != count_of_frames:
    if frame_counter + 1 == count_of_frames:
        frame_arr[frame_counter] = s[frame_start: s.size - 1]
    else:
        frame_arr[frame_counter] = s[frame_start: frame_end]
    frame_start += 512
    frame_end += 512
    frame_counter += 1

# Manually showing every frame to pick one
# for i, frame in enumerate(frame_arr):
#     t = np.arange(len(frame)) / fs
#     plt.plot(t, frame)
#     plt.title("Frame #" + str(i))
#     plt.xlabel("Time [s]")
#     plt.ylabel("Amplitude")
#     plt.show()
#     plt.pause(0.5)
#     clear_output(wait=True)

# Selected frame
frame_index = 22
picked_frame = frame_arr[frame_index]

# Plot picked frame
t = np.arange(len(picked_frame)) / fs
plt.plot(t, picked_frame)
plt.title("Picked frame #" + str(frame_index + 1))
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.savefig("4_2.pdf")
# plt.show()
plt.clf()

print("---------------4.3---------------")

# Frame signal DFT counting
s_seg_spec_my = dft(picked_frame)
s_seg_spec_func = np.fft.fft(picked_frame)
print("DFT was counted right: " + str(np.allclose(s_seg_spec_my, s_seg_spec_func)))

# Plot module of DFT for frequency Fs/2
s_seg_spec_my = s_seg_spec_my[:len(s_seg_spec_my)//2]
freq = np.arange(0, fs/2, fs/1024)
s_seg_spec_my_abs = np.abs(s_seg_spec_my)
plt.plot(freq, s_seg_spec_my_abs)
plt.title("DFT of frame #" + str(frame_index + 1))
plt.xlabel("Frequency [Hz]")
plt.ylabel("|DFT(s)|")
plt.savefig("4_3.pdf")
# plt.show()
plt.clf()

print("---------------4.4---------------")

# Spectrogram of the signal
f, t, sgr = spectrogram(s, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr+1e-20)

# Plotting spectrogram
plt.pcolormesh(t, f, sgr_log)
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
cbar = plt.colorbar()
cbar.set_label('Power spectral density [dB]', labelpad=15)
plt.savefig("4_4.pdf")
# plt.show()
plt.clf()

print("---------------4.5---------------")

# Frequencies were determined manually, from spectrogram
f1 = 720
f2 = 1440
f3 = 2160
f4 = 2880

# Frequencies f2,f3,f4 are all multiplications of f1
print("f1 = " + str(f1) + "[Hz], f2 = " + str(f2) + "[Hz], f3 = " + str(f3) + "[Hz], f4 = " + str(f4) + "[Hz]")
print("f2 is multiplication of f1 by 2:", f2 == (f1 * 2))
print("f3 is multiplication of f1 by 3:", f3 == (f1 * 3))
print("f4 is multiplication of f1 by 4:", f4 == (f1 * 4))

print("---------------4.6---------------")

# Count cosines of frequencies f1 f2 f3 f4 and their mix
cos1 = np.cos(2 * f1 * np.pi * s)
cos2 = np.cos(2 * f2 * np.pi * s)
cos3 = np.cos(2 * f3 * np.pi * s)
cos4 = np.cos(2 * f4 * np.pi * s)
cos_mix = cos1 + cos2 + cos3 + cos4
wavfile.write("../audio/4cos.wav", fs, cos_mix.astype(np.float32))

# Calculating spectrogram for cosines mix
cos_fs, cos_s = wavfile.read("../audio/4cos.wav")
cos_f, cos_t, cos_sgr = spectrogram(cos_s, cos_fs, nperseg=1024, noverlap=512)
cos_sgr_log = 10 * np.log10(cos_sgr+1e-20)

# Plotting spectrogram of cosines mix
plt.pcolormesh(cos_t, cos_f, cos_sgr_log)
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
cbar = plt.colorbar()
cbar.set_label('Power spectral density [dB]', labelpad=15)
plt.savefig("4_6.pdf")
# plt.show()
plt.clf()

print("---------------4.7---------------")

order, Wn = scipy.signal.buttord(30/fs, 130/fs, 3, 40)
wt = 50
nyq = fs * 0.5
# Filtered frequency
frequency = f1
lowest = (frequency - wt) / nyq
highest = (frequency + wt) / nyq
# Counting of coefficients
[b, a] = scipy.signal.butter(order, [lowest, highest], btype="bandstop")
print("b1 coefficients:", b)
print("a1 coefficients:", a)
frequency = f2
lowest = (frequency - wt) / nyq
highest = (frequency + wt) / nyq
[b2, a2] = scipy.signal.butter(order, [lowest, highest], btype="bandstop")
print("b2 coefficients:", b2)
print("a2 coefficients:", a2)
frequency = f3
lowest = (frequency - wt) / nyq
highest = (frequency + wt) / nyq
[b3, a3] = scipy.signal.butter(order, [lowest, highest], btype="bandstop")
print("b3 coefficients:", b3)
print("a3 coefficients:", a3)
frequency = f4
lowest = (frequency - wt) / nyq
highest = (frequency + wt) / nyq
[b4, a4] = scipy.signal.butter(order, [lowest, highest], btype="bandstop")
print("b4 coefficients:", b4)
print("a4 coefficients:", a4)

# Impulse response
N_imp = 64
imp = [1, *np.zeros(N_imp-1)]
h = lfilter(b, a, imp)

# Frequency characteristics
w, H = freqz(b, a)

# Zeros and poles
z, p, k = tf2zpk(b, a)

# Stability
is_stable = (p.size == 0) or np.all(np.abs(p) < 1)
print('Filter {} stable.'.format('is' if is_stable else 'isn\'t'))

# Plotting
plt.stem(np.arange(N_imp), h, basefmt=' ')
plt.xlabel('n')
# Change on f1, f2, f3 or f4
plt.title('Impulse response {} h[n]'.format(str(f1) + "Hz"))
plt.savefig("4_7_f1.pdf")
# plt.show()
plt.clf()

print("---------------4.8---------------")

# Plotting zeroes and poles
ang = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(ang), np.sin(ang))
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='Poles')
plt.xlabel('Real part $\mathbb{R}\{$z$\}$')
plt.ylabel('Imaginary part $\mathbb{I}\{$z$\}$')
# Change for right label f1, f2, f3, f4
plt.title("Butterworth filter " + str(f1) + "Hz")
plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper right')
plt.savefig("4_8_f1.pdf")
# plt.show()
plt.clf()

print("---------------4.9---------------")

# Plotting frequency characteristics
_, ax = plt.subplots(1, 2, figsize=(10, 3))

ax[0].plot(w / 2 / np.pi * fs, np.abs(H))
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_title('Frequency characteristics(720Hz) module $|H(e^{j\omega})|$')

ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_title('Frequency characteristics(720Hz) argument $\mathrm{arg}\ H(e^{j\omega})$')

for ax1 in ax:
    ax1.grid(alpha=0.5, linestyle='--')
plt.savefig("4_9_f1.pdf")
# plt.show()
plt.clf()

print("---------------4.10--------------")

# Frequencies filtration
sf = lfilter(b, a, s)
sf = lfilter(b2, a2, sf)
sf = lfilter(b3, a3, sf)
sf = lfilter(b4, a4, sf)

# Save filtered signal
wavfile.write("../audio/clean_bandstop.wav", fs, sf.astype(np.float32))
