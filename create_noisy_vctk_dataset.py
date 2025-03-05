import os
import numpy as np
import librosa
import soundfile as sf
import random
from scipy.signal import fftconvolve
from tqdm import tqdm

VCTK_PATH = ".\\VCTK-Corpus-0.92\\wav48_silence_trimmed"
HRIR_PATH = ".\\HRIR_database_wav_V1-1\\HRIR_database_wav\\hrir\\anechoic"
NOISE_PATH = ".\\Noises-master\\Noises-master\\NoiseX-92"

OUTPUT_PATHS = {
    "noisy_train": ".\\Dataset\\noisy_trainset_1f",
    "noisy_valid": ".\\Dataset\\noisy_valset_1f",
    "noisy_test": ".\\Dataset\\noisy_testset_1f",
    "clean_train": ".\\Dataset\\clean_trainset_1f",
    "clean_valid": ".\\Dataset\\clean_valset_1f",
    "clean_test": ".\\Dataset\\clean_testset_1f"
}

for path in OUTPUT_PATHS.values():
    os.makedirs(path, exist_ok=True)

SAMPLING_RATE = 16000
FFT_LENGTH = 512
WINDOW_LENGTH = int(0.025 * SAMPLING_RATE)
HOP_LENGTH = int(0.00625 * SAMPLING_RATE)
SNR_RANGE = (-7, 13)
EVAL_SNR_RANGE = (-6, 12)
DISTANCES = [0.8, 3.0]
AZIMUTH_RANGE = (-90, 90)
ROOM_T60_RANGE = (0.3, 1.2)
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
NUM_FILES = 100
TRAIN_SIZE = int(NUM_FILES*TRAIN_RATIO)
VALID_SIZE = int(NUM_FILES*VALID_RATIO)
TEST_SIZE = int(NUM_FILES*TEST_RATIO)

hrir_files = [os.path.join(HRIR_PATH, f) for f in os.listdir(HRIR_PATH)]
noise_files = [os.path.join(NOISE_PATH, f) for f in os.listdir(NOISE_PATH)]
vctk_speakers = [os.path.join(VCTK_PATH, f) for f in os.listdir(VCTK_PATH)]

NUM_SPEAKERS = len(vctk_speakers)

train_speakers = vctk_speakers[:int(TRAIN_RATIO * NUM_SPEAKERS)]
valid_speakers = vctk_speakers[int(TRAIN_RATIO * NUM_SPEAKERS):int((TRAIN_RATIO + VALID_RATIO) * NUM_SPEAKERS)]
test_speakers = vctk_speakers[int((TRAIN_RATIO + VALID_RATIO) * NUM_SPEAKERS):NUM_SPEAKERS]
test_speakers = vctk_speakers[int((TRAIN_RATIO + VALID_RATIO) * NUM_SPEAKERS):NUM_SPEAKERS]

train_files = [os.path.join(speaker, f) for speaker in train_speakers for f in os.listdir(speaker)]
valid_files = [os.path.join(speaker, f) for speaker in valid_speakers for f in os.listdir(speaker)]
test_files = [os.path.join(speaker, f) for speaker in test_speakers for f in os.listdir(speaker)]

train_files = train_files[:TRAIN_SIZE]
valid_files = valid_files[:VALID_SIZE]
test_files = test_files[:TEST_SIZE]

print(f"dataset size, trian: {len(train_files)}, validation: {len(valid_files)}, test: {len(test_files)}")

def apply_hrir(signal):
    hrir_file = random.choice(hrir_files)
    #hrir, _ = librosa.load(hrir_file, sr=SAMPLING_RATE)
    hrir, _ = sf.read(hrir_file)
    left_hrir = hrir[:,0]
    right_hrir = hrir[:,1]
    left_channel = fftconvolve(signal, left_hrir, mode='same')
    right_channel = fftconvolve(signal, right_hrir, mode='same')
    return np.stack([left_channel, right_channel], axis=0)


def add_noise(signal, split, num_noisers=2):
    all_noise = np.zeros_like(signal)
    for noiser in range(num_noisers):
        snr = random.uniform(*SNR_RANGE if split != 'test' else EVAL_SNR_RANGE)
        noise_file = random.choice(noise_files)
        noise, _ = librosa.load(noise_file, sr=SAMPLING_RATE, duration=2.0)
        noise = apply_hrir(noise)
        #noise = noise[:len(signal)] if len(noise) >= len(signal) else np.pad(noise, (0, len(signal) - len(noise)))
        noise = noise[:, :signal.shape[1]]
        all_noise += noise
    noise_power = np.mean(all_noise ** 2)
    signal_power = np.mean(signal ** 2)
    scaling_factor = np.sqrt((signal_power / noise_power) * 10 ** (-snr / 10))
    noisy_signal = signal + scaling_factor * all_noise
    return noisy_signal


def process_file(file, split):
    signal, _ = librosa.load(file, sr=SAMPLING_RATE, duration=2.0)
    signal = librosa.util.fix_length(signal, size=int(SAMPLING_RATE * 2.0))
    if signal.shape[0] != 32000:
        print(signal.shape)
    azimuth = random.randint(*AZIMUTH_RANGE)
    spatialized_signal = apply_hrir(signal)
    noisy_signal = add_noise(spatialized_signal, split)
    clean_output_file = os.path.join(OUTPUT_PATHS[f"clean_{split}"], os.path.basename(file))
    noisy_output_file = os.path.join(OUTPUT_PATHS[f"noisy_{split}"], os.path.basename(file))
    sf.write(clean_output_file, spatialized_signal.T, SAMPLING_RATE)
    sf.write(noisy_output_file, noisy_signal.T, SAMPLING_RATE)
    #print(spatialized_signal.T.shape)
    #print(noisy_signal.T.shape)


def generate_dataset():
    for split, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        print(f"total files in split {split}: {len(files)}")
        for i in tqdm(range(len(files))):
            process_file(files[i], split)


generate_dataset()
