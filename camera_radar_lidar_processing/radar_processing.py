import numpy as np


def get_complex_content(file, N_chirp=256, N_antenna=3):
    data = open(file, "rb").read()
    arr = np.frombuffer(data, dtype=np.uint16)

    content = np.empty((N_antenna, N_chirp, N_chirp), dtype="complex")
    size = 2 * N_chirp * N_chirp
    for i in range(N_antenna):
        sub = arr[i*size:(i+1)*size]
        content[i] = (sub[0::2] + 1j * sub[1::2]).reshape((N_chirp, N_chirp))
    return content

def FFT(RX, N_chirp=256):
    window = np.hamming(N_chirp)
    rx = RX * window[:, None]  # No windowing for now
    fft = np.fft.fft2(rx)
    rd = np.fft.fftshift(fft, axes=0)
    return np.abs(rd) ** 2

def compute_rd(file, background, remove_background=False):
    content = get_complex_content(file)
    RX1 = FFT(content[0])
    RX2 = FFT(content[1])
    RX3 = FFT(content[2])

    RD_avg = (RX1 + RX2 + RX3) / 3
    rd_sub = RD_avg / background if remove_background else RD_avg
    rd = np.clip(rd_sub, 0, None)
    return rd