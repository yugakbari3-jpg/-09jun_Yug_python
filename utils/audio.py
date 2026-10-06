"""Shared audio helpers for feeding Whisper.

Whisper (both the faster-whisper and MLX backends) always expects audio at
`WHISPER_SAMPLE_RATE`, regardless of what the microphone or `cfg.sample_rate`
actually capture at. The listener and the dictation engine each used to carry
their own ad-hoc resampler that did plain linear interpolation with no
anti-aliasing filter — fine for upsampling, but silently wrong when
downsampling: content above the target Nyquist folds back down into the
audible band instead of being removed, corrupting the transcript. `resample`
is the one place that conversion happens now.
"""

import numpy as np

# The fixed rate Whisper is trained on and always expects its input at,
# independent of whatever rate the microphone/config captures audio at.
WHISPER_SAMPLE_RATE = 16000


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample 1-D float32 PCM from `src_rate` to `dst_rate`.

    Uses FFT-domain resampling: the spectrum is truncated (downsampling) or
    zero-padded (upsampling) to the target Nyquist before being transformed
    back to the time domain. Truncating acts as a perfect brick-wall
    anti-aliasing filter, so downsampling can never fold high-frequency
    content back into the band Whisper listens to.
    """
    if src_rate == dst_rate:
        return audio

    n_in = len(audio)
    if n_in == 0:
        return audio.astype(np.float32, copy=False)

    n_out = int(round(n_in * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    spectrum = np.fft.rfft(audio.astype(np.float64, copy=False))
    n_freq_in = spectrum.shape[0]
    n_freq_out = n_out // 2 + 1

    if n_freq_out <= n_freq_in:
        # Downsampling (or equal): drop everything above the new Nyquist.
        new_spectrum = spectrum[:n_freq_out]
    else:
        # Upsampling: zero-pad the extra high-frequency bins.
        new_spectrum = np.zeros(n_freq_out, dtype=spectrum.dtype)
        new_spectrum[:n_freq_in] = spectrum

    out = np.fft.irfft(new_spectrum, n=n_out)
    # rfft/irfft normalise by 1/len(signal) on the inverse transform; since
    # the forward and inverse lengths differ here, rescale to preserve
    # amplitude (mirrors scipy.signal.resample's Fourier method).
    scale = n_out / n_in
    return (out * scale).astype(np.float32)
