import numpy as np


def merge_best(
    sys_data: np.ndarray,
    mic_data: np.ndarray,
    sample_rate: int,
    window_ms: int = 10,
    threshold: float = 0.0005,
    hold_ms: int = 200,
) -> np.ndarray:
    """Select between system and microphone audio using simple gating.

    The system channel is preferred. When its RMS level drops below ``threshold``
    and the microphone level rises above the threshold the function switches to
    the microphone channel. The reverse happens when the microphone becomes
    quiet and the system is active again. The ``hold_ms`` parameter prevents
    rapid flapping by keeping the current channel active for at least that
    duration after a switch.
    """

    length = min(len(sys_data), len(mic_data))
    if length == 0:
        return np.array([], dtype=np.float32)

    sys_data = sys_data[:length]
    mic_data = mic_data[:length]

    window_samples = max(1, int(sample_rate * window_ms / 1000))
    hold_frames = max(1, int(hold_ms / window_ms))

    active = 0  # 0 -> sys, 1 -> mic
    hold = 0
    output = np.zeros(length, dtype=np.float32)

    for start in range(0, length, window_samples):
        end = min(length, start + window_samples)
        sys_win = sys_data[start:end]
        mic_win = mic_data[start:end]
        sys_rms = float(np.sqrt(np.mean(sys_win**2))) if len(sys_win) else 0.0
        mic_rms = float(np.sqrt(np.mean(mic_win**2))) if len(mic_win) else 0.0

        if hold == 0:
            if active == 0 and sys_rms < threshold and mic_rms > threshold:
                active = 1
                hold = hold_frames
            elif active == 1 and mic_rms < threshold and sys_rms > threshold:
                active = 0
                hold = hold_frames

        output[start:end] = mic_win if active else sys_win
        if hold > 0:
            hold -= 1

    return output
