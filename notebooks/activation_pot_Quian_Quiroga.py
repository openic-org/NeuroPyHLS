import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans

try:
    import pywt
except ImportError:
    pywt = None


# Filtros de descomposicion Daubechies-4 (db4) para fallback sin PyWavelets
_DB4_DEC_LO = np.array([
    -0.010597401784997278,
     0.032883011666982945,
     0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
     0.6308807679295904,
     0.7148465705525415,
     0.23037781330885523,
], dtype=np.float64)

_DB4_DEC_HI = np.array([
    -0.23037781330885523,
     0.7148465705525415,
    -0.6308807679295904,
    -0.02798376941698385,
     0.18703481171888114,
     0.030841381835986965,
    -0.032883011666982945,
    -0.010597401784997278,
], dtype=np.float64)


def _dwt_1d_symmetric_db4(signal: np.ndarray, level: int = 4) -> np.ndarray:
    """Descomposicion DWT 1D multinivel (db4, extension simetrica) equivalente a pywt.wavedec."""
    if pywt is not None:
        coeffs = pywt.wavedec(signal, wavelet="db4", mode="symmetric", level=level)
        return np.concatenate(coeffs)

    pad_len = len(_DB4_DEC_LO) - 1
    approx = signal.astype(np.float64)
    details = []
    for _ in range(level):
        padded = np.pad(approx, pad_width=pad_len, mode="symmetric")
        cA = np.convolve(padded, _DB4_DEC_LO, mode="valid")[1::2]
        cD = np.convolve(padded, _DB4_DEC_HI, mode="valid")[1::2]
        details.append(cD)
        approx = cA
    return np.concatenate([approx, *details[::-1]])


def run_potencial_activacion_quian_quiroga(
    sampling_freq: float = 24000.0,
    samples: int = 48000,
    noise_level: float = 0.2,
    num_spikes: int = 100,
    spike_length: int = 64,
    threshold_factor: float = 4.0,
    wavelet_type: str = "db4",
    seed: int | None = None,
):
    """
    Traduccion fiel del diagrama de bloques de LabVIEW:
    Potencial_de_Activacion_Quian_Quiroga.vi

    Etapas incluidas:
      1. Generador de senhal neuronal sintetica (Raw Neural Signal).
      2. Filtro pasa banda Butterworth de 2do orden (300 Hz - 3000 Hz).
      3. Estimacion de umbral MAD segun Quian Quiroga (2004):
         sigma_n = median(|x|) / 0.6745 ; Thr = Threshold_Factor * sigma_n.
      4. Deteccion directa por cruce de umbral sobre la senhal filtrada,
         extraccion de ventanas de 64 muestras (idx - 20) y periodo refractario (+44).
      5. Reconstruccion de senhal solo con picos detectados (Peak detector).
      6. Descomposicion Wavelet Discreta (DWT, nivel 4, extension simetrica).
      7. Seleccion de los 2 coeficientes con mayor varianza (Coefficient Discriminator).
      8. Agrupamiento K-Means (k = 2) y promediado de potenciales de accion (Neuron Sorting).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / sampling_freq

    # -------------------------------------------------------------------------
    # 1. Raw Neural Signal Generator
    # -------------------------------------------------------------------------
    noise = rng.normal(loc=0.0, scale=noise_level, size=samples)

    t_spike = np.arange(spike_length) / spike_length
    spike_waveform = np.sin(2.0 * np.pi * t_spike)

    spike_train = np.zeros(samples, dtype=np.float64)
    max_start_idx = samples - spike_length
    for _ in range(num_spikes):
        start_idx = int(np.round(rng.uniform(0.0, 1.0) * max_start_idx))
        start_idx = min(start_idx, max_start_idx)
        spike_train[start_idx : start_idx + spike_length] = spike_waveform

    raw_neural_signal = noise + spike_train

    # -------------------------------------------------------------------------
    # 2. Filtro Pasa Banda Butterworth Orden 2 (300 Hz - 3000 Hz)
    # -------------------------------------------------------------------------
    nyquist = 0.5 * sampling_freq
    b, a = butter(N=2, Wn=[300.0 / nyquist, 3000.0 / nyquist], btype="bandpass")
    filtered_signal = lfilter(b, a, raw_neural_signal)

    # -------------------------------------------------------------------------
    # 3. Umbral MAD de Quian Quiroga: Thr = Threshold_Factor * median(|x|) / 0.6745
    # -------------------------------------------------------------------------
    sigma_n = float(np.median(np.abs(filtered_signal)) / 0.6745)
    threshold = threshold_factor * sigma_n

    # -------------------------------------------------------------------------
    # 4. Thresholded Signal (deteccion directa sobre filtered_signal y recorte K x 64)
    # -------------------------------------------------------------------------
    INT32_MAX = 2147483647
    peak_indices_buffer = np.full(samples, INT32_MAX, dtype=np.int32)

    peak_windows = []
    search_start = 0
    iteration = 0

    while True:
        # Equivalente a Threshold 1D Array a partir de search_start sobre filtered_signal
        sub_sig = filtered_signal[search_start:]
        crossings = np.where(sub_sig >= threshold)[0]
        idx_cross = samples if len(crossings) == 0 else search_start + int(crossings[0])

        # Registro del Indice detectado en el Shift Register
        if iteration < samples:
            peak_indices_buffer[iteration] = idx_cross

        # Extraccion de ventana de 64 muestras alineada en idx_cross - 20
        win_start = idx_cross - 20
        win_end = win_start + spike_length
        if win_start>=0 and win_end <= samples:
            window = filtered_signal[win_start:win_end].copy()
        else:
            window = np.zeros(spike_length, dtype=np.float64)
            valid_src_start = max(0, win_start)
            valid_src_end = min(samples, win_end)
            if valid_src_start < valid_src_end:
                dst_start = valid_src_start - win_start
                window[dst_start : dst_start + (valid_src_end - valid_src_start)] = (
                    filtered_signal[valid_src_start:valid_src_end]
                )

        peak_windows.append(window)

        # Periodo refractario (+44 muestras) y condicion de parada (>= samples - 1)
        next_search = idx_cross + 44
        iteration += 1
        if next_search >= samples - 1:
            break
        search_start = next_search

    # Filtrado de la iteracion centinela final (cuando idx_cross == samples)
    if len(peak_windows) > 0 and peak_indices_buffer[len(peak_windows) - 1] >= samples:
        peak_samples_kx64 = np.array(peak_windows[:-1], dtype=np.float64)
    else:
        peak_samples_kx64 = np.array(peak_windows, dtype=np.float64)

    K = peak_samples_kx64.shape[0]
    valid_peak_indices = peak_indices_buffer[:K]

    # -------------------------------------------------------------------------
    # 5. Peak Detector (Raw Neural Signal Filtered - peaks only)
    # -------------------------------------------------------------------------
    peaks_only_signal = np.zeros(samples, dtype=np.float64)
    for k in range(K):
        p_idx = int(valid_peak_indices[k])
        if 0 <= p_idx < samples:
            peaks_only_signal[p_idx] = filtered_signal[p_idx]

    # -------------------------------------------------------------------------
    # 6. DWT (db4, nivel 4, extension simetrica)
    # -------------------------------------------------------------------------
    dwt_rows = []
    for k in range(K):
        if pywt is not None:
            coeffs = pywt.wavedec(
            peak_samples_kx64[k], wavelet=wavelet_type, mode="symmetric", level=4)
            dwt_rows.append(np.concatenate(coeffs))
        else:
            dwt_rows.append(_dwt_1d_symmetric_db4(peak_samples_kx64[k], level=4))
    dwt_coeff_matrix = np.array(dwt_rows, dtype=np.float64)

    # -------------------------------------------------------------------------
    # 7. Variance (aprox. K-S test) & Coefficient Discriminator
    # -------------------------------------------------------------------------
    col_variances = np.var(dwt_coeff_matrix, axis=0, ddof=1)

    idx_coeff_1 = int(np.argmax(col_variances))
    var_1 = col_variances[idx_coeff_1]

    variances_without_first = col_variances.copy()
    variances_without_first[idx_coeff_1] = 0.0
    idx_coeff_2 = int(np.argmax(variances_without_first))
    var_2 = variances_without_first[idx_coeff_2]

    detected_waveform_coeffs_1 = dwt_coeff_matrix[:, idx_coeff_1]
    detected_waveform_coeffs_2 = dwt_coeff_matrix[:, idx_coeff_2]
    coefficients_matrix_kx2 = np.column_stack(
        (detected_waveform_coeffs_1, detected_waveform_coeffs_2)
    )

    est_noise_stddev_1 = float(np.sqrt(var_1))
    est_noise_stddev_2 = float(np.sqrt(var_2))

    # -------------------------------------------------------------------------
    # 8. K-Means (k = 2) & Neuron Sorting (Activation Potential)
    # -------------------------------------------------------------------------
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=seed)
    neuron_identifier = kmeans.fit_predict(coefficients_matrix_kx2)
    centroids = kmeans.cluster_centers_

    centroid_1 = centroids[0]  # Neuron A
    centroid_2 = centroids[1]  # Neuron B

    cluster_a_windows = peak_samples_kx64[neuron_identifier == 0]
    cluster_b_windows = peak_samples_kx64[neuron_identifier == 1]

    activation_potential_a = (
        np.mean(cluster_a_windows, axis=0)
        if len(cluster_a_windows) > 0
        else np.zeros(spike_length)
    )
    activation_potential_b = (
        np.mean(cluster_b_windows, axis=0)
        if len(cluster_b_windows) > 0
        else np.zeros(spike_length)
    )

    return {
        "dt": dt,
        "raw_neural_signal": raw_neural_signal,
        "filtered_signal": filtered_signal,
        "sigma_n": sigma_n,
        "threshold": threshold,
        "peaks_only_signal": peaks_only_signal,
        "peak_samples_kx64": peak_samples_kx64,
        "dwt_coeff_matrix": dwt_coeff_matrix,
        "coefficients_matrix_kx2": coefficients_matrix_kx2,
        "detected_waveform_coeffs_1": detected_waveform_coeffs_1,
        "detected_waveform_coeffs_2": detected_waveform_coeffs_2,
        "est_noise_stddev_1": est_noise_stddev_1,
        "est_noise_stddev_2": est_noise_stddev_2,
        "neuron_identifier": neuron_identifier,
        "centroid_1": centroid_1,
        "centroid_2": centroid_2,
        "activation_potential_a": activation_potential_a,
        "activation_potential_b": activation_potential_b,
    }


def plot_front_panel(results: dict):
    """Genera las graficas equivalentes al Panel Frontal de LabVIEW."""
    dt = results["dt"]
    t = np.arange(len(results["filtered_signal"])) * dt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        "Potencial de Activacion - Metodo Quian Quiroga (MAD + DWT)",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Raw Neural Signal
    axes[0, 0].plot(t, results["raw_neural_signal"], color="black", linewidth=0.5)
    axes[0, 0].set_title("Raw Neural Signal")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude (V)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Senhal Filtrada + Umbral MAD
    axes[0, 1].plot(
        t, results["filtered_signal"], color="darkgreen", linewidth=0.5, label="Filtered Signal"
    )
    axes[0, 1].axhline(
        results["threshold"], color="red", linestyle="--", linewidth=1.2, label="MAD Threshold"
    )
    axes[0, 1].set_title("Bandpass Signal (300 Hz - 3 kHz) & Threshold")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude (V)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Raw Neural Signal Filtered (peaks only)
    axes[1, 0].plot(t, results["peaks_only_signal"], color="navy", linewidth=0.6)
    axes[1, 0].set_title("Raw Neural Signal Filtered (peaks only)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Amplitude (V)")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Spike Clusters (Coeff 1 vs Coeff 2)
    coeffs_kx2 = results["coefficients_matrix_kx2"]
    labels = results["neuron_identifier"]
    axes[1, 1].scatter(
        coeffs_kx2[labels == 0, 0],
        coeffs_kx2[labels == 0, 1],
        c="red",
        alpha=0.7,
        s=25,
        label="Cluster A (0)",
    )
    axes[1, 1].scatter(
        coeffs_kx2[labels == 1, 0],
        coeffs_kx2[labels == 1, 1],
        c="limegreen",
        alpha=0.7,
        s=25,
        label="Cluster B (1)",
    )
    axes[1, 1].scatter(*results["centroid_1"], c="darkred", marker="X", s=120, label="Centroid 1")
    axes[1, 1].scatter(*results["centroid_2"], c="darkgreen", marker="X", s=120, label="Centroid 2")
    axes[1, 1].set_title("Spike Clusters (Coeff 1 vs Coeff 2)")
    axes[1, 1].set_xlabel("Coeff 1")
    axes[1, 1].set_ylabel("Coeff 2")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Detected Waveform Coefficients 1 & 2
    axes[2, 0].plot(results["detected_waveform_coeffs_1"], label="Coeff 1", linewidth=1.0)
    axes[2, 0].plot(results["detected_waveform_coeffs_2"], label="Coeff 2", linewidth=1.0)
    axes[2, 0].set_title("Detected Waveform Coefficients 1 & 2")
    axes[2, 0].set_xlabel("Detected Spike Index")
    axes[2, 0].set_ylabel("Amplitude")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Activation Potential (Neuron A vs Neuron B)
    axes[2, 1].plot(
        results["activation_potential_a"], color="red", linewidth=2.0, label="Neuron A (0)")
    axes[2, 1].plot(
        results["activation_potential_b"], color="limegreen", linewidth=2.0, label="Neuron B (1)"
    )
    axes[2, 1].set_title("Activation Potential")
    axes[2, 1].set_xlabel("Samples")
    axes[2, 1].set_ylabel("Amplitude (V)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res = run_potencial_activacion_quian_quiroga(
        sampling_freq=24000.0,
        samples=48000,
        noise_level=0.2,
        num_spikes=100,
        spike_length=64,
        threshold_factor=4.0,
        wavelet_type="db4",
        seed=42,
    )
    print(f"Estimated sigma_n (MAD): {res['sigma_n']:.6f}")
    print(f"Threshold: {res['threshold']:.6f}")
    print(f"Size of DWT Coefficient Matrix: {res['dwt_coeff_matrix'].shape}")
    print(f"Centroid 1 (Neuron A): {res['centroid_1']}")
    print(f"Centroid 2 (Neuron B): {res['centroid_2']}")
    plot_front_panel(res)
