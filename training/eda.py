#!/usr/bin/env python3
"""
Exploratory Data Analysis — TinyML Gesture Classifier
Analiza la distribución de datos, señales IMU y justifica el tamaño de ventana.

Uso:
    python eda.py                        # muestra plots interactivos
    python eda.py --save                 # guarda plots en ../eval/eda/
    python eda.py --data-dir ruta/data   # directorio de datos personalizado
    python eda.py --overlap 0.5          # ratio de solapamiento para visualización
"""
import argparse
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── scienceplots es opcional ──────────────────────────────────────────────────
try:
    import scienceplots  # noqa: F401
    plt.style.use("science")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURE_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_label(raw: str) -> str:
    """Normaliza etiqueta a ASCII uppercase (Círculo → CIRCULO)."""
    nfkd = unicodedata.normalize("NFKD", raw)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper()


def load_recordings(data_dir: str) -> list[pd.DataFrame]:
    """
    Carga todos los CSVs de data_dir.
    Añade columnas 'label' y 'subject' a cada DataFrame.
    """
    recordings = []
    for root, _, files in os.walk(data_dir):
        subject = os.path.basename(root)
        for fname in sorted(files):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(root, fname)
            label = normalize_label(fname.split("_")[0])
            try:
                df = pd.read_csv(fpath)
                if not all(c in df.columns for c in FEATURE_COLS):
                    print(f"  [WARN] columnas faltantes: {fpath}")
                    continue
                if "timestamp_ms" not in df.columns:
                    df["timestamp_ms"] = (
                        np.arange(len(df)) * 10.0
                    )  # 100 Hz → 10 ms/muestra
                df["label"]   = label
                df["subject"] = subject
                recordings.append(df)
            except Exception as e:
                print(f"  [WARN] {fpath}: {e}")

    if not recordings:
        print(f"ERROR: No se encontraron CSVs válidos en '{data_dir}'.")
        sys.exit(1)

    return recordings


def gesture_cycle_duration(df: pd.DataFrame, raw: bool = False):
    """
    Duración media de un ciclo de gesto (ms) mediante detección de picos.
    Si raw=True devuelve el array de duraciones individuales.
    """
    acc_mag   = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    threshold = float(np.percentile(acc_mag, 10))
    peaks, _  = find_peaks(acc_mag, height=threshold, distance=20)
    if len(peaks) > 1:
        deltas = np.diff(df["timestamp_ms"].iloc[peaks].values)
        return deltas if raw else float(np.mean(deltas))
    return np.array([]) if raw else 0.0


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_class_distribution(consolidated: pd.DataFrame, save_dir: str | None) -> None:
    """Distribución de clases y señales de muestra."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_list   = axes.flat

    counts = consolidated["label"].value_counts()
    counts.plot.bar(ax=ax_list[0], color="steelblue", edgecolor="white")
    ax_list[0].set_title("Distribución de Clases")
    ax_list[0].set_ylabel("N° de muestras")
    ax_list[0].tick_params(axis="x", rotation=0)

    recordings = [
        grp for _, grp in consolidated.groupby(["subject", "label"], sort=False)
    ]
    indices = [0, len(recordings) // 2, len(recordings) - 1]

    for i, idx in enumerate(indices):
        ax    = ax_list[i + 1]
        sdf   = recordings[idx]
        t0    = sdf["timestamp_ms"].iloc[0]
        mask  = (sdf["timestamp_ms"] - t0) <= 400
        for col in ["ax", "ay", "az"]:
            ax.plot(
                sdf["timestamp_ms"][mask] - t0,
                sdf[col][mask],
                label=col,
                alpha=0.8,
            )
        ax.set_title(f"Señal: {sdf['label'].iloc[0]} ({sdf['subject'].iloc[0]})")
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Aceleración")
        ax.legend(fontsize="small")

    fig.suptitle("Distribución de Clases y Señales de Muestra", fontsize=13)
    plt.tight_layout()
    _show_or_save(fig, save_dir, "01_class_distribution.png")


def plot_feature_histograms(
    recordings: list[pd.DataFrame], save_dir: str | None
) -> None:
    """Histogramas por feature para el primer sujeto."""
    sample = recordings[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        ax.hist(sample[col], bins=30, color="steelblue", edgecolor="white")
        ax.set_title(f"Feature: {col}")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")

    subject = sample["subject"].iloc[0]
    fig.suptitle(f"Histogramas de Features — Sujeto: {subject}", fontsize=13)
    plt.tight_layout()
    _show_or_save(fig, save_dir, "02_feature_histograms.png")


def plot_gesture_peaks(df: pd.DataFrame, save_dir: str | None) -> None:
    """Detección de eventos de gesto sobre la magnitud normalizada de aceleración."""
    acc_mag   = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    acc_norm  = (acc_mag - acc_mag.mean()) / acc_mag.std()
    acc_final = np.abs(acc_norm)
    threshold = 1.2

    peaks, _ = find_peaks(
        acc_final, height=threshold, distance=50, prominence=0.5
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df["timestamp_ms"],
        acc_final,
        label="Magnitud normalizada (|z|)",
        color="mediumpurple",
        alpha=0.85,
    )
    ax.scatter(
        df["timestamp_ms"].iloc[peaks],
        acc_final.iloc[peaks],
        color="red",
        marker="x",
        s=70,
        label="Eventos detectados",
        zorder=5,
    )
    ax.axhline(threshold, color="orange", linestyle="--", label=f"Threshold: {threshold}")
    ax.set_title("Detección de Gestos — Señal Normalizada + Valor Absoluto")
    ax.set_xlabel("Tiempo (ms)")
    ax.set_ylabel("Amplitud (desv. estándar)")
    ax.legend()

    label   = df["label"].iloc[0]
    subject = df["subject"].iloc[0]
    fig.suptitle(f"Señal: {label} ({subject})", fontsize=11)
    plt.tight_layout()
    _show_or_save(fig, save_dir, "03_gesture_peaks.png")


def plot_cycle_duration_histogram(
    recordings: list[pd.DataFrame], save_dir: str | None
) -> None:
    """Histograma de duración de ciclos de gesto."""
    all_deltas = []
    for df in recordings:
        deltas = gesture_cycle_duration(df, raw=True)
        if isinstance(deltas, np.ndarray) and len(deltas):
            all_deltas.extend(deltas.tolist())

    if not all_deltas:
        print("  [WARN] No se pudieron detectar ciclos — omitiendo plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_deltas, bins=40, color="salmon", edgecolor="white")
    ax.axvline(
        float(np.mean(all_deltas)),
        color="darkred",
        linestyle="--",
        label=f"Media: {np.mean(all_deltas):.0f} ms",
    )
    ax.set_title("Histograma de Duración de Ciclos de Gesto")
    ax.set_xlabel("Duración (ms)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    plt.tight_layout()
    _show_or_save(fig, save_dir, "04_cycle_duration.png")


def compute_window_size(
    recordings: list[pd.DataFrame], overlap: float
) -> tuple[int, int]:
    """
    Calcula WINDOW_SIZE y STEP_SIZE empíricamente a partir de los ciclos de gesto.
    Retorna (window_size, step_size).
    """
    cycle_durations = [gesture_cycle_duration(df) for df in recordings]
    valid           = [d for d in cycle_durations if d > 0]

    if not valid:
        print("  [WARN] No se detectaron ciclos válidos — usando WINDOW_SIZE=100.")
        sample_rate_hz = 100.0
        window_size    = 100
    else:
        deltas         = np.diff(recordings[0]["timestamp_ms"].values[:100])
        sample_rate_hz = 1000.0 / float(np.mean(deltas))
        mean_ms        = float(np.mean(valid))
        window_size    = max(20, int(mean_ms * sample_rate_hz / 1000))

    step_size = max(1, int(window_size * (1 - overlap)))

    print(f"  Frecuencia estimada : {sample_rate_hz:.1f} Hz")
    print(f"  Duración media ciclo: {float(np.mean(valid)):.1f} ms" if valid else "  (sin ciclos válidos)")
    print(f"  WINDOW_SIZE         : {window_size} muestras")
    print(f"  STEP_SIZE           : {step_size} muestras (overlap {overlap*100:.0f}%)")

    return window_size, step_size


def plot_windowing_visualization(
    consolidated: pd.DataFrame,
    recordings: list[pd.DataFrame],
    window_size: int,
    step_size: int,
    save_dir: str | None,
) -> None:
    """Visualización de ventanas deslizantes sobre señales de muestra."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_list   = axes.flat
    time_ms   = 1000  # ms a mostrar

    counts = consolidated["label"].value_counts()
    counts.plot.bar(ax=ax_list[0], color="steelblue", edgecolor="white")
    ax_list[0].set_title("Distribución de Clases")
    ax_list[0].set_ylabel("N° de muestras")
    ax_list[0].tick_params(axis="x", rotation=0)

    win_colors = plt.cm.tab20.colors
    indices    = [0, len(recordings) // 2, len(recordings) - 1]

    for i, idx in enumerate(indices):
        ax    = ax_list[i + 1]
        sdf   = recordings[idx].reset_index(drop=True)
        t0    = sdf["timestamp_ms"].iloc[0]
        mask  = (sdf["timestamp_ms"] - t0) <= time_ms
        t_vis = sdf["timestamp_ms"][mask]
        n_vis = mask.sum()

        for col, color in zip(["ax", "ay", "az"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            ax.plot(t_vis - t0, sdf[col].values[:n_vis], color=color, label=col, alpha=0.85, lw=1.2)

        n_total = len(sdf)
        for w_i, start in enumerate(range(0, n_total - window_size + 1, step_size)):
            t_start = sdf["timestamp_ms"].iloc[start] - t0
            t_end   = sdf["timestamp_ms"].iloc[min(start + window_size - 1, n_total - 1)] - t0
            if t_start > time_ms:
                break
            t_end_clamped = min(t_end, time_ms)
            c = win_colors[w_i % len(win_colors)]
            ax.axvspan(t_start, t_end_clamped, alpha=0.12, color=c)
            ax.axvline(t_start, color=c, lw=0.8, linestyle="--", alpha=0.7)
            ax.text(
                t_start, ax.get_ylim()[0],
                f" w{w_i+1}", fontsize=6, color=c, va="bottom", rotation=90,
            )

        n_wins = len(range(0, n_total - window_size + 1, step_size))
        ax.set_xlim(0, time_ms)
        ax.set_title(
            f"'{sdf['label'].iloc[0]}' ({sdf['subject'].iloc[0]}) — "
            f"{n_wins} ventanas | size={window_size} | step={step_size}"
        )
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Aceleración")
        ax.legend(fontsize="small")

    plt.tight_layout()
    _show_or_save(fig, save_dir, "05_windowing.png")


def _show_or_save(fig: plt.Figure, save_dir: str | None, filename: str) -> None:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {path}")
        plt.close(fig)
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA para el clasificador de gestos TinyML."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(SCRIPT_DIR, "data"),
        help="Directorio raíz con los CSVs de entrenamiento (default: training/data).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guarda los plots en ../eval/eda/ en lugar de mostrarlos.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Ratio de solapamiento para visualización de ventanas (default: 0.5).",
    )
    args = parser.parse_args()

    save_dir = (
        os.path.join(SCRIPT_DIR, "..", "eval", "eda") if args.save else None
    )
    if not args.save:
        matplotlib.use("TkAgg" if "DISPLAY" in os.environ else "MacOSX")

    print("=" * 55)
    print("EDA — TinyML Gesture Classifier")
    print("=" * 55)
    print(f"  Datos  : {args.data_dir}")
    print(f"  Salida : {save_dir or 'interactivo'}")
    print()

    print("Cargando datos...")
    recordings   = load_recordings(args.data_dir)
    consolidated = pd.concat(recordings, ignore_index=True)
    print(f"  {len(recordings)} archivos | {len(consolidated):,} muestras | clases: {sorted(consolidated['label'].unique())}")
    print()

    print("Generando plots...")
    plot_class_distribution(consolidated, save_dir)
    plot_feature_histograms(recordings, save_dir)
    plot_gesture_peaks(recordings[0], save_dir)
    plot_cycle_duration_histogram(recordings, save_dir)

    print("\nCalculando tamaño de ventana óptimo...")
    window_size, step_size = compute_window_size(recordings, args.overlap)

    print("\nGenerando visualización de ventanas...")
    plot_windowing_visualization(consolidated, recordings, window_size, step_size, save_dir)

    print("\nListo.")


if __name__ == "__main__":
    main()
