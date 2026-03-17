from __future__ import annotations

import glob
import math
from pathlib import Path

import _pickle as pickle
import numpy as np
import wfdb
from scipy.signal import resample_poly


def prepare(qt_path: str | Path, output_path: str | Path) -> None:
    qt_root = Path(qt_path)
    output_file = Path(output_path)

    new_fs = 360
    names_path = glob.glob(str(qt_root / "*.dat"))
    qt_database_signals: dict[str, list[np.ndarray]] = {}

    for signal_path in names_path:
        signal_file = Path(signal_path)
        register_name = signal_file.stem
        signal, fields = wfdb.rdsamp(str(signal_file.with_suffix("")))
        qu = len(signal)

        ann = wfdb.rdann(str(signal_file.with_suffix("")), "pu1")
        ann_type = ann.symbol
        ann_samples = ann.sample

        ann_type_np = np.array(ann_type)
        p_idx = ann_samples[ann_type_np == "p"]
        s_idx = ann_samples[ann_type_np == "("]
        r_idx = ann_samples[ann_type_np == "N"]

        ind = np.zeros(len(p_idx))
        for idx, p_sample in enumerate(p_idx):
            arr = np.where(p_sample > s_idx)[0]
            ind[idx] = arr[-1]

        ind = ind.astype(np.int64)
        p_start = s_idx[ind]
        p_start = p_start - int(0.04 * fields["fs"])

        aux_sig = signal[0:qu, 0]

        beats = []
        for idx in range(len(p_start) - 1):
            remove = (r_idx > p_start[idx]) & (r_idx < p_start[idx + 1])
            if np.sum(remove) < 2:
                beats.append(aux_sig[p_start[idx] : p_start[idx + 1]])

        beats_resampled = []
        for beat in beats:
            length = math.ceil(len(beat) * new_fs / fields["fs"])
            norm_beat = list(reversed(beat)) + list(beat) + list(reversed(beat))
            resampled = resample_poly(norm_beat, new_fs, fields["fs"])
            resampled = resampled[length - 1 : 2 * length - 1]
            beats_resampled.append(resampled)

        qt_database_signals[register_name] = beats_resampled

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as output:
        pickle.dump(qt_database_signals, output)
    print("=========================================================")
    print(f"MIT QT database saved as pickle file: {output_file}")
