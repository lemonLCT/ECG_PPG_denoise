from __future__ import annotations

from pathlib import Path

import _pickle as pickle
import numpy as np

from . import Prepare_NSTDB, Prepare_QTDatabase


def _resolve_data_root(data_root: str | Path | None = "D:/Code/data/QT/") -> Path:
    if data_root is not None:
        return Path(data_root).resolve()
    return Path(__file__).resolve().parents[1] / "db" / "QTDB"
    # return "D:/Code/data/QT/"

def Data_Preparation(noise_version: int = 1, data_root: str | Path | None = None):
    data_root = _resolve_data_root(data_root)
    qt_pickle_path = data_root / "QTDatabase.pkl"
    noise_pickle_path = data_root / "NoiseBWL.pkl"

    print("Getting the Data ready ... ")

    # The seed is used to ensure the ECG always have the same contamination level.
    seed = 1234
    np.random.seed(seed=seed)

    Prepare_QTDatabase.prepare(
        qt_path=data_root / "qt-database-1.0.0",
        output_path=qt_pickle_path,
    )
    Prepare_NSTDB.prepare(
        nstdb_path=data_root / "mit-bih-noise-stress-test-database-1.0.0",
        output_path=noise_pickle_path,
    )

    with qt_pickle_path.open("rb") as input_file:
        qtdb = pickle.load(input_file)

    with noise_pickle_path.open("rb") as input_file:
        nstdb = pickle.load(input_file)

    [bw_signals, _, _] = nstdb
    bw_signals = np.array(bw_signals)

    bw_noise_channel1_a = bw_signals[0 : int(bw_signals.shape[0] / 2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0] / 2) : -1, 0]
    bw_noise_channel2_a = bw_signals[0 : int(bw_signals.shape[0] / 2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0] / 2) : -1, 1]

    if noise_version == 1:
        noise_test = bw_noise_channel2_b
        noise_train = bw_noise_channel1_a
    elif noise_version == 2:
        noise_test = bw_noise_channel1_b
        noise_train = bw_noise_channel2_a
    else:
        raise ValueError("noise_version 只能是 1 或 2")

    beats_train = []
    beats_test = []
    test_set = [
        "sel123",
        "sel233",
        "sel302",
        "sel307",
        "sel820",
        "sel853",
        "sel16420",
        "sel16795",
        "sele0106",
        "sele0121",
        "sel32",
        "sel49",
        "sel14046",
        "sel15814",
    ]

    skip_beats = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())

    for signal_name in qtdb_keys:
        for beat in qtdb[signal_name]:
            beat_np = np.zeros(samples)
            beat_sq = np.array(beat)

            init_padding = 16
            if beat_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            beat_np[init_padding : beat_sq.shape[0] + init_padding] = beat_sq - (beat_sq[0] + beat_sq[-1]) / 2

            signal_id = signal_name.replace("\\", "/").split("/")[-1]
            if signal_id in test_set:
                beats_test.append(beat_np)
            else:
                beats_train.append(beat_np)

    sn_train = []
    sn_test = []

    noise_index = 0
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i, beat in enumerate(beats_train):
        noise = noise_train[noise_index : noise_index + samples]
        beat_max_value = np.max(beat) - np.min(beat)
        noise_max_value = np.max(noise) - np.min(noise)
        ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / ase
        signal_noise = beat + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    np.save(data_root / "rnd_test.npy", rnd_test)
    print("rnd_test shape: " + str(rnd_test.shape))

    for i, beat in enumerate(beats_test):
        noise = noise_test[noise_index : noise_index + samples]
        beat_max_value = np.max(beat) - np.min(beat)
        noise_max_value = np.max(noise) - np.min(noise)
        ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / ase
        signal_noise = beat + alpha * noise

        sn_test.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_test) - samples):
            noise_index = 0

    if len(sn_train) == 0 or len(beats_train) == 0:
        raise ValueError("训练集为空，请检查 QTDatabase 读取和 train/test 划分逻辑。")
    if len(sn_test) == 0 or len(beats_test) == 0:
        raise ValueError("测试集为空，请检查 qtdb 键名格式是否与 test_set 一致。")

    x_train = np.array(sn_train)
    y_train = np.array(beats_train)
    x_test = np.array(sn_test)
    y_test = np.array(beats_test)

    x_train = np.expand_dims(x_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    dataset = [x_train, y_train, x_test, y_test]

    print(f"Skipped beats: {skip_beats}")
    print("Dataset ready to use.")
    return dataset
