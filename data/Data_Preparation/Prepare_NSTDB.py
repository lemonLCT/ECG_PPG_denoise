from __future__ import annotations

from pathlib import Path

import _pickle as pickle
import wfdb


def prepare(nstdb_path: str | Path, output_path: str | Path) -> None:
    nstdb_root = Path(nstdb_path)
    output_file = Path(output_path)

    bw_signals, bw_fields = wfdb.rdsamp(str(nstdb_root / "bw"))
    em_signals, em_fields = wfdb.rdsamp(str(nstdb_root / "em"))
    ma_signals, ma_fields = wfdb.rdsamp(str(nstdb_root / "ma"))

    for key, value in bw_fields.items():
        print(key, value)

    for key, value in em_fields.items():
        print(key, value)

    for key, value in ma_fields.items():
        print(key, value)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as output:
        pickle.dump([bw_signals, em_signals, ma_signals], output)
    print("=========================================================")
    print(f"MIT BIH dataset noise stress test database (NSTDB) saved as pickle: {output_file}")
