"""Write the sensor table of a calorimeter variant (minicalo.get_sensor_data()) as an .npz file.

  python3 extract_sensor_data.py <output.npz> [--calo v3|v4]

run_simulations.py writes the same table to <output_path>/sensor_data.npz; this script makes it without simulating.
"""
import argparse
import json
import os

import numpy as np

from calo.calo_generator import CaloV3Generator, CaloV4Generator
from calo.config import ensure_tls, pythia_data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output_file')
    ap.add_argument('--calo', default='v3', choices=['v3', 'v4'])
    a = ap.parse_args()
    import minicalo
    spec = {'v3': CaloV3Generator, 'v4': CaloV4Generator}[a.calo]().generate()
    minicalo.initialize(json.dumps(spec), pythia_data(), False, 1, 2, 3, 4)  # third argument is collect_full_data
    sd = {k: np.asarray(v) for k, v in minicalo.get_sensor_data().items()}
    out = a.output_file if a.output_file.endswith('.npz') else a.output_file + '.npz'
    np.savez(out + '.tmp.npz', **sd)
    os.replace(out + '.tmp.npz', out)
    print("Sensor data (%d sensors) saved to %s for the %s calorimeter." % (len(sd['sensors_x']), out, a.calo))


if __name__ == '__main__':
    ensure_tls()
    main()
