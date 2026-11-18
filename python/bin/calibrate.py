"""Energy calibration of the toy calorimeter (python/notebooks/calibrate.ipynb as a script).

  python3 calibrate.py --sensor-data <sims>/sensor_data.npz --em <photon sims> --had <pion sims> --out calib.npz
                       [--cut 0.0009] [--calo v4]

Inputs are simulation libraries from run_simulations.py with simtype singlepart_face (the sim_calibration_dataset_em*
and sim_calibration_dataset_had* sections of configuration/production.ini: photons and charged pions, 1.7 < eta < 2.8,
filter_deposited=True). Steps, as in the notebook:
  1. sensors_scaling = (1 + pre-absorber thickness / sensor thickness) * 2.9 for every sensor
  2. g_em: least-squares fit of the photon energy to the summed scaled deposits
  3. g_had: the pion energy minus the EM-calibrated deposit in the EM layers, fitted to the summed scaled deposits in
     the hadronic layers (events with 0.5 < ratio < 2 and more than 5 GeV left for the hadronic part)
  4. sensors_scaling *= g_em in the EM layers (active layer < 2 x EE layers) and g_had in the hadronic layers
Writes the sensor table with sensors_scaling (the calibration file create_dataset.py reads) plus g_em, g_had and the
response of the calibration samples. The constants have to be redone for every Geant4 version and detector variant.
"""
import argparse
import time

import numpy as np

from calo.calo_generator import CaloGenerator
from calo.event_generator import EventGenerator
from calo.shards import sim_library


def deposits(gen, sims, limit):
    out, true = [], []
    for i in range(min(limit, sims.get_total())):
        sim, gun = sims.get_element(i)
        gen.add([sim])
        out.append(np.sum(gen.process()['rechit_energy']))
        true.append(gun['energy'])
    return np.array(out, np.float64), np.array(true, np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sensor-data', required=True)
    ap.add_argument('--em', required=True)
    ap.add_argument('--had', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cut', type=float, default=0.0009)
    ap.add_argument('--limit', type=int, default=1000000, help='use at most this many simulations of each sample')
    ap.add_argument('--global-scale', type=float, default=2.9)
    a = ap.parse_args()

    with np.load(a.sensor_data) as f:
        sd = {k: f[k] for k in f.files}
    em_layers = sd['sensors_active_layer_num'] < 2 * CaloGenerator().num_ee_layers
    base = (1 + sd['sensors_pre_absorber_thickness'] / sd['sensors_thickness']) * a.global_scale
    noise = ('type_a', 0, 5.2e-13)                         # negligible noise, as in the notebook
    kw = dict(num_hits_cut=3, reduce=True, area_normed_cut=True, merge_closeby_particles=False, collect_truth=False)

    def gen(scaling, cut):
        return EventGenerator(dict(sd, sensors_scaling=scaling), noise, cut=cut, **kw)

    em, had = sim_library(a.em), sim_library(a.had)
    print("calibration samples: %d EM, %d hadronic; %d sensors" % (em.get_total(), had.get_total(), len(base)))

    t = time.time()
    em_dep, em_true = deposits(gen(base, a.cut), em, a.limit)
    g_em = float(np.linalg.lstsq(em_dep[:, None], em_true[:, None], rcond=None)[0][0, 0])
    print("EM calibration constant g_em = %.5f (%d photons, %.0f s)" % (g_em, len(em_dep), time.time() - t))

    t = time.time()
    em_part, had_true = deposits(gen(np.where(em_layers, base * g_em, 0.0), 0.0), had, a.limit)
    had_part, _ = deposits(gen(np.where(em_layers, 0.0, base), 0.0), had, a.limit)
    left = had_true - em_part
    with np.errstate(divide='ignore', invalid='ignore'):
        r = had_part / left
    sel = (r > 0.5) & (r < 2) & (left > 5)
    g_had = float(np.linalg.lstsq(had_part[sel][:, None], left[sel][:, None], rcond=None)[0][0, 0])
    print("Had calibration constant g_had = %.5f (%d of %d pions used, %.0f s)" % (g_had, sel.sum(), len(sel),
                                                                                  time.time() - t))

    scaling = base * np.where(em_layers, g_em, g_had)
    em_resp, _ = deposits(gen(scaling, a.cut), em, a.limit)
    had_resp, _ = deposits(gen(scaling, a.cut), had, a.limit)
    for name, d, e in (("photons", em_resp, em_true), ("pions", had_resp, had_true)):
        q = d / e
        print("calibrated response %-8s mean %.3f, median %.3f, rms/mean %.3f" % (name, q.mean(), np.median(q),
                                                                                    q.std() / q.mean()))
    np.savez(a.out, **dict(sd, sensors_scaling=scaling), g_em=g_em, g_had=g_had, cut=a.cut,
             global_scale=a.global_scale, em_true=em_true, em_response=em_resp, had_true=had_true,
             had_response=had_resp)
    print("wrote", a.out)


if __name__ == '__main__':
    main()
