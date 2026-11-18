import numpy as np
import time
from numba import njit

@njit
def _merge_hits(particle_id, deposits, sensor_idx):
    # sensor_idx has to be sorted
    particle_tracking = particle_id[0]
    sensor_tracking = sensor_idx[0]
    total_sensor_dep = 0.0

    sensor_idx_result = np.zeros_like(sensor_idx) - 1
    deposit_result = np.zeros_like(deposits) - 1
    particle_id_result = np.zeros_like(particle_id) - 1
    last = 0

    for i in range(len(deposits)):
        if i!=0:
            if sensor_idx[i] < sensor_idx[i-1]:
                raise RuntimeError('Alignment issue')

        if not (sensor_idx[i] == sensor_tracking and particle_id[i] == particle_tracking):
            # Accumulate data
            particle_id_result[last] = particle_tracking
            sensor_idx_result[last] = sensor_tracking
            deposit_result[last] = total_sensor_dep
            last += 1

            # Reset state
            sensor_tracking = sensor_idx[i]
            particle_tracking = particle_id[i]
            total_sensor_dep = 0.0

        total_sensor_dep += deposits[i]

    sensor_idx_result[last] = sensor_tracking
    particle_id_result[last] = particle_tracking
    deposit_result[last] = total_sensor_dep

    return particle_id_result, sensor_idx_result, deposit_result

def merge_hits(particle_idx, sensor_idx, deposit):
    sort_idx = np.argsort(particle_idx)
    particle_idx_s = particle_idx[sort_idx]
    deposit_s = deposit[sort_idx]
    sensor_idx_s = sensor_idx[sort_idx]

    sort_idx = np.argsort(sensor_idx_s)
    particle_idx_s = particle_idx_s[sort_idx]
    deposit_s = deposit_s[sort_idx]
    sensor_idx_s = sensor_idx_s[sort_idx]


    particle_id_result, sensor_idx_result, deposit_result = _merge_hits(particle_idx_s, deposit_s, sensor_idx_s)
    filt = sensor_idx_result != -1
    particle_id_result = particle_id_result[filt]
    sensor_idx_result = sensor_idx_result[filt]
    deposit_result = deposit_result[filt]

    return particle_id_result, sensor_idx_result, deposit_result
