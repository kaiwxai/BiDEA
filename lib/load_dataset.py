import os
import numpy as np

def load_st_dataset(dataset):
    if dataset == 'CHITaxi':
        data_path = os.path.join('./data/CHITaxi/CHITaxi_flow.npy')
        data = np.load(data_path)
        matrix_wr_in = np.load('./data/CHITaxi/CHITaxi_od_matrix_in_walk_random.npy')
        matrix_wr_out = np.load('./data/CHITaxi/CHITaxi_od_matrix_out_walk_random.npy')
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data, matrix_wr_in, matrix_wr_out
