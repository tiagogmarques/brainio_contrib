
import numpy as np
from brainio_collection.packaging import package_data_assembly
import scipy.io as sio
from brainio_base.assemblies import DataAssembly

DATA_DIR = '/braintree/data2/active/users/tmarques/bs_datasets/Ringach2002.mat'
ASSEMBLY_NAME = 'dicarlo.Marques2020_Ringach2002'
ORIENTATION_STIM_NAME = 'dicarlo.Marques2020_orientation'
PROPERTY_NAMES = ['baseline', 'maxdc', 'mindc', 'maxac', 'mod_ratio', 'circ_var', 'bandwidth', 'orth_pref_ratio',
                  'or_sel', 'cv_bw_ratio', 'opr_cv_diff', 'opr_bw_ratio']


def collect_data(data_dir):
    ringach2002 = sio.loadmat(data_dir)
    or_data = ringach2002['db']

    # Response magnitudes
    baseline = or_data['spont'][0, 0].T
    n_neuroids = baseline.shape[0]

    maxdc = or_data['maxdc'][0, 0].T
    mindc = or_data['mindc'][0, 0].T
    maxac = or_data['maxfirst'][0, 0].T
    mod_ratio = maxac / maxdc

    # Orientation tuning properties
    circ_var = or_data['orivar'][0, 0].T
    bandwidth = or_data['bw'][0, 0].T
    bandwidth[bandwidth > 90] = np.nan
    orth_pref_ratio = or_data['po'][0, 0].T

    or_sel = np.ones((n_neuroids, 1))
    or_sel[np.isnan(bandwidth)] = 0

    # Orientation tuning properties covariances
    cv_bw_ratio = circ_var / bandwidth
    opr_cv_diff = orth_pref_ratio - circ_var
    opr_bw_ratio = orth_pref_ratio/bandwidth

    # Bins
    maxdc_bins = np.logspace(0, 3, 10, base=10)
    maxac_bins = np.logspace(0, 3, 10, base=10)
    mindc_bins = np.logspace(-1-1/3, 2, 11, base=10)
    baseline_bins = np.logspace(-1-1/3, 2, 11, base=10)
    mod_ratio_bins = np.linspace(0, 2, 11)

    circ_var_bins = np.linspace(0, 1, num=14)
    bandwidth_bins = np.linspace(0, 90, num=18)
    orth_pref_ratio_bins = np.linspace(0, 1, num=14)
    or_sel_bins = np.linspace(0, 1, num=3)

    cv_bw_ratio_bins = np.logspace(-3, 0, num=16, base=10)
    opr_bw_ratio_bins = np.logspace(-3, 0, num=16, base=10)
    opr_cv_diff_bins = np.linspace(-1, 1, num=20)

    # Create DataAssembly with single neuronal properties and bin information
    assembly = np.concatenate((baseline, maxdc, mindc, maxac, mod_ratio, circ_var, bandwidth, orth_pref_ratio, or_sel,
                               cv_bw_ratio, opr_cv_diff, opr_bw_ratio), axis=1)

    # Filters neurons with weak responses
    good_neuroids = maxdc > baseline + 5
    assembly = assembly[np.argwhere(good_neuroids)[:, 0], :]

    assembly = DataAssembly(assembly, coords={'neuroid_id': ('neuroid', range(assembly.shape[0])),
                                              'region': ('neuroid', ['V1'] * assembly.shape[0]),
                                              'neuronal_property': PROPERTY_NAMES},
                            dims=['neuroid', 'neuronal_property'])

    assembly.attrs['number_of_trials'] = 40

    for p in assembly.coords['neuronal_property'].values:
        assembly.attrs[p+'_bins'] = eval(p+'_bins')

    return assembly


def main():
    assembly = collect_data(DATA_DIR)
    assembly.name = ASSEMBLY_NAME
    print('Packaging assembly')
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=ORIENTATION_STIM_NAME,
                          assembly_class='PropertyAssembly', bucket_name='brainio.dicarlo')


if __name__ == '__main__':
    main()

