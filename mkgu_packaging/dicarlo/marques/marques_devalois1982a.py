
import numpy as np
import pandas as pd
from marques_utils import gen_sample
from brainio_collection.packaging import package_data_assembly
from brainio_base.assemblies import DataAssembly

DATA_DIR = '/braintree/data2/active/users/tmarques/bs_datasets/or_po_DeValois1982.csv'
ASSEMBLY_NAME = 'dicarlo.Marques_devalois1982a'
ORIENTATION_STIM_NAME = 'dicarlo.Marques2020_orientation'


def collect_data(data_dir):
    devalois1982a = pd.read_csv(data_dir, header=None)

    # Preferred orientation data
    pref_or_hist = np.squeeze(np.fliplr(np.roll(devalois1982a[[1]].values.T, 1)))
    pref_or_bins = np.linspace(-22.5, 157.5, 5)

    pref_or = gen_sample(pref_or_hist, pref_or_bins)

    # Create DataAssembly with single neuronal properties and bin information
    assembly = DataAssembly(pref_or, coords={'neuroid_id': ('neuroid', range(pref_or.shape[0])),
                                             'region': ('neuroid', ['V1'] * pref_or.shape[0]),
                                             'neuronal_property': ['pref_or']},
                            dims=['neuroid', 'neuronal_property'])

    assembly.attrs['number_of_trials'] = 20

    for p in assembly.coords['properties'].values:
        assembly.attrs[p+'_bins'] = eval(p+'_bins')

    return assembly


def main():
    assembly = collect_data(DATA_DIR)
    assembly.name = ASSEMBLY_NAME
    print('Packaging assembly')
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=ORIENTATION_STIM_NAME,
                          assembly_class='PropertiyAssembly', bucket_name='brainio.dicarlo')


if __name__ == '__main__':
    main()

