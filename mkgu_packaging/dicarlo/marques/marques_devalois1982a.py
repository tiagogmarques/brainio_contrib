
import numpy as np
from marques_utils import gen_sample
from brainio_collection.packaging import package_data_assembly
from brainio_base.assemblies import DataAssembly

ASSEMBLY_NAME = 'devalois.DeValois1982a'
ORIENTATION_STIM_NAME = 'dicarlo.Marques2020_orientation'


def collect_data():
    # Preferred orientation data
    preferred_orientation_hist = np.array([77, 49, 66, 54])
    preferred_orientation_bins = np.linspace(-22.5, 157.5, 5)

    preferred_orientation = gen_sample(preferred_orientation_hist, preferred_orientation_bins, scale='linear')

    # Create DataAssembly with single neuronal properties and bin information
    assembly = DataAssembly(preferred_orientation, coords={'neuroid_id': ('neuroid',
                                                                          range(preferred_orientation.shape[0])),
                                                           'region': ('neuroid', ['V1'] * preferred_orientation.shape[0]),
                                                           'neuronal_property': ['preferred_orientation']},
                            dims=['neuroid', 'neuronal_property'])

    assembly.attrs['number_of_trials'] = 20

    for p in assembly.coords['neuronal_property'].values:
        assembly.attrs[p+'_bins'] = eval(p+'_bins')

    return assembly


def main():
    assembly = collect_data()
    assembly.name = ASSEMBLY_NAME
    print('Packaging assembly')
    package_data_assembly(assembly, assembly_identifier=assembly.name, stimulus_set_identifier=ORIENTATION_STIM_NAME,
                          assembly_class='PropertyAssembly', bucket_name='brainio.contrib')


if __name__ == '__main__':
    main()

