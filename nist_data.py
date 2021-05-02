from nistasd import NISTLines, NISTASD
import numpy as np


def prepare_levels(name):
    nist = NISTLines(spectrum=name)
    energy_levels = nist.get_energy_level_data()
    data_dictionary = {}

    for number, ion_stage in enumerate(energy_levels):
        value_g, value_level = [], []
        df = energy_levels[ion_stage]

        for i in range(len(df)):
            value_g.append(df[i].get('g'))
            value_level.append(df[i].get('level (eV)'))

        values = np.array([value_g, value_level])
        ion_stage_name = name + '_' + str(number + 1)
        data_dictionary[ion_stage_name] = values

    np.savez("test_files/nist_data" + "_" + name + ".npz", **data_dictionary)

    return


if __name__ == '__main__':
    prepare_levels('Al')
    arr = np.load("test_files/nist_data_Al.npz")
    print(arr['Al_1'])
    print(arr['Al_10'])
