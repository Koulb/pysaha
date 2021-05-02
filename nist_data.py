from nistasd import NISTLines, NISTASD
import numpy as np


def prepare_levels(name):
    nist = NISTLines(spectrum=name)
    energy_levels = nist.get_energy_level_data()

    # print("energy_levels.keys() = ", energy_levels.keys())


    # for ion_stage in energy_levels:
    #     print("Number of levels: {0} for {1}".format(len(energy_levels[ion_stage]), ion_stage))
    #     df = pd.DataFrame(energy_levels[ion_stage])
    #
    #
    #
    #     break
    # values = np.empty()

    data_dictionary = {}

    for number, ion_stage in enumerate(energy_levels):
        #g_factor, energy levels
        value_g, value_level = [], []

        df = energy_levels[ion_stage]#energy_levels['Al II']#pd.DataFrame(
        for i in range(len(df)):
            value_g.append(df[i].get('g'))
            value_level.append(df[i].get('level (eV)'))


        values = np.array([value_g, value_level])
        ion_stage_name = name + '_' + str(number + 1) #str(ion_stage.replace(' ', '_'))
        # print(number)
        # print(ion_stage_name)

        data_dictionary[ion_stage_name] = values
        # print(data_dictionary)
        # exit()

    # print(data_dictionary)
    np.savez("test_files/nist_data" + "_" + name + ".npz", **data_dictionary)


if __name__ == '__main__':
    prepare_levels('Al')
    arr =  np.load("test_files/nist_data_Al.npz")
    print(arr['Al_1'])
    print(arr['Al_10'])
    # print(len(df))



    # nist = NISTLines(spectrum='O', lower_wavelength=2., upper_wavelength=50., order=1)

    # Columns: [configuration, term, J, level(eV), reference, level splittings(eV), uncertainty(eV)]
    # Index: []
