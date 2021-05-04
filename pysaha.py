"""

Saha model

Scipt allows to calculate pressure using Saha model. All notation is taken from reference [1].
Statistical sums' cutoff technique are taken from reference [2].
See https://github.com/Koulb/pysaha/blob/master/README.md for more details.

This file can also be imported as a module and contains the following
functions:

    * calculate_pressure - returns pressure in atomic units for given element
    * calculate_pressure_array - returns array of pressure values in atomic units for given element

"""

import numpy as np
from mendeleev import element
from scipy.optimize import root_scalar


def calculate_pressure(element_name, tempreature, relative_density=1e-3):
    """
    Returns pressure in atomic units for given element_name (i.e. 'Al'), temperature in eV
    and relative_density rho/rho0 where rho0 is density near room temperature
    """

    elem = element(element_name)
    rho = relative_density * elem.density
    mass = elem.atomic_weight
    Z = elem.atomic_number

    path_to_data = "test_files/nist_data" + "_" + element_name + ".npz"
    try:
        elem_data = np.load(path_to_data)
    except IOError:
        from nist_data import prepare_levels
        prepare_levels(element_name)
        elem_data = np.load(path_to_data)

    avogadro = 6.022140857e+23  # N/mol
    a_vol = 5.2917720859e-9 ** 3
    hartree = 13.605693009 * 2  # eV
    e0 = 1.60217662e-19
    k_b = 1.38064853e-23  # J/K
    e_v = e0 / k_b  # K
    a_enrg = k_b * hartree * e_v
    a_pres = a_enrg / a_vol

    v = mass / (avogadro * rho * a_vol)
    r0 = (3.0 * v / 4.0 / np.pi) ** (1.0 / 3.0)

    T = tempreature / hartree
    j_max = Z

    def ionization_energy(j):
        return elem.ionenergies.get(j+1) / hartree

    def excited_energy(j, s):
        key = element_name + '_' + str(j)
        energy_value = elem_data[key][1][s]

        return energy_value / hartree

    def g(j, s):
        key = element_name + '_' + str(j)
        g_value = elem_data[key][0][s]

        return g_value

    def statsum(j):
        if j == j_max:
            return 1.0

        key = element_name + '_' + str(j)
        s_max = len(elem_data[key][1])
        statsum_value = g(j, 0)

        for s in range(1, s_max):
            if (ionization_energy(j) - excited_energy(j, s)) <= T:
                break
            if ionization_energy(j) <= excited_energy(j, s):
                break
            statsum_value += g(j, s) * \
                np.exp(-(excited_energy(j, s) - excited_energy(j, 0)) / T)

        return statsum_value

    def phi(j, z_temp):
        factor = (2 / 3) * np.sqrt(2 / np.pi) * r0 ** 3 * T ** (3.0 / 2.0)
        n_e = 3 * z_temp / (4 * np.pi * r0 ** 3.0)
        r_d = np.sqrt(T / n_e)
        delta_ionization_energy = j / r_d

        if (ionization_energy(j) - delta_ionization_energy) / T >= 30:
            return 0.0

        result = (statsum(j + 1) / statsum(j)) * \
            np.exp(- (ionization_energy(j) - delta_ionization_energy) / T)

        return factor * result

    def a(j, Z_temp):
        product = 1.0
        for k in range(0, j):
            product *= phi(k, Z_temp) / Z_temp
            if product == 0.0:
                break
        return product

    def z0_resolve(Z_temp):
        sum1 = 0.0
        sum2 = 1.0
        for j in range(1, j_max+1):
            sum1 += j * a(j, Z_temp)
            sum2 += a(j, Z_temp)
        return Z_temp - sum1 / sum2

    def z0_value():
        z0 = root_scalar(
            f=z0_resolve,
            method='brentq',
            bracket=(0.1, 2 * Z),
            rtol=1e-3
        ).root
        return z0

    pressure = (z0_value()+1) * (T/v)
    return pressure


def calculate_pressure_array(element_name, tempreature, relative_density=1e-3):
    """Returns array of pressure values for each T in eV in temperature array"""

    return np.array([calculate_pressure(element_name, T_i) for T_i in tempreature])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    elem_name = 'Al'

    # print(calculate_pressure(elem_name, 1000))
    # exit()

    NpointsT = 30
    T = 10 ** np.linspace(np.log10(1), np.log10(1000), NpointsT)  # log scale

    P = np.array([calculate_pressure(elem_name, T_i) for T_i in T])

    print("Plot is ready")
    print(P)

    plt.plot(T, P, '-x')
    plt.grid()
    plt.xscale('log')
    plt.savefig("Pressure_saha.pdf")
    plt.show()
