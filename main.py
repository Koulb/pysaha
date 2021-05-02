import numpy as np
from mendeleev import element
from scipy.optimize import root_scalar
from nist_data import prepare_levels
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar

def calculate_pressure(elem_name, tempreature, relative_density=1e-3):
    # elem_name = 'Al'
    elem = element(elem_name)
    rho = relative_density*elem.density   #2.7 g/cm^3
    mass = elem.atomic_weight # 27 g/mol
    Z = elem.atomic_number #13.0
    elem_data = np.load("test_files/nist_data_Al.npz")
    # Saha model


    Avogadro = 6.022140857e+23 # N/mol
    aVol = 5.2917720859e-9**3
    hartree = 13.605693009*2 # eV
    e0       = 1.60217662e-19 #
    kB       = 1.38064853e-23  # J/K
    eV       = e0/kB # K


    aEnrg    = kB*hartree*eV
    aPres    = aEnrg/aVol

    V = mass/(Avogadro*rho*aVol)
    r0 = (3.0*V/4.0/np.pi)**(1.0/3.0)

    T = tempreature / hartree
    j_max = Z

    # key = elem_name + '_' + str(j_max)
    # energy_value = elem_data[key][1]
    #
    # print(energy_value)
    # exit()
    def ionization_energy(j):
        return elem.ionenergies.get(j)/ hartree

    # print(ionization_energy(1))
    # exit()
    def excited_energy(j, s):
        key = elem_name + '_' + str(j)
        energy_value = elem_data[key][1][s]
        # if energy_value < ionization_energy(j):
        return energy_value / hartree
        # else:
        #     raise ValueError('Energy is higher than Ionization energy')

    # print(excited_energy(0,100))
    # print(ionization_energy(0))
    # exit()

    def g(j, s):
        key = elem_name + '_' + str(j)
        g_value = elem_data[key][0][s]
        # if energy_value < ionization_energy(j):
        return g_value
        # else:
        #     raise ValueError('Energy is higher than Ionization energy')


    def statsum(j):
        key = elem_name + '_' + str(j)
        s_max = len(elem_data[key][1])
        sum = g(j, 0)
        temp = ionization_energy(j)

        for s in range(1, s_max):
            if ((ionization_energy(j)  - excited_energy(j, s)) <= T):
                # print(s ,' is ionization barier')
                break

            if (ionization_energy(j) <= excited_energy(j, s)): break
            temp2 = excited_energy(j, s)
            sum += g(j, s) * np.exp(-(excited_energy(j, s) - excited_energy(j, 0)) / T)

        return sum


    def phi(j,Z_temp):
        factor = (2 / 3) * np.sqrt(2 / np.pi) * r0 ** 3 * T ** (3.0 / 2.0)
        n_e = 3 * Z_temp / (4 * np.pi * r0**3.0)
        rD = np.sqrt(T /n_e)
        dI = j  / rD #(Z - j) ??
        # print(dI)

        if (ionization_energy(j) - dI) / T >= 30:
            print(j, 'is to high ')
            return 0.0

        result = (statsum(j + 1) / statsum(j)) * np.exp(- (ionization_energy(j) - dI) / T)

        return factor * result


    # def x_j(j):
    #     return j
    #
    #
    # def x(j, s):
    #     result = (x_j(j) / statsum(j)) * g(j, s) * np.exp(-(excited_energy(j, s) - (excited_energy(j, 0)) / T))
    #     return result


    def a(j, Z_temp):
        product = 1.0
        for k in range(1,j):
            # print(k)
            product *= phi(k,Z_temp) / Z_temp
            if product == 0.0:
                print(k, ' is zero')
                break
            #print(k, product)
        return product

    def Z0_resolve(Z_temp):
        sum1=0.0
        sum2=1.0
        for j in range(1, j_max+1):
            # print(j)
            sum1 += j*a(j,Z_temp)
            sum2 += a(j,Z_temp)
            # print(sum1,sum2)
        # print('sum1/sum2 =', sum1/sum2)
        return  Z_temp - sum1/sum2

    # x = np.linspace(1, 13, 13)
    # y = np.array([Z0_resolve(x_i) for x_i in x])


    #bar = IncrementalBar('Test plot', max = len(x))
    # y = []
    # for x_i in x:
    #     y.append(Z0_resolve(x_i))
    #     print('ready')
    #     # bar.next()

    #bar.finish()
    # y = np.array(y)
    # plt.plot(x,y)
    # plt.show()
    #
    # exit()

    def Z0_value():
        Z0 = root_scalar(
            f=Z0_resolve,
            method='brentq',
            bracket=(0.1, 2*Z),
            rtol=1e-3
        ).root
        return Z0

    P  = (Z0_value() + 1)
    return  P

elem_name = 'Al'

# print(calculate_pressure(elem_name, 10))
# exit()

# T = [10,50, 100,500, 1000]
NpointsT = 15
T = 10 ** np.linspace(np.log10(1) , np.log10(1000), NpointsT)# log scale

P = np.array([calculate_pressure(elem_name, T_i)for T_i in T])

plt.plot(T,P, '-x')
plt.grid()
plt.xscale('log')
plt.show()
print("Plot is ready")
print(P)


# print('rho/rho0 =', rho/elem.density)
# print('T =', T)
# print('P =',P)
# print('P * aPres =',P * aPres)
