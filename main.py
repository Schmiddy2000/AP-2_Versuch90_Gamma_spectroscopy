import pandas as pd
import chardet
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


# Loading the data:

folder_path = "/Users/lucas1/Desktop/Uni/Physiklabor A/Physiklabor für Anfänger Teil 2/Versuch 90 - " \
              "Gammaspektroskopie/Materialien/"

path_Na22 = folder_path + "NA22_1.txt"
path_Na22_background = folder_path + "NA22_Untergrund_1.txt"

path_Co60 = folder_path + "Co60_1.txt"
path_Co60_background = folder_path + "Co60_Untergrund_1.txt"

path_Cs137 = folder_path + "CS137_1.txt"
path_Cs137_background = folder_path + "NA22_Untergrund_1.txt"

path_unknown1 = folder_path + "Unbekannt1_M1.txt"
path_unknown1_background = folder_path + "Unbekannt1_Untergrund1.txt"

path_unknown1_v2 = folder_path + "Unbekannt1_M2.txt"
path_unknown1_v2_background = folder_path + "Unbekannt1_Untergrund1.txt"

path_unknown2 = folder_path + "Unbekannt2_M1.txt"
path_unknown2_background = folder_path + "Unbekannt2_Untergrund1.txt"

path_unknown3 = folder_path + "Unbekannt3_M1.txt"
path_unknown3_background = folder_path + "Unbekannt1_Untergrund1.txt"

path_unknown4 = folder_path + "Unbekannt4_M1.txt"
path_unknown4_background = folder_path + "Co60_Untergrund_1.txt"


# Transformation:
transformation_best_a = 19.105837666119562
transformation_best_a_error = 0.08439201148372207
transformation_best_c = -27.146238702736007
transformation_best_c_error = 4.2808739156513615

isotope_dict = {'U-236': [112.0], 'U-234': [120.0], 'Pb-214': [186.0, 241.0, 295.0, 351.0], 'Pb-212': [238.0],
                'Ac-228': [338.0, 794.0, 911.0, 964.0, 968.0, 1588.0], 'Tl-208': [510.0, 583.0, 860.0, 2614.0],
                'Bi-214': [609.0, 768.0, 934.0, 1120.0, 1238.0, 1377.0, 1407.0, 1729.0, 1764.0, 1847.0, 2118.0,
                           2204.0, 2293.0, 2447.0, 3053.0], 'Bi-212': [727.0],
                'Pa-234m': [1001.0, 1737.0, 1831.0, 1867.0, 1874.0, 1911.0, 1937.0], 'K-40': [1461.0]}


def r(x): return round(x, 3)


# Read the data from .tsv
def getDataframe(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    df = pd.read_csv(file_path, delimiter='\t', decimal=',', skiprows=1, header=None,
                     names=['Counts'], encoding=result['encoding'])

    return df


# Creates the x-axis as specified in the initial .tsv files
def getChannelNumbers(df_counts): return np.arange(0, len(df_counts) / 5, 0.2)


# Transform channel numbers to (predicted) energies (in keV)
def transformToEnergies(x_array, with_errors=False):
    best_energie = transformation_best_a * x_array + transformation_best_c

    if not with_errors:
        return best_energie
    else:
        best_energie_error = np.sqrt((x_array * transformation_best_a_error) ** 2
                                     + transformation_best_c_error ** 2)
        return best_energie, best_energie_error


# Gaussian error propagation for the 'mean'-energy from 'makeGaussian()'
def getEnergyError(energy, delta_energy):
    first = energy * transformation_best_a_error
    second = transformation_best_a * delta_energy

    return np.sqrt(first ** 2 + second ** 2 + transformation_best_c_error ** 2)


# Updated (and hopefully correct) version to calculate the energy errors
def getEnergyErrorByTransformingDeltaMean(delta_energy):
    a_plus = transformation_best_a + transformation_best_a_error
    c_plus = transformation_best_c + transformation_best_c_error

    energy_error = delta_energy * a_plus + c_plus

    return energy_error


# Calculates the corrected count rate (per second) with errors from both the measurement with and without the
# radiation source
def getCorrectedCountsWithErrors(file_path, background_file_path, measurement_time=75,
                                 background_measurement_time=150):
    time_ratio = measurement_time / background_measurement_time

    counts = getDataframe(file_path).to_numpy().flatten()
    background_counts = getDataframe(background_file_path).to_numpy().flatten()

    max_index = min(len(counts), len(background_counts))

    count_rate = counts / measurement_time
    background_count_rate = background_counts / background_measurement_time

    count_rate_error = np.sqrt(counts) / measurement_time
    adjusted_background_count_rate_error = np.sqrt(background_counts * time_ratio) / measurement_time

    corrected_count_rate = count_rate[:max_index] - background_count_rate[:max_index]
    corrected_count_rate_range = getChannelNumbers(corrected_count_rate)

    corrected_count_rate_error = np.sqrt(count_rate_error[:max_index] ** 2
                                         + adjusted_background_count_rate_error[:max_index] ** 2)

    return corrected_count_rate_range, corrected_count_rate, corrected_count_rate_error


# Function to describe a gaussian bell curve
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2)


# Function to draw gaussian bell curve and find its optimal parameters. Best guess has to be specified right above
# the 'plt.figure' statement
def makeGaussian(file_path, background_file_path, transform_to_energies=False, disable_print_output=False):
    x_data, y_data, y_data_errors = getCorrectedCountsWithErrors(file_path, background_file_path)

    if transform_to_energies:
        x_data = transformToEnergies(x_data)
        min_index = int(5 * ((min_channel - transformation_best_c) / transformation_best_a))
        max_index = int(5 * ((max_channel - transformation_best_c) / transformation_best_a)) + 1
    else:
        min_index = int(min_channel * 5)
        max_index = int(max_channel * 5) + 1

    x_data = x_data[min_index:max_index]
    y_data = y_data[min_index:max_index]

    if len(x_data) > 0 and len(y_data) > 0:
        mean_guess = x_data[int(np.argmax(y_data))]
        amplitude_guess = max(y_data)
        stddev_guess = FWHM / (2 * np.sqrt(2 * np.log(2)))
    else:
        print("WARNING: \nx_data and/or y_data are empty. Please check if the 'transform_to_energies' parameter is "
              "set correctly.\n")
        mean_guess = 1
        amplitude_guess = 1
        stddev_guess = FWHM / (2 * np.sqrt(2 * np.log(2)))

    initial_guesses = (amplitude_guess, mean_guess, stddev_guess)

    # (optionally) print run information:
    if not disable_print_output:
        print('Array indices:', str('[') + str(min_index) + ':' + str(int(max_index)) + str(']'))
        print('min_channel = ' + str(min_channel), 'max_channel = ' + str(max_channel), 'FWHM = ' + str(FWHM),
              sep=', ')
        print('Initial guesses:', '(' + str(r(initial_guesses[0])) + ', ' + str(r(initial_guesses[1])) + ', '
              + str(r(initial_guesses[2])) + ')')

    # Perform the curve fitting
    if len(x_data) > 0 and len(y_data) > 0:
        this_popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guesses)

        amplitude_opt, mean_opt, stddev_opt = this_popt

        amplitude_err = np.sqrt(pcov[0][0])
        mean_err = np.sqrt(pcov[1][1])
        stddev_err = np.sqrt(pcov[2][2])

        # (optionally) print the optimized parameters
        if not disable_print_output:
            print("Optimized Amplitude:", r(amplitude_opt), '+-', r(amplitude_err))
            print("Optimized Mean:", r(mean_opt), '+-', r(mean_err))
            print("Optimized Stddev:", r(stddev_opt), '+-', r(stddev_err))
            print('This yields E =', r(mean_opt), '+-', r(getEnergyErrorByTransformingDeltaMean(mean_err)))

        if transform_to_energies:
            this_x_lin = np.linspace(mean_opt - 100, mean_opt + 100, 1000)
        else:
            this_x_lin = np.linspace(mean_opt - 10, mean_opt + 10, 1000)
    else:
        return 1, (1, 1, 1)

    return this_x_lin, this_popt


# Create a visualisation of the .tsv files. Optionals let you choose the type of plot and other features
def createPlot(file_path, background_file_path, do_gaussian=False, show_errors=True, show_line=True,
               show_dots=False, show_bars=False, upper_x_lim=None, transform_to_energies=False):
    name_pre = file_path.split('/')
    name = name_pre[len(name_pre) - 1].split('.')[0]

    if show_bars:
        show_errors = False
        show_line = False

    plt.title('Spektrum der Kanäle von ' + name)
    plt.xlabel('Kanalnummer (in 0,2er Schritten)')
    plt.ylabel('Zahlrate in [s$^{-1}$]')

    x_range, count_rate, count_rate_error = getCorrectedCountsWithErrors(file_path, background_file_path)

    if transform_to_energies:
        x_range = transformToEnergies(x_range)
        plt.xlabel('Energie in [keV]')

    if show_errors:
        upper = count_rate + count_rate_error
        lower = count_rate - count_rate_error

        plt.fill_between(x_range, upper, lower, where=upper >= lower, interpolate=True, alpha=0.5,
                         label='Konfidenzband')

    if show_line:
        plt.plot(x_range, count_rate, lw=0.4, c='black', label='Verbindungslinie')

    if show_bars:
        bar_width = 1.05 * max(x_range) / len(x_range)
        print(bar_width)
        plt.bar(x_range, count_rate, width=bar_width, color='black', label='Korrigierte Zählrate')
        plt.ylim(0, None)

    if show_dots:
        plt.scatter(x_range, count_rate, s=1, label='Messwerte')

    if upper_x_lim is None:
        plt.xlim(min(x_range), max(x_range))
    else:
        plt.xlim(min(x_range), upper_x_lim)

    plt.legend()

    if do_gaussian:
        if transform_to_energies:
            x_lin, popt = makeGaussian(file_path, background_file_path, transform_to_energies)
        else:
            x_lin, popt = makeGaussian(file_path, background_file_path)

        plt.plot(x_lin, gaussian(x_lin, *popt), lw=1, ls='--', c='r')

    return None


# Plot with linear regression to find a relation between channel number and energy
def FitChannelsAndEnergies():
    channels = np.array([28.172, 68.392, 36.078, 62.395, 70.991])
    energies = np.array([511, 1277, 662, 1172.6, 1332.75])
    channel_error = np.array([0.027, 0.056, 0.021, 0.107, 0.116])
    energy_errors = np.array([1, 1, 1, 2, 2]) / np.sqrt(3)
    channel_lin = np.linspace(0, 140)

    def lin_model(x, a, c): return a * x + c

    popt_lin, pcov_lin = curve_fit(lin_model, channels, energies, sigma=energy_errors)

    best_a = popt_lin[0]
    best_c = popt_lin[1]
    # best_c = None

    best_a_err = np.sqrt(pcov_lin[0][0])
    best_c_err = np.sqrt(pcov_lin[1][1])
    # best_c_err = None

    upper = lin_model(channel_lin, best_a + best_a_err, best_c - best_c_err)
    lower = lin_model(channel_lin, best_a - best_a_err, best_c + best_c_err)

    plt.title('Kanalnummer gegen Energie')
    plt.xlabel('Kanalnummer (in 0,2er Schritten)')
    plt.ylabel('Energie in [keV]')

    plt.fill_between(channel_lin, upper, lower, where=upper >= lower, interpolate=True, color='pink', alpha=0.5)
    plt.fill_between(channel_lin, upper, lower, where=upper < lower, interpolate=True, color='pink', alpha=0.5,
                     label='Konfidenzband')

    plt.errorbar(channels, energies, xerr=channel_error, yerr=energy_errors, fmt='none',
                 capthick=0.8, capsize=5, elinewidth=0.8, ecolor='black', label='Fehler')

    plt.plot(channel_lin, lin_model(channel_lin, *popt_lin), lw=1, ls='--', c='black', label='Ausgleichsgerade')
    print('a =', best_a, '+-', best_a_err, '\nc =', best_c, '+-', best_c_err)

    plt.scatter(channels, energies, marker='x', c='magenta', label='Bestwerte')

    plt.xlim(22, 75)
    plt.ylim(400, 1400)
    plt.legend()

    plt.savefig('LinReg_Kanalnummer_gegen_Energie.png', dpi=300)

    return best_a, best_c, best_a_err, best_c_err


def plotElementVLines(element_identifiers):
    for element in element_identifiers:
        plt.vlines()


# for element in energy_dict:
#     for energy in energy_dict[element]:
#         plt.vlines(energy, 0, 3, ls='--', lw=0.5, colors='black')


# Set initial parameter guesses for 'makeGaussian()'
min_channel = 203
max_channel = 264.5
FWHM = 85


# Plotting:
plt.figure(figsize=(12, 5))

# FitChannelsAndEnergies()

createPlot(path_unknown1_v2, path_unknown1_v2_background, do_gaussian=True, transform_to_energies=True,
           show_bars=True)

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.95)

# plt.savefig('Co-66_gamma_spectrum.png', dpi=300)
plt.show()

# Na-22:
# --- Peak 1:
# Array indices: [127:156]
# min_channel = 25.4, max_channel = 31, FWHM = 3.4
# Initial guesses: (4.467, 28.2, 1.444)
# Optimized Amplitude: 4.373 +- 0.064
# Optimized Mean: 28.172 +- 0.027
# Optimized Stddev: 2.221 +- 0.043
# This yields E = 28.172 +- 23.57
# -- -- Energies:
# Array indices: [128:155]
# min_channel = 460, max_channel = 565, FWHM = 70
# Initial guesses: (4.467, 510.551, 29.726)
# Optimized Amplitude: 4.383 +- 0.065
# Optimized Mean: 510.134 +- 0.529
# Optimized Stddev: 43.054 +- 0.882
# This yields E = 510.134 +- 194.371

# --- Peak 2: (Maybe also try a fit for the maybe two peaks)
# Array indices: [327:359]
# min_channel = 65.4, max_channel = 71.6, FWHM = 3.8
# Initial guesses: (1.653, 67.8, 1.614)
# Optimized Amplitude: 1.506 +- 0.037
# Optimized Mean: 68.392 +- 0.056
# Optimized Stddev: 2.715 +- 0.095
# This yields E = 68.392 +- 33.348


# Cs-137:
# --- Peak 1: (Not really a peak there !?!)

# --- Peak2:
# Array indices: [161:200]
# min_channel = 32.2, max_channel = 39.8, FWHM = 3.26
# Initial guesses: (7.64, 35.4, 1.384)
# Optimized Amplitude: 7.695 +- 0.097
# Optimized Mean: 36.078 +- 0.021
# Optimized Stddev: 2.011 +- 0.029
# This yields E = 36.078 +- 25.062


# Co-60:
# --- Peak 1:
# Array indices: [298:327]
# min_channel = 59.6, max_channel = 65.2, FWHM = 3.15
# Initial guesses: (0.747, 62.4, 1.338)
# Optimized Amplitude: 0.564 +- 0.027
# Optimized Mean: 62.395 +- 0.107
# Optimized Stddev: 2.66 +- 0.194
# This yields E = 62.395 +- 31.671

# --- Peak 2:
# Array indices: [338:371]
# min_channel = 67.6, max_channel = 74, FWHM = 4.97
# Initial guesses: (0.507, 71.2, 2.111)
# Optimized Amplitude: 0.42 +- 0.018
# Optimized Mean: 70.991 +- 0.116
# Optimized Stddev: 3.133 +- 0.216
# This yields E = 70.991 +- 34.174

# ----- Unknown elements:

# Unknown 1 (v2 is probably better):
# --- Peak 1 (is it there ?):

# --- Peak 2 (@ approx. 75):
# Array indices: [350:413]
# Initial guesses: (0.22, 75.0, 2.251)
# Optimized Amplitude: 0.10475641692909951
# Optimized Mean: 76.1656184330993
# Optimized Stddev: 5.957252853489777

# Unknown 1 v2:
# Peak 1:
# Array indices: [60:77]
# min_channel = 203, max_channel = 264.5, FWHM = 85
# Initial guesses: (2.667, 221.049, 36.096)
# Optimized Amplitude: 2.6 +- 0.073
# Optimized Mean: 236.53 +- 1.404
# Optimized Stddev: 46.041 +- 3.601
# This yields E = 236.53 +- 31.549
