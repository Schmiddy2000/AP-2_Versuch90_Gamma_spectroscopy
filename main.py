import pandas as pd
import chardet
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# import requests
# from io import StringIO

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


def getDataframe(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    df = pd.read_csv(file_path, delimiter='\t', decimal=',', skiprows=1, header=None,
                     names=['Counts'], encoding=result['encoding'])

    return df


def getChannelNumbers(df_counts): return np.arange(0, len(df_counts) / 5, 0.2)


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


# counts_array = np.array([Na22_counts, Co60_counts, Cs137_counts])
# range_array = np.array([Na22_range, Co60_range, Cs137_range])
# label_array = np.array(["Na-22", "Co-60", "Cs-137"])


def createPlot(file_path, background_file_path, do_gaussian=False, show_errors=True, show_line=True,
               show_dots=False, show_bars=False, upper_x_lim=None):

    name_pre = file_path.split('/')
    name = name_pre[len(name_pre) - 1].split('.')[0]

    plt.title('Spektrum der Kanäle von ' + name)
    plt.xlabel('Kanalnummer (in 0,2er Schritten)')
    plt.ylabel('Zahlrate in [s$^{-1}$]')

    x_range, count_rate, count_rate_error = getCorrectedCountsWithErrors(file_path, background_file_path)

    if show_errors:
        upper = count_rate + count_rate_error
        lower = count_rate - count_rate_error

        plt.fill_between(x_range, upper, lower, where=upper >= lower, interpolate=True, alpha=0.5,
                         label='Konfidenzband')

    if show_line:
        plt.semilogy(x_range, count_rate, lw=0.4, c='black', label='Verbindungslinie')

    if show_bars:
        plt.bar(x_range, count_rate, width=0.2, color='black', label='Korrigierte Zählrate')
        plt.ylim(0, None)

    if show_dots:
        plt.scatter(x_range, count_rate, s=1, label='Messwerte')

    if upper_x_lim is None:
        plt.xlim(min(x_range), max(x_range))
    else:
        plt.xlim(min(x_range), upper_x_lim)

    plt.legend()

    if do_gaussian:
        x_lin, popt = make_gaussian(file_path, background_file_path)
        plt.plot(x_lin, gaussian(x_lin, *popt), lw=1, ls='--', c='r')

    return None


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2)


def r(x): return round(x, 3)


def make_gaussian(file_path, background_file_path):
    # Example x and y data (replace these with your actual data points)
    x_data, y_data, y_data_errors = getCorrectedCountsWithErrors(file_path, background_file_path)
    min_index = int(min_channel * 5)
    max_index = int(max_channel * 5) + 1
    x_data = x_data[min_index:max_index]
    y_data = y_data[min_index:max_index]

    mean_guess = x_data[np.argmax(y_data)]
    amplitude_guess = max(y_data)
    stddev_guess = FWHM / (2 * np.sqrt(2 * np.log(2)))

    initial_guesses = (amplitude_guess, mean_guess, stddev_guess)

    # Print run information:
    print('Array indices:', str('[') + str(int(min(x_data) * 5)) + ':' + str(int(max(x_data) * 5 + 1)) + str(']'))
    print('Initial guesses:', '(' + str(r(initial_guesses[0])) + ', ' + str(r(initial_guesses[1])) + ', '
          + str(r(initial_guesses[2])) + ')')

    # Perform the curve fitting
    this_popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guesses)

    # popt contains the optimized parameters: [amplitude, mean, stddev]
    amplitude_opt, mean_opt, stddev_opt = this_popt

    # Print the optimized parameters
    print("Optimized Amplitude:", amplitude_opt)
    print("Optimized Mean:", mean_opt)
    print("Optimized Stddev:", stddev_opt)

    this_x_lin = np.linspace(mean_opt - 10, mean_opt + 10)

    return this_x_lin, this_popt


def FitChannelsAndEnergie():
    channels = np.array([28.178, 66.397, 36.078, 62.395, 70.991])
    energies = np.array([511, 1277, 662, 1172.6, 1332.75])
    channel_lin = np.linspace(0, 140)

    def lin_model(x, a, c): return a * x + c

    popt_lin, pcov_lin = curve_fit(lin_model, channels, energies)

    plt.plot(channel_lin, lin_model(channel_lin, *popt_lin), lw=1, ls='--', c='black')
    print('a =', popt_lin[0], 'and c =', popt_lin[1])

    plt.scatter(channels, energies, marker='x', c='b')


plt.figure(figsize=(12, 5))

min_channel = 70
max_channel = 82.4
FWHM = 5.3


createPlot(path_unknown1_v2, path_unknown1_v2_background, do_gaussian=False, show_line=False, show_errors=False,
           show_bars=True)

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.95)

# plt.savefig('Co-66_gamma_spectrum.png', dpi=300)
plt.show()


# Na-22:
# --- Peak 1:
# Array indices: [128:156]
# Initial guesses: (4.467, 28.2, 1.444)
# Optimized Amplitude: 4.38399800348803
# Optimized Mean: 28.178289087889052
# Optimized Stddev: 2.2036925453995844

# --- Peak 2: (Maybe also try a fit for the maybe two peaks)
# Array indices: [327:358]
# Initial guesses: (1.653, 67.8, 1.609)
# Optimized Amplitude: 1.5032818226468205
# Optimized Mean: 68.39677643280002
# Optimized Stddev: 2.7276249821754535


# Cs-137:
# --- Peak 1: (Not really a peak there !?!)

# --- Peak2:
# Array indices: [161:200]
# Initial guesses: (7.64, 35.4, 1.384)
# Optimized Amplitude: 7.695428607853561
# Optimized Mean: 36.07753219323244
# Optimized Stddev: 2.0108442466204393


# Co-60:
# --- Peak 1:
# Array indices: [298:327]
# Initial guesses: (0.747, 62.4, 1.338)
# Optimized Amplitude: 0.5639292064517797
# Optimized Mean: 62.39534614179052
# Optimized Stddev: 2.659880798250231

# --- Peak 2:
# Array indices: [338:371]
# Initial guesses: (0.507, 71.2, 2.111)
# Optimized Amplitude: 0.42033726173426816
# Optimized Mean: 70.99116909203838
# Optimized Stddev: 3.1332700158790696

# ----- Unknown elements:

# Unknown 1 (v2 is probably better):
# --- Peak 1 (is it there ?):

# --- Peak 2 (@ approx. 75):
# Array indices: [350:413]
# Initial guesses: (0.22, 75.0, 2.251)
# Optimized Amplitude: 0.10475641692909951
# Optimized Mean: 76.1656184330993
# Optimized Stddev: 5.957252853489777


