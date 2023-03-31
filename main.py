import pandas as pd
import chardet
import numpy as np
from matplotlib import pyplot as plt


path_Na22 = "/Users/lucas1/Desktop/Uni/Physiklabor A/Physiklabor für Anfänger Teil 2/Versuch 90 - " \
            "Gammaspektroskopie/Materialien/NA22_1.txt"

path_Na22_background = "/Users/lucas1/Desktop/Uni/Physiklabor A/Physiklabor für Anfänger Teil 2/Versuch 90 - " \
                       "Gammaspektroskopie/Materialien/NA22_Untergrund_1.txt"

path_Co60 = "/Users/lucas1/Desktop/Uni/Physiklabor A/Physiklabor für Anfänger Teil 2/Versuch 90 - " \
            "Gammaspektroskopie/Materialien/Co60_1.txt"

path_Cs137 = "/Users/lucas1/Desktop/Uni/Physiklabor A/Physiklabor für Anfänger Teil 2/Versuch 90 - " \
             "Gammaspektroskopie/Materialien/CS137_1.txt"


def getDataframe(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    df = pd.read_csv(file_path, delimiter='\t', decimal=',', skiprows=1, header=None,
                     names=['Counts'], encoding=result['encoding'])

    return df


def get_x_range(df_counts): return np.arange(0, len(df_counts) / 5, 0.2)


def getCorrectedCountsWithErrors(file_path, background_file_path, measurement_time=75,
                                 background_measurement_time=150):
    time_ratio = measurement_time / background_measurement_time

    counts = getDataframe(file_path).values
    background_counts = getDataframe(background_file_path).values

    count_rate = counts / measurement_time
    background_count_rate = background_counts / background_measurement_time

    max_index = min(len(counts), len(background_counts))

    count_rate_error = np.sqrt(counts) / measurement_time
    adjusted_background_count_rate_error = np.sqrt(background_counts * time_ratio) / measurement_time

    corrected_count_rate = count_rate[:max_index] - background_count_rate[:max_index]
    corrected_count_rate_range = get_x_range(corrected_count_rate)

    corrected_count_rate_error = np.sqrt(count_rate_error[:max_index] ** 2
                                         + adjusted_background_count_rate_error[:max_index] ** 2)

    print(len(corrected_count_rate), len(corrected_count_rate_error))

    return np.array(corrected_count_rate_range), np.array(corrected_count_rate), \
        np.array(corrected_count_rate_error)


Na22_counts = getDataframe(path_Na22).values
Na22_range = get_x_range(Na22_counts)
Na22_background_counts = getDataframe(path_Na22_background).values
Na22_background_range = get_x_range(Na22_background_counts)

Na_combined_counts = Na22_counts - Na22_background_counts[:705]
Na_combined_range = get_x_range(Na_combined_counts)

# print(len(Na22_background_counts), len(Na22_counts))

Co60_counts = getDataframe(path_Co60).values
Co60_range = get_x_range(Co60_counts)

Cs137_counts = getDataframe(path_Cs137)
Cs137_range = get_x_range(Cs137_counts)


# counts_array = np.array([Na22_counts, Co60_counts, Cs137_counts])
# range_array = np.array([Na22_range, Co60_range, Cs137_range])
label_array = np.array(["Na-22", "Co-60", "Cs-137"])


def createPlot(file_path, background_file_path, show_errors=True, show_line=True, show_dots=False):
    plt.figure(figsize=(12, 5))
    plt.title('Spektrum der Kanäle')
    plt.xlabel('Kanalnummer (in 0,2er Schritten)')
    plt.ylabel('Zahlrate in [s$^{-1}$]')

    x_range, count_rate, count_rate_error = getCorrectedCountsWithErrors(file_path, background_file_path)

    if show_errors:
        upper = (count_rate + count_rate_error).flatten()
        lower = (count_rate - count_rate_error).flatten()

        plt.fill_between(x_range, upper, lower, where=upper >= lower, interpolate=True, alpha=0.5,
                         label='Konfidenzband')

    if show_line:
        plt.plot(x_range, count_rate, lw=0.6, c='black', label='Verbindungslinie')

    if show_dots:
        plt.scatter(x_range, count_rate, s=3, label='Messwerte')

    plt.xlim(min(x_range), max(x_range))
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.07, right=0.95)
    plt.legend()

    return None


createPlot(path_Na22, path_Na22_background)

plt.show()
