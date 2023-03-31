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


def getCorrectedCounts(file_path, background_file_path, measurement_time=75, background_measurement_time=150):
    time_ratio = measurement_time / background_measurement_time

    counts = getDataframe(file_path).values
    background_counts = getDataframe(background_file_path).values * time_ratio

    max_index = min(len(counts), len(background_counts))

    corrected_counts = counts[:max_index] - background_counts[:max_index]
    corrected_count_range = get_x_range(corrected_counts)

    return corrected_count_range, corrected_counts


Na22_counts = getDataframe(path_Na22).values
Na22_range = get_x_range(Na22_counts)
Na22_background_counts = getDataframe(path_Na22_background).values
Na22_background_range = get_x_range(Na22_background_counts)

Na_combined_counts = Na22_counts - Na22_background_counts[:705]
Na_combined_range = get_x_range(Na_combined_counts)

print(len(Na22_background_counts), len(Na22_counts))

Co60_counts = getDataframe(path_Co60).values
Co60_range = get_x_range(Co60_counts)

Cs137_counts = getDataframe(path_Cs137)
Cs137_range = get_x_range(Cs137_counts)


# counts_array = np.array([Na22_counts, Co60_counts, Cs137_counts])
# range_array = np.array([Na22_range, Co60_range, Cs137_range])
label_array = np.array(["Na-22", "Co-60", "Cs-137"])


def createPlot(x_range, counts):



plt.figure(figsize=(12, 5))

# plt.scatter(Na22_range, Na22, s=3)

plt.plot(*getCorrectedCounts(path_Na22, path_Na22_background), lw=0.7)

# for i in range(0, len(counts_array)):
#     print(range_array[i], counts_array[i])
#     plt.scatter(range_array[i], counts_array[i], s=3, label= label_array[i])

# plt.scatter(Na22_range, Na22_counts, s=3, label=label_array[0])
# plt.scatter(Na22_background_range, Na22_background_counts, s=3, label="Na-22 Hintergrund")
# plt.scatter(Na_combined_range, Na_combined_counts, s=3, label="Na-22 Kombiniert")
# plt.scatter(Co60_range, Co60_counts, s=3, label=label_array[1])
# plt.scatter(Cs137_range, Cs137_counts, s=3, label=label_array[2])


# plt.legend()
plt.show()
