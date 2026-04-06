# constants.py
from typing import Final
import os

# Используем UPPER_CASE для констант
# Добавляем типизацию для подсказок в IDE

# Primaries
LAMBDA_RED:   Final[float] = 620 
LAMBDA_GREEN: Final[float] = 530
LAMBDA_BLUE:  Final[float] = 470

LAMBDA_M_START: Final[float] = 530

Y_STEP_START: Final[float] = 0.05

# ##########################################################################################
# #калибровка от 23 марта

# EPS_INT: Final[float] = 1 # Величина интенсивности ниже которой нет смысла рассматривать, потому что спектр сливается с шумом
# Y_SCALE: Final[float] = 80 # насколько нужно увеличить фукнцию видности чтобы получить Y_SUM
# Y_R_START_Y_SUM: Final[float] = 10 # яркость красного для оценки координат цветов при выставлении lambda
 

# # начальные яркости для запуска программы 
# Y_R_START: Final[float] = 5
# Y_G_START: Final[float] = 5
# Y_B_START: Final[float] = 1
# Y_M_START: Final[float] = 5

# dir_path = os.path.dirname(__file__)

# CALIBRATION_PATH =   os.path.join(dir_path, "..", "ao-system",   "ao", "calibration", "2026-03-23", "amplitude_intensity_calibration.csv")        
# TABLE_SPECTRA_PATH = os.path.join(dir_path, "..", "ao-system",    "ao", "calibration", "2026-03-23", "wv_intens_spectra")

# #########################################################################################

##########################################################################################
#калибровка от 3 марта

EPS_INT: Final[float] = 1 # Величина интенсивности ниже которой нет смысла рассматривать, потому что спектр сливается с шумом
Y_SCALE: Final[float] = 80 # насколько нужно увеличить фукнцию видности чтобы получить Y_SUM
Y_R_START_Y_SUM: Final[float] = 10 # яркость красного для оценки координат цветов при выставлении lambda
 

# начальные яркости для запуска программы 
Y_R_START: Final[float] = 5
Y_G_START: Final[float] = 5
Y_B_START: Final[float] = 1
Y_M_START: Final[float] = 5

dir_path = os.path.dirname(__file__)

CALIBRATION_PATH =   os.path.join(dir_path, "..", "ao-system",   "ao", "calibration", "2026-03-03_1", "amplitude_intensity_calibration.csv")        
TABLE_SPECTRA_PATH = os.path.join(dir_path, "..", "ao-system",    "ao", "calibration", "2026-03-03_1", "wv_intens_spectra")

#########################################################################################


# Яркостные коэффициенты для нормировки CMF из интенсивностей
L_R: Final[float] = 1
L_G: Final[float] = 2.263
L_B: Final[float] = 0.239

#Стандартные яркостные коэффициенты для нормировки CMF из интенсивностей 
L_R_1931: Final[float] = 1
L_G_1931: Final[float] = 4.5907
L_B_1931: Final[float] = 0.0601 


