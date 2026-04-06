# constants.py
from typing import Final

# Используем UPPER_CASE для констант
# Добавляем типизацию для подсказок в IDE

# Primaries
LAMBDA_RED:   Final[float] = 620 
LAMBDA_GREEN: Final[float] = 530
LAMBDA_BLUE:  Final[float] = 470

LAMBDA_M_START: Final[float] = 500

Y_STEP_START: Final[float] = 0.05

EPS_INT: Final[float] = 3 # Величина интенсивности ниже которой нет смысла рассматривать, потому что спектр сливается с шумом

# Яркостные коэффициенты для нормировки CMF из интенсивностей
L_R: Final[float] = 1
L_G: Final[float] = 2.263
L_B: Final[float] = 0.239

#Стандартные яркостные коэффициенты для нормировки CMF из интенсивностей 
L_R_1931: Final[float] = 1
L_G_1931: Final[float] = 4.5907
L_B_1931: Final[float] = 0.0601 



