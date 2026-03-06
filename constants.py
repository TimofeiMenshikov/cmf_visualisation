# constants.py
from typing import Final

# Используем UPPER_CASE для констант
# Добавляем типизацию для подсказок в IDE

# Primaries
LAMBDA_RED:   Final[float] = 620 
LAMBDA_GREEN: Final[float] = 530
LAMBDA_BLUE:  Final[float] = 490

# Яркостные коэффициенты для нормировки CMF из интенсивностей
L_R: Final[float] = 1
L_G: Final[float] = 2.263
L_B: Final[float] = 0.239

EPS_Y = 0.3 # Величина яркости ниже которой нет смысла рассматривать, потому что спектр сливается с шумом