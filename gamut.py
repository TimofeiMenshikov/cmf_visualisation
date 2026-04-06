import colour
import matplotlib.pyplot as plt
import matplotlib

from scripts.render_max_xyY_aoc import find_max_Y

import numpy as np
import os

from ao.spectral_converter import SpectralConverter, xyY_to_XYZ, XYZ_to_xyY
from ao.ao_converter import AoConverter
from ao.ao_device import AoDevice, Channel, ChannelsValidator


from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.patches import Wedge
from matplotlib.axes import Axes

from ao_color_setter import AoColorSetterStatic

from typing import Tuple, Optional, Union, List

from constants import LAMBDA_RED, LAMBDA_GREEN, LAMBDA_BLUE, LAMBDA_M_START, Y_STEP_START
from constants import L_R, L_G, L_B 
from constants import EPS_INT

#from ui_constants import TEXT_INFO_X, 



# Тип для координат: список, кортеж или массив numpy
CoordType = Union[Tuple[float, float], List[float]]


class Point:
    def __init__(
        self, 
        ax: Axes, 
        xy: CoordType, 
        text:   str = 'default text', 
        color:  str = 'purple', 
        legend: str = 'no text',
        annotation_offset: Tuple[int, int] = (5, 5)
    ):
        # 1. Валидация осей
        if not isinstance(ax, Axes):
            raise TypeError(f"ax должен быть matplotlib.axes.Axes, получено {type(ax)}")
        
        # 2. Валидация и нормализация координат
        self.xy = self._validate_coords(xy)
        
        # 3. Валидация текста
        if not isinstance(text, str):
            raise TypeError("text должен быть строкой (str)")
            
        # 4. Валидация цвета
        if not isinstance(color, str):
            raise TypeError("color должен быть строкой (str)")

        # 5. Валидация смещения
        if not isinstance(annotation_offset, (tuple, list)) or len(annotation_offset) != 2:
            raise ValueError("annotation_offset должен быть кортежем из 2 чисел")

        self.ax = ax
        self.text = text
        self.annotation_offset = annotation_offset

        # Отрисовка

        if legend=='no text':
            self.point, = self.ax.plot(
            self.xy[0], self.xy[1], 'ro', markersize=5, color=color
            )
        else:
            self.point, = self.ax.plot(
                self.xy[0], self.xy[1], 'ro', markersize=5, color=color, label=legend
            )
        
        self.annotation = self.ax.annotate(
            text, 
            xy=self.xy, 
            xytext=self.annotation_offset,
            textcoords='offset points',
            fontsize=15
        )

    @staticmethod
    def _validate_coords(xy) -> Tuple[float, float]:
        """
        Безопасная проверка и преобразование координат в кортеж.
        Отдельный метод нужен, чтобы не дублировать код и изолировать логику.
        """
        # Проверка на None (явная, до любых других операций)
        if xy is None:
            raise ValueError("Координаты xy не могут быть None")
        
        # Проверка на итерируемость и длину
        try:
            if len(xy) != 2:
                raise ValueError("xy должен содержать ровно 2 значения (x, y)")
        except TypeError:
            raise TypeError("xy должен быть списком, кортежем или массивом")
        
        # Проверка, что элементы — числа
        try:
            x, y = float(xy[0]), float(xy[1])
        except (ValueError, TypeError):
            raise TypeError("Координаты должны быть числами (int или float)")
            
        return (x, y)


    def update_point(
        self, 
        xy: Optional[CoordType] = None, 
        text: Optional[str] = None
    ) -> None:
        """
        Обновляет координаты и/или текст.
        Аргументы равные None игнорируются.
        """
        # --- Обновление координат ---
        # Используем 'is not None' для безопасности с любыми типами данных
        if xy is not None:
            new_xy = self._validate_coords(xy)
            self.xy = new_xy
            # set_data принимает последовательности
            self.point.set_data([self.xy[0]], [self.xy[1]])
            self.annotation.xy = self.xy

        # --- Обновление текста ---
        if text is not None:
            if not isinstance(text, str):
                raise TypeError("text должен быть строкой (str)")
            self.annotation.set_text(text)
            self.text = text


def find_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    """
    Находит точку пересечения двух прямых, заданных четырьмя точками.
    
    Args:
        p1, p2: две точки первой прямой
        p3, p4: две точки второй прямой
    
    Returns:
        Point: точка пересечения, или None если прямые параллельны,
        или строка с описанием особого случая
    """
    x1, y1 = p1.xy
    x2, y2 = p2.xy
    x3, y3 = p3.xy
    x4, y4 = p4.xy
    
    # Коэффициенты для уравнений прямых: A*x + B*y + C = 0
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = x2 * y1 - x1 * y2
    
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = x4 * y3 - x3 * y4
    
    # Определитель системы
    determinant = A1 * B2 - A2 * B1
    
    # Проверка на параллельность (с учётом погрешности float)
    if abs(determinant) < 1e-10:
        # Проверяем, совпадают ли прямые
        if abs(A1 * C2 - A2 * C1) < 1e-10 and abs(B1 * C2 - B2 * C1) < 1e-10:
            return "Прямые совпадают"
        else:
            return None  # Прямые параллельны
    
    # Находим координаты точки пересечения
    x = (B1 * C2 - B2 * C1) / determinant
    y = (C1 * A2 - C2 * A1) / determinant
    
    return (x, y)


def get_Y_ratio(point1, point2, x_sum, y_sum): # функция для получения соотношения яркостей от координат xy в пространстве цветности
    x1, y1 = point1.xy
    x2, y2 = point2.xy
    
    Y_ratio = ((x2 - x_sum) / (x_sum - x1)) * (y1/y2)

    return Y_ratio


class SpectralConverterMOD(SpectralConverter): # добавлен метод для получения координат xyY из Y и lambda
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Вызов конструктора родителя

        CMF = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

        self.WAVELENGTHS = CMF.wavelengths
        self.V_LAMBDA    = CMF.values[:, 1]
    
        self.K_M = 683.0 / 1000 # лм/Вт


    def get_v_lambda(self, wavelength):
        """
        Получить значение V(λ) для заданной длины волны
        
        Параметры:
            wavelength: Длина волны в нм
        
        Возвращает:
            v_lambda: Значение функции светимости
        """
        return np.interp(wavelength, self.WAVELENGTHS, self.V_LAMBDA)


    def luminance_to_intensity(self, luminance_Y, wavelength):
        """
        Преобразование яркости Y в интенсивность для одной длины волны
        
        Формула: I_e = L_v / (K_m × V(λ))
        
        Параметры:
            luminance_Y: Яркость в кд/м²
            wavelength: Длина волны в нм
        
        Возвращает:
            intensity: Интенсивность в Вт/м²/ср
        """
        v_lambda = self.get_v_lambda(wavelength)
    
        if v_lambda <= 0:
            print(f"⚠️  Предупреждение: V(λ) = 0 при {wavelength} нм")
            print("   Невозможно вычислить интенсивность (деление на ноль)")
            return float('inf')
        
        intensity = luminance_Y /(v_lambda * self.K_M)
        return intensity
    

    def luminances_to_intensities(self, luminances_Y, wavelengths):

        if len(luminances_Y) != len(wavelengths):
            print("len is not equal")
            return None
        
        intensities = np.zeros(len(luminances_Y))

        for i in range(len(luminances_Y)):
            intensities[i] = self.luminance_to_intensity(luminances_Y[i], wavelengths[i])

        return intensities


    def wavelength_and_Y_to_XYZ(self, wavelength, Y):

        intensity = self.luminance_to_intensity(Y, wavelength)

        x, y = self.wavelengths_to_xy(wavelength, intensity)
        xyY = np.array([x, y, Y])
        return xyY_to_XYZ(xyY)


    def two_wavelength_and_Y_to_xyY(self, Y1, Y2, wavelength1, wavelength2):

        XYZ1 = self.wavelength_and_Y_to_XYZ(wavelength1, Y1)
        XYZ2 = self.wavelength_and_Y_to_XYZ(wavelength2, Y2)

        XYZ = XYZ1 + XYZ2

        return XYZ_to_xyY(XYZ)


    def two_wavelength_and_Y_to_xy(self, Y1, Y2, wavelength1, wavelength2):

        return self.two_wavelength_and_Y_to_xyY(Y1, Y2, wavelength1, wavelength2)[:-1] # без яркости
 






class Gamut():

    def __init__(self, visualize_spectra = True, visualize_legend = False, full_screen = False):
        
        
        # открытие ao_converter:
        dir_path = os.path.dirname(__file__)
     
        calibration_path =   os.path.join(dir_path,   "ao", "calibration", "2026-02-04_median_without_IR", "amplitude_intensity_calibration.csv")        
        table_spectra_path = os.path.join(dir_path,   "ao", "calibration", "2026-02-04_median_without_IR", "wv_intens_spectra")

        self.converter = SpectralConverterMOD(observer="1931_2", model="table", table_spectra_path=table_spectra_path)
        self.ao_converter = AoConverter(calibration_path, table_spectra_path=table_spectra_path)

        self.visualize_spectra = visualize_spectra

        self.fig = plt.figure(figsize=(18, 9))

        # Левая половина - гамут (ax)
        self.ax = self.fig.add_axes([0.0, 0.2, 0.5, 0.8])  # [left, bottom, width, height]

        # Правая половина - график (ax2)
        self.ax2 = self.fig.add_axes([0.5, 0.2, 0.5, 0.5])  # начинается с 0.5 по ширине

        self.ax3 = self.fig.add_axes([0.5, 0.7, 0.5, 0.3])
        
        if full_screen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()  # переключить в полноэкранный режим

        if not self.visualize_spectra:
            self.ax2.axis('off')

        plt.rcParams.update({
            'font.size': 18,              # базовый размер шрифта
            'axes.labelsize': 18,         # подписи осей
            'axes.titlesize': 18,         # заголовок
            'xtick.labelsize': 18,        # метки оси X
            'ytick.labelsize': 18,        # метки оси Y
            'legend.fontsize': 18,        # легенда
        })


        # Первый подграфик — CIE 1931
        colour.plotting.plot_chromaticity_diagram_CIE1931(
            axes=self.ax,
            standalone=False,
            show_diagram_colours=True,
            show_spectral_locus=True,
            title='Цветовой гамут CIE 1931',
            bounding_box=(-0.1, 1, -0.1, 1)
        )


        self.ax.xaxis.label.set_size(12)
        self.ax.yaxis.label.set_size(12)

        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12



        # яркости для каждой из primaries и для монохроматической волны
        self.Y_R = 1
        self.Y_G = 1
        self.Y_B = 1
        self.Y_m = 1
        # шаг изменения яркости
        self.Y_STEP = Y_STEP_START

        # длины волн для primaries - не меняются, для монохроматичной волны - меняются
        self.LAMBDA_RED   = LAMBDA_RED
        self.LAMBDA_GREEN = LAMBDA_GREEN
        self.LAMBDA_BLUE  = LAMBDA_BLUE

        self.LAMBDA_M     = LAMBDA_M_START

        self.N_ROUND = 4 # округление значений для того, чтобы Y корректно суммировался с Y_STEP

        #self.update_Y_s()

        self.point_m, self.point_R, self.point_G, self.point_B, self.point_sum_1, self.point_sum_2, self.text_info = self.__init_points_and_text_info()

        self.line = self.__init_primaries_triangle()

        if visualize_legend:

            self.ax.legend(
                loc='upper right',
                frameon=True,
                framealpha=0.95,
                edgecolor='black',
                fancybox=True,
                shadow=True,
                fontsize=14
            )        

        self.slider =  self.__init_slider()

        self.__init_spectral_visualizer()

        

        #self.__init_color_patch() # инициализация полукругов происходит в самой функции потому что сразу вызывается update_color_patch(self)


    def __init_spectral_visualizer(self):

        if not self.visualize_spectra: return
        
        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        sd_red   = self.converter.wavelengths_to_sd(self.LAMBDA_RED, I_R)
        sd_green = self.converter.wavelengths_to_sd(self.LAMBDA_GREEN, I_G)
        sd_blue  = self.converter.wavelengths_to_sd(self.LAMBDA_BLUE, I_B)
        sd_mono  = self.converter.wavelengths_to_sd(self.LAMBDA_M, I_m)

        wavelenghts = sd_red.wavelengths

        #self.fig_spectra, self.ax_spectra = plt.subplots(figsize=(12, 8))

        self.spectra_line_R, = self.ax2.plot(wavelenghts, sd_red.values,    color='red',   label = 'spectra red')
        self.spectra_line_G, = self.ax2.plot(wavelenghts, sd_green.values,  color='green', label = 'spectra green')
        self.spectra_line_B, = self.ax2.plot(wavelenghts, sd_blue.values,   color='blue',  label = 'spectra blue')

        self.spectra_line_M, = self.ax2.plot(wavelenghts, sd_mono.values,    color='purple',label = 'spectra mono')


    def __init_primaries_triangle(self, label='primaries triangle'):
        """
        Рисует треугольник, соединяющий три основные точки (primaries) пунктирной линией.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Оси, на которых рисуется треугольник
        primaries : array-like of shape (3, 2)
            Массив из трех координат (x, y) для основных цветов
            Пример: [[x_red, y_red], [x_green, y_green], [x_blue, y_blue]]
        label : str
            Подпись для легенды
        
        Returns:
        --------
        line : matplotlib.lines.Line2D
            Объект линии для управления свойствами
        """

        xy_red   = self.point_R.xy
        xy_green = self.point_G.xy
        xy_blue  = self.point_B.xy

        # Извлекаем координаты x и y, замыкаем треугольник (первая точка = последняя)
        x_coords = [xy_red[0], xy_green[0], xy_blue[0], xy_red[0]]
        y_coords = [xy_red[1], xy_green[1], xy_blue[1], xy_red[1]]
        
        # Рисуем треугольник пунктирной линией
        line, = self.ax.plot(x_coords, y_coords, 'purple', linestyle='--', 
                        linewidth=1.5, alpha=0.7, label=label)
        

        return line


    def __init_slider(self):

        #if self.visualize_spectra: return 

        slider_ax = plt.axes([0.55, 0.83, 0.3, 0.02], facecolor='lightgray')
        slider = Slider(slider_ax, 'Y_STEP', 0.05, 1, valinit=0.05, valstep=0.05)
        slider.label.set_fontsize(10)
        slider.on_changed(self.update_slider)

        return slider


    def __init_points_and_text_info(self):
        """
        1) получение координат xy для primaries и стартовых координат для монохроматичной волны
        2) инициализация точек 
        """

        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        print(I_R, I_G, I_B, I_m)

        self.XY_RED   = self.converter.wavelengths_to_xy(self.LAMBDA_RED, I_R)  # неизменяемые константы
        self.XY_GREEN = self.converter.wavelengths_to_xy(self.LAMBDA_GREEN, I_G)
        self.XY_BLUE  = self.converter.wavelengths_to_xy(self.LAMBDA_BLUE, I_B)
        xy_m          = self.converter.wavelengths_to_xy(self.LAMBDA_M, I_m)    # нужна только для инициализации

        point_R = Point(self.ax, self.XY_RED,   "red",   legend = 'primaries', annotation_offset=(-30, -5))
        point_G = Point(self.ax, self.XY_GREEN, "green")
        point_B = Point(self.ax, self.XY_BLUE,  "blue")
        point_m = Point(self.ax, xy_m,          "mono",  legend = 'mono', color = 'blue', annotation_offset=(-30, 5))

        

        xy_sum1, xy_sum2, text1, text2, Y_sum1, Y_sum2 = self.get_sum_color_xy_and_Y()

        point_sum_1 = Point(self.ax, xy_sum1, text=text1, legend = 'sum 2 primaries'      , color = 'red', annotation_offset=(10, -5))
        point_sum_2 = Point(self.ax, xy_sum2, text=text2, legend = 'sum primarie and mono', color = 'magenta', annotation_offset=(10, 5))


        text_info = self.__init_text_info(Y_sum1, Y_sum2)

        return point_m, point_R, point_G, point_B, point_sum_1, point_sum_2, text_info
    

    def get_intensities_from_Y(self):
        I_R = round(self.converter.luminance_to_intensity(self.Y_R, self.LAMBDA_RED),   self.N_ROUND)
        I_G = round(self.converter.luminance_to_intensity(self.Y_G, self.LAMBDA_GREEN), self.N_ROUND)
        I_B = round(self.converter.luminance_to_intensity(self.Y_B, self.LAMBDA_BLUE),  self.N_ROUND)      
        I_m = round(self.converter.luminance_to_intensity(self.Y_m, self.LAMBDA_M),     self.N_ROUND)


        wavelengths = [self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE, self.LAMBDA_M]
        intensities = [I_R, I_G, I_B, I_m]

        # должна быть проверка на то что интенсивности доступны

        return I_R, I_G, I_B, I_m


    def __init_text_info(self, Y_sum1, Y_sum2):
        
        self.ax3.axis('off')  # Скрыть оси

        text_info_1 = self.ax3.text(0.1, 0.9, f'Y_R = {self.Y_R}, Y_B = {self.Y_B}, Y_G = {self.Y_G}, Y_m = {self.Y_m}', fontsize = 12)
        text_info_2 = self.ax3.text(0.1, 0.8, f'Y_sum1 = {round(Y_sum1, self.N_ROUND)} Y_sum2 = {round(Y_sum2, self.N_ROUND)}', fontsize = 12)
        text_info_3 = self.ax3.text(0.1, 0.7, f'Y_step = {self.Y_STEP}', fontsize = 12)

        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        text_info_4 = self.ax3.text(0.1, 0.6, f"I_R = {I_R} I_G = {I_G} I_B = {I_B} I_m =  {I_m}", fontsize = 12)  

        return [text_info_1, text_info_2, text_info_3, text_info_4]


    def get_sum_color_xy_and_Y(self):
        """
        point_sum - точка на плоскости xy, которая является суммой 2 цветов
        Определяется какие 2 из 3 цветов в "левом полуполе" а какой цвет в "правом полуполе" вместе с монохроматической волной
        """

        if self.LAMBDA_M <= self.LAMBDA_BLUE or self.LAMBDA_M >= self.LAMBDA_RED:

            xyY_sum1 = self.converter.two_wavelength_and_Y_to_xyY(self.Y_R, self.Y_B, self.LAMBDA_RED, self.LAMBDA_BLUE)
            xyY_sum2 = self.converter.two_wavelength_and_Y_to_xyY(self.Y_G, self.Y_m, self.LAMBDA_GREEN, self.LAMBDA_M)

            text1 = "red + blue"
            text2 = "green + mono"

        elif self.LAMBDA_M <= self.LAMBDA_GREEN:
            
            xyY_sum1 = self.converter.two_wavelength_and_Y_to_xyY(self.Y_G, self.Y_B, self.LAMBDA_GREEN, self.LAMBDA_BLUE)
            xyY_sum2 = self.converter.two_wavelength_and_Y_to_xyY(self.Y_R, self.Y_m, self.LAMBDA_RED,   self.LAMBDA_M)            
            
            text1 = "blue + green"
            text2 = "red + mono"

        elif self.LAMBDA_M <= self.LAMBDA_RED:
            
            xyY_sum1 = self.converter.two_wavelength_and_Y_to_xyY(self.Y_R, self.Y_G, self.LAMBDA_RED,   self.LAMBDA_GREEN)
            xyY_sum2 = self.converter.two_wavelength_and_Y_to_xyY(self.Y_B, self.Y_m, self.LAMBDA_BLUE,  self.LAMBDA_M)            
            
            text1 = "red + green"
            text2 = "blue + mono"

        xy_sum1 = xyY_sum1[:-1]
        xy_sum2 = xyY_sum2[:-1]


        Y_sum1 = xyY_sum1[-1]
        Y_sum2 = xyY_sum2[-1]

        return xy_sum1, xy_sum2, text1, text2, Y_sum1, Y_sum2
    

    def update_Y_sum(self, x_sum, y_sum):

        print("start update_Y_sum")

        #Y_sum = find_max_Y((x_sum, y_sum), self.ao_converter) / 2
        Y_sum = 80 * self.converter.get_v_lambda(self.LAMBDA_M)

        return Y_sum


    def update_Y_s(self): # при изменении длины волны меняет яркости primaries, считая яркость монохроматической волны фиксированной
        
        self.Y_R = 15

        self.Y_B = self.Y_R * L_B 
        
        self.Y_G = self.Y_R * L_G

        self.Y_m = self.Y_R * self.converter.get_v_lambda(self.LAMBDA_M) / self.converter.get_v_lambda(self.LAMBDA_RED)

        self.update_points_and_text_info()

        abs_lambda = 7 # если монохроматический цвет близок к primaries то выставляем суммарную яркость на этот primaries и монохроматический цвет
        
        Y_sum = 20 * self.converter.get_v_lambda(self.LAMBDA_M)

        for i in range(5):
            if abs(self.LAMBDA_M - self.LAMBDA_RED)    < abs_lambda:
                self.Y_m = Y_sum
                self.Y_R = Y_sum
                self.Y_G = 0
                self.Y_B = 0

            elif abs(self.LAMBDA_M - self.LAMBDA_GREEN) < abs_lambda:
                self.Y_m = Y_sum
                self.Y_R = 0
                self.Y_G = Y_sum
                self.Y_B = 0

            elif abs(self.LAMBDA_M - self.LAMBDA_BLUE)  < abs_lambda:
                self.Y_m = Y_sum
                self.Y_R = 0
                self.Y_G = 0
                self.Y_B = Y_sum        

            elif self.LAMBDA_M <= self.LAMBDA_BLUE or self.LAMBDA_M >= self.LAMBDA_RED:
        
                x_sum, y_sum = find_intersection(self.point_R, self.point_B, self.point_G, self.point_m)
                Y_sum = self.update_Y_sum(x_sum, y_sum)

                Y_R_B_ratio = get_Y_ratio(self.point_R, self.point_B, x_sum, y_sum)
                Y_G_m_ratio = get_Y_ratio(self.point_G, self.point_m, x_sum, y_sum)
            
                self.Y_m = Y_sum / (1 + Y_G_m_ratio)
                self.Y_G = self.Y_m * Y_G_m_ratio

                self.Y_B = Y_sum / (1 + Y_R_B_ratio)
                self.Y_R = self.Y_B * Y_R_B_ratio


            elif self.LAMBDA_M <= self.LAMBDA_GREEN:
        
                x_sum, y_sum = find_intersection(self.point_B, self.point_G, self.point_R, self.point_m)
                Y_sum = self.update_Y_sum(x_sum, y_sum)

                Y_B_G_ratio = get_Y_ratio(self.point_B, self.point_G, x_sum, y_sum)
                Y_R_m_ratio = get_Y_ratio(self.point_R, self.point_m, x_sum, y_sum)

                self.Y_m = Y_sum / (1 + Y_R_m_ratio)
                self.Y_R = self.Y_m * Y_R_m_ratio

                self.Y_G = Y_sum / (1 + Y_B_G_ratio)
                self.Y_B = self.Y_G * Y_B_G_ratio

            elif self.LAMBDA_M <= self.LAMBDA_RED:

                x_sum, y_sum = find_intersection(self.point_R, self.point_G, self.point_B, self.point_m)
                Y_sum = self.update_Y_sum(x_sum, y_sum)

                Y_R_G_ratio = get_Y_ratio(self.point_R, self.point_G, x_sum, y_sum)
                Y_B_m_ratio = get_Y_ratio(self.point_B, self.point_m, x_sum, y_sum)

                self.Y_m = Y_sum / (1 + Y_B_m_ratio)
                self.Y_B = self.Y_m * Y_B_m_ratio

                self.Y_G = Y_sum / (1 + Y_R_G_ratio)
                self.Y_R = self.Y_G * Y_R_G_ratio


            self.Y_R = round(self.Y_R, self.N_ROUND)   
            self.Y_G = round(self.Y_G, self.N_ROUND)
            self.Y_B = round(self.Y_B, self.N_ROUND)

            eps_Y_R = EPS_INT * self.converter.get_v_lambda(self.LAMBDA_RED) 
            eps_Y_G = EPS_INT * self.converter.get_v_lambda(self.LAMBDA_GREEN)
            eps_Y_B = EPS_INT * self.converter.get_v_lambda(self.LAMBDA_BLUE)
            eps_Y_m = EPS_INT * self.converter.get_v_lambda(self.LAMBDA_M)


            if self.Y_R < eps_Y_R: self.Y_R = 0 #eps_Y_R
            if self.Y_G < eps_Y_G: self.Y_G = 0 #eps_Y_G
            if self.Y_B < eps_Y_B: self.Y_B = 0 #eps_Y_B
            if self.Y_m < eps_Y_m: self.Y_m = 0 #eps_Y_m

            self.update_points_and_text_info()
        
    def get_frequency_and_power_from_Y_wavelength(self, Y_s, wavelengths):

        frequencies = self.ao_converter._get_frequency(wavelengths)
        intensities = self.converter.luminances_to_intensities(Y_s, wavelengths)
        
        powers = self.ao_converter._get_power(frequencies, intensities)

        return frequencies, powers

    
    def dump_info(self):            # выводит информацию о яркостях и частотах колориметра
        print("Y_R",self.Y_R, "Y_B", self.Y_B, "Y_G", self.Y_G, "Y_m", self.Y_m)
        print("I_R", round(self.converter.luminance_to_intensity(self.Y_R, self.LAMBDA_RED), self.N_ROUND), "I_B", round(self.converter.luminance_to_intensity(self.Y_B, self.LAMBDA_BLUE), self.N_ROUND), "I_G", round(self.converter.luminance_to_intensity(self.Y_G, self.LAMBDA_GREEN), self.N_ROUND), round(self.converter.luminance_to_intensity(self.Y_m, self.LAMBDA_M), self.N_ROUND))
        print("LAMBDA_M", self.LAMBDA_M)


    def update_spectral_visualizer(self):
        
        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        sd_red   = self.converter.wavelengths_to_sd(self.LAMBDA_RED, I_R)
        sd_green = self.converter.wavelengths_to_sd(self.LAMBDA_GREEN, I_G)
        sd_blue  = self.converter.wavelengths_to_sd(self.LAMBDA_BLUE, I_B)
        sd_mono  = self.converter.wavelengths_to_sd(self.LAMBDA_M, I_m)

        wavelenghts = sd_red.wavelengths

        self.spectra_line_R.set_data(wavelenghts, sd_red.values)    
        self.spectra_line_G.set_data(wavelenghts, sd_green.values)
        self.spectra_line_B.set_data(wavelenghts, sd_blue.values)   

        self.spectra_line_M.set_data(wavelenghts, sd_mono.values)



    def update_text_info(self, Y_sum1=None, Y_sum2=None):

        #if self.visualize_spectra: return

        self.text_info[0].set_text(f'Y_R = {self.Y_R}, Y_B = {self.Y_B}, Y_G = {self.Y_G}, Y_m = {self.Y_m}')
        if Y_sum1 != None or Y_sum2 != None:

            Y_sum1 = round(Y_sum1, self.N_ROUND)
            Y_sum2 = round(Y_sum2, self.N_ROUND)

            self.text_info[1].set_text(f'Y_sum1 = {Y_sum1} Y_sum2 = {Y_sum2}')
        
        
        self.text_info[2].set_text(f'Y_step = {self.Y_STEP}')

        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()


        self.text_info[3].set_text(f"I_R = {I_R} I_G = {I_G} I_B = {I_B} I_m =  {I_m}")

         
    def update_points_and_text_info(self):

        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        print("start update points and text info")

        is_error = False

        xy_m          = self.converter.wavelengths_to_xy(self.LAMBDA_M,     I_m)
        self.XY_RED   = self.converter.wavelengths_to_xy(self.LAMBDA_RED,   I_R)
        self.XY_GREEN = self.converter.wavelengths_to_xy(self.LAMBDA_GREEN, I_G)
        self.XY_BLUE  = self.converter.wavelengths_to_xy(self.LAMBDA_BLUE,  I_B)

        try:
            self.point_m.update_point(xy_m)
            self.point_R.update_point(self.XY_RED)
            self.point_G.update_point(self.XY_GREEN)
            self.point_B.update_point(self.XY_BLUE)
        except:
            is_error = True


        xy_sum1, xy_sum2, text1, text2, Y_sum1, Y_sum2 = self.get_sum_color_xy_and_Y()

        self.point_sum_1.update_point(xy_sum1, text=text1)
        self.point_sum_2.update_point(xy_sum2, text=text2)

        self.update_text_info(Y_sum1, Y_sum2)

        return is_error


    def update_primaries_triangle(self):

        xy_red   = self.point_R.xy
        xy_green = self.point_G.xy
        xy_blue  = self.point_B.xy

        # Извлекаем координаты x и y, замыкаем треугольник (первая точка = последняя)
        x_coords = [xy_red[0], xy_green[0], xy_blue[0], xy_red[0]]
        y_coords = [xy_red[1], xy_green[1], xy_blue[1], xy_red[1]]

        self.line.set_data(x_coords, y_coords)


    def update_slider(self, Y_STEP):
        
        self.Y_STEP = round(Y_STEP, self.N_ROUND)
        self.update_text_info()


    def redraw_gamut(self, prev_values = None):
        is_error = self.update_points_and_text_info()

        if is_error:
            self.Y_R, self.Y_G, self.Y_B, self.Y_m, self.LAMBDA_M = prev_values
            self.update_points_and_text_info()

        self.update_primaries_triangle()
        self.update_spectral_visualizer()
        
        self.ax.relim()                     # Пересчитываем границы осей
        self.ax.autoscale_view()            # Применяем новые границы
        self.fig.canvas.draw_idle()         # Запрашиваем перерисовку [[7]]

       
    def update_gamut(self, event):

        n_changed_channel = -1 # Номер измененного канала 0 - красный 1 - зеленый 2 - синий 3 - монохроматичный свет
        update_color_setter_mode = -1


        is_changed = False

        prev_values = [self.Y_R, self.Y_G, self.Y_B, self.Y_m, self.LAMBDA_M] # значения чтобы сделать бекап на случай ошибки
       

        if event.key == 'r':  # Обновляем данные
            
            self.Y_R += self.Y_STEP
            self.Y_R = round(self.Y_R, self.N_ROUND)

            is_changed = True
            n_changed_channel = 0

        elif event.key == 'R':

            self.Y_R -= self.Y_STEP
            self.Y_R = round(self.Y_R, self.N_ROUND)

            if self.Y_R < 0: self.Y_R = 0
            else:       
                is_changed = True
                n_changed_channel = 0

        elif event.key == 'g':

            self.Y_G += self.Y_STEP
            self.Y_G = round(self.Y_G, self.N_ROUND)

            is_changed = True
            n_changed_channel = 1

        elif event.key == 'G':

            self.Y_G -= self.Y_STEP
            self.Y_G = round(self.Y_G, self.N_ROUND)

            if self.Y_G < 0: self.Y_G = 0
            else: 
                is_changed = True
                n_changed_channel = 1

        elif event.key == 'b':

            self.Y_B += self.Y_STEP
            self.Y_B = round(self.Y_B, self.N_ROUND)  

            is_changed = True
            n_changed_channel = 2

        elif event.key == 'B':

            self.Y_B -= self.Y_STEP
            self.Y_B = round(self.Y_B, self.N_ROUND)     

            if self.Y_B < 0: self.Y_B = 0
            else: 
                is_changed = True
                n_changed_channel = 2

        elif event.key == 'i':

            self.Y_m += self.Y_STEP
            self.Y_m = round(self.Y_m, self.N_ROUND)

            is_changed = True
            n_changed_channel = 3

        elif event.key == 'I':

            self.Y_m -= self.Y_STEP
            self.Y_m = round(self.Y_m, self.N_ROUND)
            if self.Y_m < 0: self.Y_m = 0
            else:
                is_changed = True
                n_changed_channel = 3
                

        elif event.key == 'm':
            self.LAMBDA_M += 1

            if self.LAMBDA_M in [self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE]: # чтобы длина монохроматической волны не была равна primaires
                self.LAMBDA_M += 1
            
            if self.LAMBDA_M > 780: self.LAMBDA_M = 780
            else:           
                Point(self.ax, self.point_m.xy, "",  legend = 'mono', color = 'blue')        
                is_changed = True
                n_changed_channel = 3
                self.update_Y_s()


        elif event.key == 'M':        
            self.LAMBDA_M -= 1

            if self.LAMBDA_M in [self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE]: # чтобы длина монохроматической волны не была равна primaires
                self.LAMBDA_M -= 1

            if self.LAMBDA_M < 380: self.LAMBDA_M = 380
            else:                   
                is_changed = True
                n_changed_channel = 3
                self.update_Y_s()


        elif event.key == 'y':
            update_color_setter_mode = 1

        elif event.key == 'Y':
            update_color_setter_mode = 2

        
        elif event.key == 'q':             # Закрываем окно
            plt.close(self.fig)
            
                
        if is_changed:

            self.redraw_gamut(prev_values)


        return is_changed, n_changed_channel, update_color_setter_mode

