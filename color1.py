import colour
import matplotlib.pyplot as plt

import numpy as np
import os

from ao.spectral_converter import SpectralConverter
from ao.ao_converter import AoConverter


class Point():
    def __init__(self, ax, xy, text='default text', color='purple', annotation_offset=(5, 5)):

        self.ax   = ax
        self.xy   = xy
        self.text = text

        self.point,   = self.ax.plot(xy[0], xy[1], 'ro', markersize=3, color=color)
        self.annotation = self.ax.annotate(text, 
            xy=xy, 
            xytext = annotation_offset,
            textcoords='offset points',   # ⭐ Ключевой параметр!
            fontsize=10)

    def update_point(self, xy=None, text=None):

        #if xy != None:
        self.point.set_data([xy[0]], [xy[1]])
        self.annotation.xy = ([xy[0], xy[1]])

        if text != None:
            self.annotation.set_text(text)   

               
class SpectralConverterMOD(SpectralConverter): # добавлен метод для получения координат xyY из Y и lambda
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Вызов конструктора родителя

        # Длины волн (нм)
        self.WAVELENGTHS = np.array([
            380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450,
            455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525,
            530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600,
            605, 610, 615, 620, 625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
            680, 685, 690, 695, 700, 705, 710, 715, 720, 725, 730, 735, 740, 745, 750,
            755, 760, 765, 770, 775, 780
        ])

        # Функция светимости V(λ) - y_bar из CIE 1931
        self.V_LAMBDA = np.array([
            0.0000, 0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022,
            0.0040, 0.0073, 0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480, 0.0600,
            0.0739, 0.0910, 0.1126, 0.1390, 0.1693, 0.2080, 0.2586, 0.3230, 0.4073,
            0.5030, 0.6082, 0.7100, 0.7932, 0.8620, 0.9149, 0.9540, 0.9803, 0.9949,
            1.0000, 0.9950, 0.9786, 0.9520, 0.9154, 0.8700, 0.8163, 0.7570, 0.6949,
            0.6310, 0.5668, 0.5030, 0.4412, 0.3810, 0.3210, 0.2650, 0.2170, 0.1750,
            0.1382, 0.1070, 0.0816, 0.0610, 0.0446, 0.0320, 0.0232, 0.0170, 0.0119,
            0.0082, 0.0057, 0.0039, 0.0027, 0.0019, 0.0013, 0.0009, 0.0006, 0.0004,
            0.0003, 0.0002, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ])

        self.K_M = 683.0  # лм/Вт

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
        
        intensity = luminance_Y / (self.K_M * v_lambda)
        return intensity
    

    def luminances_to_intensities(self, luminances_Y, wavelengths):

        if len(luminances_Y) != len(wavelengths):
            print("len is not equal")
            return None
        
        intensities = np.zeros(len(luminances_Y))

        for i in range(len(luminances_Y)):
            intensities[i] = self.luminance_to_intensity(luminances_Y[i], wavelengths[i])

        return intensities


    def two_wavelength_and_Y_to_xyY(self, Y1, Y2, wavelength1, wavelength2):
        intensity1 = self.luminance_to_intensity(Y1, wavelength1)
        intensity2 = self.luminance_to_intensity(Y2, wavelength2)

        return self.wavelenths_to_xyY([wavelength1, wavelength2], [intensity1, intensity2])


    def two_wavelength_and_Y_to_xy(self, Y1, Y2, wavelength1, wavelength2):

        return self.two_wavelength_and_Y_to_xyY(Y1, Y2, wavelength1, wavelength2)[:-1] # без яркости
    

class Gamut():
    def __init__(self, *args, **kwargs):
        
        self.converter = SpectralConverterMOD(observer="1931_2", model="band")
        
        # открытие ao_converter:
        dir_path = os.path.dirname(__file__)
        calibration_path = os.path.join(dir_path, "ao", "calibration", "2025-08-07", "amplitude_intensity_calibration_new.csv")        
        self.ao_converter = AoConverter(calibration_path)

        # 5. Строим диаграмму CIE 1931
        self.fig, self.ax = colour.plotting.plot_chromaticity_diagram_CIE1931(
            standalone=False,
            show_diagram_colours=False,
            show_spectral_locus=True,
            title='Цветовой гамут на диаграмме CIE 1931',
            bounding_box=(-0.1, 0.8, -0.1, 0.9)
        )

        # яркости для каждой из primaries и для монохроматической волны
        self.Y_R = 1
        self.Y_G = 1
        self.Y_B = 1
        self.Y_m = 1
        # шаг изменения яркости
        self.Y_step = 0.05

        # длины волн для primaries - не меняются, для монохроматичной волны - меняются
        self.LAMBDA_RED   = 620
        self.LAMBDA_GREEN = 530
        self.LAMBDA_BLUE  = 470

        self.LAMBDA_M     = 500

        self.point_m, self.point_sum_1, self.point_sum_2 = self.__init_points()


        self.fig.canvas.mpl_connect('key_press_event', self.update_gamut)


    def __init_points(self):
        """
        1) получение координат xy для primaries и стартовых координат для монохроматичной волны
        2) инициализация точек 
        """
        self.XY_RED   = self.converter.wavelengths_to_xy(self.LAMBDA_RED)  # неизменяемые константы
        self.XY_GREEN = self.converter.wavelengths_to_xy(self.LAMBDA_GREEN)
        self.XY_BLUE  = self.converter.wavelengths_to_xy(self.LAMBDA_BLUE)
        xy_m          = self.converter.wavelengths_to_xy(self.LAMBDA_M)    # нужна только для инициализации

        point_R = Point(self.ax, self.XY_RED,   "red", annotation_offset=(-10, 5))
        point_G = Point(self.ax, self.XY_GREEN, "green")
        point_B = Point(self.ax, self.XY_BLUE,  "blue")
        point_m = Point(self.ax, xy_m,     "monochroma")

        xy_sum1, xy_sum2, text1, text2 = self.get_sum_color_xy()

        point_sum_1 = Point(self.ax, xy_sum1, text=text1)
        point_sum_2 = Point(self.ax, xy_sum2, text=text2)

        return point_m, point_sum_1, point_sum_2
    

    def get_sum_color_xy(self):
        """
        point_sum - точка на плоскости xy, которая является суммой 2 цветов
        Определяется какие 2 из 3 цветов в "левом полуполе" а какой цвет в "правом полуполе" вместе с монохроматической волной
        """

        if self.LAMBDA_M <= self.LAMBDA_BLUE or self.LAMBDA_M >= self.LAMBDA_RED:

            xy_sum1 = self.converter.two_wavelength_and_Y_to_xy(self.Y_R, self.Y_B, self.LAMBDA_RED, self.LAMBDA_BLUE)
            xy_sum2 = self.converter.two_wavelength_and_Y_to_xy(self.Y_G, self.Y_m, self.LAMBDA_GREEN, self.LAMBDA_M)

            text1 = "red + blue"
            text2 = "green + mono"

        elif self.LAMBDA_M <= self.LAMBDA_GREEN:
            
            xy_sum1 = self.converter.two_wavelength_and_Y_to_xy(self.Y_G, self.Y_B, self.LAMBDA_GREEN, self.LAMBDA_BLUE)
            xy_sum2 = self.converter.two_wavelength_and_Y_to_xy(self.Y_R, self.Y_m, self.LAMBDA_RED,   self.LAMBDA_M)            
            
            text1 = "blue + green"
            text2 = "red + mono"

        elif self.LAMBDA_M <= self.LAMBDA_RED:
            
            xy_sum1 = self.converter.two_wavelength_and_Y_to_xy(self.Y_R, self.Y_G, self.LAMBDA_RED,   self.LAMBDA_GREEN)
            xy_sum2 = self.converter.two_wavelength_and_Y_to_xy(self.Y_B, self.Y_m, self.LAMBDA_BLUE,  self.LAMBDA_M)            
            
            text1 = "red + green"
            text2 = "blue + mono"

        return xy_sum1, xy_sum2, text1, text2
    

    def get_frequency_and_power_from_Y_wavelength(self, Y_s, wavelengths):

        frequencies = self.ao_converter._get_frequency(wavelengths)
        intensities = self.converter.luminances_to_intensities(Y_s, wavelengths)
        
        powers = self.ao_converter._get_power(frequencies, intensities)

        return frequencies, powers

    
    def dump_info(self):            # выводит информацию о яркостях и частотах колориметра
        print("Y_R",self.Y_R, "Y_B", self.Y_B, "Y_G", self.Y_G, "Y_m", self.Y_m)
        print("LAMBDA_M", self.LAMBDA_M)

        Y_s     = np.array([self.Y_R, self.Y_G, self.Y_B, self.Y_m])
        lambdas = np.array([self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE, self.LAMBDA_M]) 

        print(self.get_frequency_and_power_from_Y_wavelength(Y_s, lambdas))

          
    def update_points(self):

        xy_m = self.converter.wavelengths_to_xy(self.LAMBDA_M)
        self.point_m.update_point(xy_m)

        xy_sum1, xy_sum2, text1, text2 = self.get_sum_color_xy()
        self.point_sum_1.update_point(xy_sum1, text=text1)
        self.point_sum_2.update_point(xy_sum2, text=text2)

        
    def update_gamut(self, event):
        is_changed = False
        N_ROUND = 4 # округление значений для того, чтобы Y корректно суммировался с Y_STEP

        if event.key == 'r':  # Обновляем данные
            
            self.Y_R += self.Y_step 
            self.Y_R = round(self.Y_R, N_ROUND)

            is_changed = True

        elif event.key == 'R':

            self.Y_R -= self.Y_step
            self.Y_R = round(self.Y_R, N_ROUND)

            if self.Y_R < 0: self.Y_R = 0
            else:       is_changed = True

        elif event.key == 'g':

            self.Y_G += self.Y_step
            self.Y_G = round(self.Y_G, N_ROUND)


            is_changed = True

        elif event.key == 'G':

            self.Y_G -= self.Y_step
            self.Y_G = round(self.Y_G, N_ROUND)

            if self.Y_G < 0: self.Y_G = 0
            else: is_changed = True

        elif event.key == 'b':

            self.Y_B += self.Y_step
            self.Y_B = round(self.Y_B, N_ROUND)  

            is_changed = True

        elif event.key == 'B':

            self.Y_B -= self.Y_step
            self.Y_B = round(self.Y_B, N_ROUND)     

            if self.Y_B < 0: self.Y_B = 0
            else: is_changed = True

        elif event.key == 'm':
            self.LAMBDA_M += 1
            
            if self.LAMBDA_M > 780: self.LAMBDA_M = 780
            else:                   is_changed = True


        elif event.key == 'M':        
            self.LAMBDA_M -= 1

            if self.LAMBDA_M < 380: self.LAMBDA_M = 380
            else:                   is_changed = True

        elif event.key == 'q':             # Закрываем окно
            plt.close(self.fig)
            
                
        if is_changed:
            self.update_points()

            self.dump_info()

            self.ax.relim()                     # Пересчитываем границы осей
            self.ax.autoscale_view()            # Применяем новые границы
            self.fig.canvas.draw_idle()         # Запрашиваем перерисовку [[7]]


    
def main():

    gamut = Gamut()

    """draw_lines_to_lambda(ax, lam, xy_primaries)

    line,  = ax.plot(                # Запятая нужна, чтобы сохранить ссылку на отрезок, чтобы менять его (магические константы в координатах случайны и не влияют на отображение так как нужны для инициализации)
        [0, 0],  # x-координаты: от зелёного к синему
        [1, 1],  # y-координаты: от зелёного к синему
        color='black',               # Цвет линии (циан для контраста)
        linewidth=1,
        linestyle='--',
        marker='o',                 # Кружки в точках
        markersize=2,        
    )

    point, = ax.plot(0.5, 0.5, 'ro', markersize=3, color='black') # точка на на отрезке без антицвета (магические константы в координатах случайны и не влияют на отображение так как нужны для инициализации)

    handler = partial(on_key, line=line, ax=ax, fig=fig, lam=lam, xy_primaries=xy_primaries, point=point)
    fig.canvas.mpl_connect('key_press_event', handler)



    draw_point(fig, ax, lam, xy_primaries, REAL_COLORS_RATIO, ANTI_COLOR_RATIO, line, point)
    #draw_monochroma_point(ax, lam)"""


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
