import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import colour
from matplotlib.widgets import Slider
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
target_dir = current_dir.parent / "ao-system"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(target_dir) not in sys.path:
    sys.path.insert(1, str(target_dir))

from ao.spectral_converter import SpectralConverter

from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy import integrate

import os
from colour.algebra.interpolation import SpragueInterpolator


LAMBDA_RED_STD   = 700
LAMBDA_GREEN_STD = 546.1
LAMBDA_BLUE_STD  = 435.8

NORM_COEF = 4 # нормировочный коэффициент для площади под графиком синего


FILENAME = current_dir / "data.txt"

LAMBDA_INDEX = 0
I_R_INDEX    = 1
I_G_INDEX    = 2
I_B_INDEX    = 3
I_m_INDEX    = 4


from constants import LAMBDA_RED, LAMBDA_GREEN, LAMBDA_BLUE 
from constants import L_R, L_G, L_B


def read_data_from_file(filename):
    with open(filename, "r") as file:
        data_raw = file.readlines()

        data = []

        for i in range(len(data_raw)):
            data.append(list(map(float, data_raw[i].split())))

    return data

def append_primaries_lambda_to_data(data): # добавляет теоритеские значения на длинах волн равных primaries
    data.append([LAMBDA_RED,   1, 0, 0, 1])
    data.append([LAMBDA_GREEN, 0, 1, 0, 1])
    data.append([LAMBDA_BLUE,  0, 0, 1, 1])




    
class Plot():

    def __init__(self, data, show_trackbars = True, build_plot = True):

        # Значения для Primaries CIE 1931
        # self.L_R = 1
        # self.L_G = 4.5907
        # self.L_B = 0.0601 

        # Значения для Primaries, которые использовались в нашем опыте 
        self.L_R = L_R
        self.L_G = L_G
        self.L_B = L_B

        self.LAMBDA_RED   = LAMBDA_RED
        self.LAMBDA_GREEN = LAMBDA_GREEN
        self.LAMBDA_BLUE  = LAMBDA_BLUE

        self.ymin = -0.15
        self.ymax =  0.6


        self.data = data

        append_primaries_lambda_to_data(data)

        self.data.sort(key = lambda x: x[0])
        self.get_real_data()

        self.wavelength = [row[0] for row in self.data]
        self.wavelength_fine = np.arange(min(self.wavelength), max(self.wavelength) + 1, 1)

        print(self.wavelength)

        print("max wv:", max(self.wavelength))
        print("min wv:", min(self.wavelength))

        self.V_lambda_dict = self.get_V_lambda_dict()

        self.cmf_R, self.cmf_G, self.cmf_B = self.get_norm_cmf_by_integration()

        self.save_cmf_to_txt()

        #self.cmf_R, self.cmf_G, self.cmf_B = self.convert_cmf_to_standard_primaries()

        #self.cmf_R, self.cmf_G, self.cmf_B = self.get_cmf_XYZ_by_std_primaries()

        self.cmf_X, self.cmf_Y, self.cmf_Z = self.get_cmf_XYZ()

        #_, self.cmf_Y, _ = self.get_cmf_XYZ_by_ls()

        _, _, _ = self.get_cmf_Y_LS()

        self.show_trackbars = show_trackbars
        if build_plot:
            self.__init_plot()

    def save_cmf_to_txt(self, filename=current_dir / "normalized_cmf.txt"):
        """
        Сохраняет длины волн и отнормированные CMF в текстовый файл.
        Столбцы: Wavelength, R, G, B
        """
        # Собираем данные в одну матрицу (N строк, 4 колонки)
        data = np.column_stack((self.wavelength, self.cmf_R, self.cmf_G, self.cmf_B))
        
        header = "Wavelength(nm) R_CMF G_CMF B_CMF"
        
        # fmt='%.4e' сохранит в научной нотации (удобно для маленьких значений)
        # или '%d %.6f %.6f %.6f' для более привычного вида
        np.savetxt(filename, data, header=header, fmt='%d %.8f %.8f %.8f', delimiter='\t')
        
        print(f"Данные CMF сохранены в текстовый файл: {os.path.abspath(filename)}")


    def load_cmf_from_txt(self, filename=current_dir / "normalized_cmf.txt"):
        """
        Загружает данные CMF из текстового файла и записывает их в атрибуты объекта.
        """
        if not os.path.exists(filename):
            print(f"Ошибка: файл {filename} не найден.")
            return

        # unpack=True позволяет сразу разложить столбцы в отдельные переменные
        data = np.loadtxt(filename, unpack=True)
        
        # Распределяем данные обратно по атрибутам
        self.wavelength = data[0]
        self.cmf_R = data[1]
        self.cmf_G = data[2]
        self.cmf_B = data[3]
        
        print(f"Данные CMF успешно загружены из: {os.path.abspath(filename)}")
        print(f"Загружено точек: {len(self.wavelength)}")
        
       
    @staticmethod
    def get_V_lambda_dict():
        # Get V(λ) at 1 nm intervals from 380-780 nm
        wavelengths = np.arange(380, 781, 1)
        V_lambda = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].values[:, 1]  # y̅ = V(λ)

        V_lambda_dict = dict(zip(wavelengths, V_lambda))

        #print(V_lambda_dict)

        return V_lambda_dict
    

    @staticmethod
    def norm_y(y, x, target_area = 30, k = 1): # k - коэффициент на который делится target_area (учитывает что не вся ось x доступна)
        area = integrate.simpson(y, x)

        return (y / area) * (target_area / k)

    
    def get_norm_cmf_by_integration(self, mode = 'default'):  # вызывает функцию получающую cmf, нормирует cmf по площади и сохраняет отдельные масивы с каналами
        
        cmf_R, cmf_G, cmf_B = self.get_cmf()



        normed_R = self.norm_y(cmf_R, self.wavelength)
        normed_G = self.norm_y(cmf_G, self.wavelength)
        normed_B = self.norm_y(cmf_B, self.wavelength, k = NORM_COEF)

        return normed_R, normed_G, normed_B
        

    @staticmethod
    def invert_value(arr, index): # проверяет неотрицательность выражения и инвертирует его в положительном случае
        if arr[index] < 0:
            print("get_real_data is already used or problems with data")
            return

        arr[index] = -arr[index]        


    def get_real_data(self): # по длине волны находит какая из 3 primaries с отрицательной интенсивностью и ставит знак минус перед интенсивностью
        for i in range(len(self.data)):
            lam = self.data[i][LAMBDA_INDEX]

            if   lam <= self.LAMBDA_BLUE   or lam >= self.LAMBDA_RED: self.invert_value(self.data[i], I_G_INDEX)
            elif lam <= self.LAMBDA_GREEN:                            self.invert_value(self.data[i], I_R_INDEX)
            elif lam <= self.LAMBDA_RED:                              self.invert_value(self.data[i], I_B_INDEX)


    @staticmethod
    def sprague_interp(wl_coarse, values, wl_fine):
        interpolator = SpragueInterpolator(wl_coarse, values)
        return interpolator(wl_fine)

    
    def interpolate_cmf(self, cmf_r, cmf_g, cmf_b, wl_coarse, wl_fine):
        cmf_r_interp = self.sprague_interp(wl_coarse, cmf_r, wl_fine)
        cmf_g_interp = self.sprague_interp(wl_coarse, cmf_g, wl_fine)
        cmf_b_interp = self.sprague_interp(wl_coarse, cmf_b, wl_fine)

        return cmf_r_interp, cmf_g_interp, cmf_b_interp


    def get_cmf(self):

        cmf_R = []
        cmf_G = []
        cmf_B = []

        for i in range(len(self.data)):
            lam = self.data[i][LAMBDA_INDEX]

            int_r = self.data[i][I_R_INDEX]
            int_g = self.data[i][I_G_INDEX]
            int_b = self.data[i][I_B_INDEX]
            int_m = self.data[i][I_m_INDEX]

            norm_R, norm_G, norm_B = self.norm_int(int_r, int_g, int_b, int_m, lam)

            cmf_R.append(norm_R)
            cmf_G.append(norm_G)
            cmf_B.append(norm_B)

    
        return cmf_R, cmf_G, cmf_B
    

    def convert_cmf_to_standard_primaries(self):
        cie_rgb = colour.RGB_COLOURSPACES['CIE RGB']
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        primaries_std = cie_rgb.primaries

        wavelengths = [self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE]

        xy_primaries = []

        print("Координаты спектральных цветов (primaries):")
        for wl in wavelengths:
            # Извлекаем XYZ для конкретной длины волны
            XYZ = cmfs[wl]
            # Переводим XYZ в xy
            xy = colour.XYZ_to_xy(XYZ)

            xy_primaries.append(xy)

        white_D50 = np.array([0.34567, 0.3585])

        # СОЗДАЕМ ОБЪЕКТ (это исправит ошибку)
        my_space = colour.RGB_Colourspace("My_Custom_Space", xy_primaries, white_D50)

        # 2. Целевое пространство (Target) - берем стандартное из словаря
        cie_space = colour.RGB_COLOURSPACES['CIE RGB']

        # 3. Теперь вызываем RGB_to_RGB
        M_transfer = colour.matrix_RGB_to_RGB(my_space, cie_space)


        CMF_MATRIX = np.vstack([self.cmf_R, self.cmf_G, self.cmf_B])
        CMF_MATRIX_STD_PRIMARIES = M_transfer @ CMF_MATRIX

        cmf_r, cmf_g, cmf_b = CMF_MATRIX_STD_PRIMARIES

        cmf_r_normed = self.norm_y(cmf_r, self.wavelength)
        cmf_g_normed = self.norm_y(cmf_g, self.wavelength)
        cmf_b_normed = self.norm_y(cmf_b, self.wavelength, k = NORM_COEF)

        

        return cmf_r_normed, cmf_g_normed, cmf_b_normed
    

    def get_cmf_XYZ(self):
        wavelengths = [self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE]

        # Получаем CMF (стандартный наблюдатель 1931 2 градуса)
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        whitepoint_E = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E']

        xy_primaries = []

        print("Координаты спектральных цветов (primaries):")
        for wl in wavelengths:
            # Извлекаем XYZ для конкретной длины волны
            XYZ = cmfs[wl]
            # Переводим XYZ в xy
            xy = colour.XYZ_to_xy(XYZ)

            xy_primaries.append(xy)

        M_RGB_to_XYZ_E = colour.normalised_primary_matrix(xy_primaries, whitepoint_E)

        

        print("МАТРИЦА ПЕРЕВОДА")
        print(M_RGB_to_XYZ_E)
        print("___________________")

        CMF_MATRIX = np.vstack([self.cmf_R, self.cmf_G, self.cmf_B])

        CMF_MATRIX_XYZ =  M_RGB_to_XYZ_E @ CMF_MATRIX

        cmf_x, cmf_y, cmf_z = CMF_MATRIX_XYZ

        cmf_x_normed = self.norm_y(cmf_x, self.wavelength)
        cmf_y_normed = self.norm_y(cmf_y, self.wavelength)
        cmf_z_normed = self.norm_y(cmf_z, self.wavelength, k = NORM_COEF)

        return cmf_x_normed, cmf_y_normed, cmf_z_normed
            
    
    def get_cmf_XYZ_by_std_primaries(self):
        normed_cmf_R, normed_cmf_G, normed_cmf_B = self.convert_cmf_to_standard_primaries()

        M_rgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])

        CMF_MATRIX_STD_NORMED = np.vstack([normed_cmf_R, normed_cmf_G, normed_cmf_B])

        CMF_MATRIX_XYZ = M_rgb_to_xyz @ CMF_MATRIX_STD_NORMED

        cmf_x, cmf_y, cmf_z = CMF_MATRIX_XYZ

        return cmf_x, cmf_y, cmf_z
    
    def get_cmf_XYZ_by_ls(self):
        
        """
        target_wavelengths: массив длин волн (N,)
        my_cmf_values: ваши CMF (N, 3) -> [x_bar, y_bar, z_bar]
        """
        # 1. Получаем эталонные CMF CIE 1931 и подгоняем под вашу сетку

        cmf_cie_x, cmf_cie_y, cmf_cie_z = self.get_interp_xyz_std_cmf()

        cmf_cie_values = np.stack([cmf_cie_x, cmf_cie_y, cmf_cie_z], axis=1)
        my_cmf_values = np.stack([self.cmf_R, self.cmf_G, self.cmf_B], axis=1)
        # Важно: используем align для корректной интерполяции на вашу сетку
        
        
        # 2. Решаем задачу линейной регрессии: my_cmf * M = cie_values
        # Мы ищем матрицу M (3x3), которая минимизирует норму (my_cmf @ M - cie_values)
        M, residuals, rank, s = np.linalg.lstsq(my_cmf_values, cmf_cie_values, rcond=None)

        print("матрица для метода наименьших квадратов")
        print(M)
        print("))))))))))))))")
        
        # 3. Рассчитываем восстановленные (аппроксимированные) значения
        cie_recovered = my_cmf_values @ M

        return cie_recovered.T


    def norm_int(self, int_r, int_g, int_b, int_m, lam): 

        sum_int = (int_r + int_g + int_b) 

        if sum_int == 0: return 0, 0, 0



        #L = int_r * self.L_R + int_g * self.L_G + int_b * self.L_B

        # cmf_r = int_r * self.V_lambda_dict[lam] / L
        # cmf_g = int_g * self.V_lambda_dict[lam] / L
        # cmf_b = int_b * self.V_lambda_dict[lam] / L


        cmf_r = int_r / (int_m)
        cmf_g = int_g / (int_m)
        cmf_b = int_b / (int_m)


        return cmf_r, cmf_g, cmf_b
    
    def plot_rg_diagram(self):

        r = self.cmf_R / (self.cmf_R + self.cmf_G + self.cmf_B)
        g = self.cmf_G / (self.cmf_R + self.cmf_G + self.cmf_B)

        fig, ax = plt.subplots()

        for i in range(len(r)):
            ax.plot(r[i], g[i])
            ax.annotate(f'{self.wavelength[i]}', (r[i], g[i]), xytext=None)
            

        ax.plot(r, g)


    @staticmethod
    def calculate_sam(spectrum_ref, spectrum_test):
        """
        Рассчитывает спектральный угол (SAM) между двумя массивами.
        
        :param spectrum_ref: Массив значений первого спектра (эталон)
        :param spectrum_test: Массив значений второго спектра (исследуемый)
        :return: Угол в радианах. 0 означает полное совпадение формы.
        """
        # 1. Преобразуем входные данные в массивы numpy (на случай если поданы списки)
        s1 = np.array(spectrum_ref)
        s2 = np.array(spectrum_test)
        
        # 2. Считаем скалярное произведение (числитель)
        dot_product = np.dot(s1, s2)
        
        # 3. Считаем произведение евклидовых норм (знаменатель)
        norm_s1 = np.linalg.norm(s1)
        norm_s2 = np.linalg.norm(s2)
        
        # 4. Находим косинус угла
        # np.clip нужен, чтобы избежать ошибок из-за точности float (значения чуть > 1 или < -1)
        cos_theta = np.clip(dot_product / (norm_s1 * norm_s2), -1.0, 1.0)
        
        # 5. Возвращаем арккосинус (угол в радианах)
        return np.degrees(np.arccos(cos_theta))
    

    @staticmethod
    def normalized_l1_spectral_distance(ref, test):
        ref = np.array(ref)
        test = np.array(test)
        
        # L1 норма разности
        absolute_diff = np.sum(np.abs(ref - test))
        
        # L1 норма референса
        ref_norm = np.sum(np.abs(ref))
        
        # Защита от деления на ноль (если референс пустой)
        if ref_norm == 0:
            return np.nan
            
        return absolute_diff / ref_norm

    def calc_spectral_metrics(self, print_results=True):
        x_bar, y_bar, z_bar = self.get_interp_xyz_std_cmf()

        sam_X = self.calculate_sam(x_bar, self.cmf_X)
        sam_Y = self.calculate_sam(y_bar, self.cmf_Y)
        sam_Z = self.calculate_sam(z_bar, self.cmf_Z)

        nmae_X = self.normalized_l1_spectral_distance(x_bar, self.cmf_X)
        nmae_Y = self.normalized_l1_spectral_distance(y_bar, self.cmf_Y)
        nmae_Z = self.normalized_l1_spectral_distance(z_bar, self.cmf_Z)

        summary = {
            'sam': {
                'X': float(sam_X),
                'Y': float(sam_Y),
                'Z': float(sam_Z),
                'average': float((sam_X + sam_Y + sam_Z) / 3),
            },
            'nmae': {
                'X': float(nmae_X),
                'Y': float(nmae_Y),
                'Z': float(nmae_Z),
                'average': float((nmae_X + nmae_Y + nmae_Z) / 3),
            },
        }

        if print_results:
            print("SAM X:", summary['sam']['X'])
            print("SAM Y:", summary['sam']['Y'])
            print("SAM Z:", summary['sam']['Z'])
            print("average SAM", summary['sam']['average'])
            print("nmae_X", summary['nmae']['X'])
            print("nmae_Y", summary['nmae']['Y'])
            print("nmae_Z", summary['nmae']['Z'])
            print("average nmae", summary['nmae']['average'])

        return summary

    def get_cmf_Y_LS(self):
        cmf_xyz = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        wavelength = cmf_xyz.wavelengths
        interp_values = cmf_xyz[self.wavelength]
        x_bar = interp_values[:, 0]
        y_bar = interp_values[:, 1]
        z_bar = interp_values[:, 2]


        
        x_bar_normed = self.norm_y(x_bar, self.wavelength)
        y_bar_normed = self.norm_y(y_bar, self.wavelength)
        z_bar_normed = self.norm_y(z_bar, self.wavelength, k = NORM_COEF)
        

        CMF_MATRIX = np.vstack([self.cmf_R, self.cmf_G, self.cmf_B]).T

        #print(f"Shape of C: {CMF_MATRIX.shape}") # Ожидаем (N, 3)
        #print(f"Shape of V: {y_bar.shape}")     # Ожидаем (N,)

        m, residuals, rank, s = np.linalg.lstsq(CMF_MATRIX, y_bar_normed, rcond=None)

        print("VECTOR Y LS")
        print(m)

        cmf_y = CMF_MATRIX @ m

        m, residuals, rank, s = np.linalg.lstsq(CMF_MATRIX, x_bar_normed, rcond=None)

        cmf_x = CMF_MATRIX @ m

        m, residuals, rank, s = np.linalg.lstsq(CMF_MATRIX, z_bar_normed, rcond=None)

        cmf_z = CMF_MATRIX @ m



        return cmf_x, cmf_y, cmf_z
    
    def interp_color_checker(self):
        # 2. Безопасное получение данных ColorChecker
        checkers = colour.characterisation.SDS_COLOURCHECKERS
        # Ищем подходящий ключ (сначала 2005, потом Classic, потом любой первый)
        key = next((k for k in ['ColorChecker 2005', 'X-Rite ColorChecker Classic'] if k in checkers), 
                list(checkers.keys())[0])
        
        checker_data = checkers[key].data

        print("checker data dark skin")
        color_checker_dark_skin = checker_data['dark skin']()

        interp_color_checker_dark_skin = self.interp_sd(color_checker_dark_skin)

        return interp_color_checker_dark_skin

        
    def interp_sd(self, sd):
        # 1. Настраиваем метод интерполяции (Cubic Spline — стандарт CIE)
        sd.interpolation = colour.algebra.InterpolationPolynomialCubic()

        # 2. Настраиваем экстраполяцию (чтобы не было ошибок на краях)
        sd.extrapolator = colour.Extrapolator()

        # 3. Получаем значения интерполированного спектра для вашего массива
        interp_sd = sd.values_at_wavelengths(self.wavelengths)

        return interp_sd
       

    def calc_delta_E_cmf_equal_energy(self):
        
        """
        my_cmf_values: np.array формы (N, 3) - ваши x_bar, y_bar, z_bar
        """
        # 1. Получаем стандартные CMF CIE 1931 на вашей сетке
        x_bar, y_bar, z_bar = self.get_interp_xyz_std_cmf()
        
        # 2. Спектр с равной энергией (везде 1.0)
        s_e = np.ones(len(self.wavelength))
        
        # 3. Расчет XYZ методом трапеций (интеграл по self.wavelength)
        # Формула: Integral( S(l) * CMF(l) * dl )
        
        scale_k = 100 / 30

        # Считаем для стандартных CMF
        x_cie = np.trapezoid(s_e * x_bar, self.wavelength)
        y_cie = np.trapezoid(s_e * y_bar, self.wavelength)
        z_cie = np.trapezoid(s_e * z_bar, self.wavelength)
        xyz_cie = np.array([x_cie, y_cie, z_cie]) * scale_k
        
        # Считаем для ваших CMF
        x_my = np.trapezoid(s_e * self.cmf_X, self.wavelength)
        y_my = np.trapezoid(s_e * self.cmf_Y, self.wavelength)
        z_my = np.trapezoid(s_e * self.cmf_Z, self.wavelength)
        xyz_my = np.array([x_my, y_my, z_my]) * scale_k
        
        # 4. Расчет Delta E 2000
        # Точка белого для спектра E (Equi-energy)
        white_e = np.array([1/3, 1/3])
        
        # Нормируем к Y=100 для корректного перевода в Lab
        lab_cie = colour.XYZ_to_Lab(xyz_cie / xyz_cie[1] * 100, illuminant=white_e)
        lab_my = colour.XYZ_to_Lab(xyz_my / xyz_my[1] * 100, illuminant=white_e)
        
        de2000 = colour.delta_E(lab_cie, lab_my, method='CIE 2000')
        
        return xyz_cie, xyz_my, de2000
    

    def calc_delta_E_mono_color_equal_energy(self, print_results=True):

        # 1. Получаем стандартные CMF CIE 1931 на вашей сетке
        x_bar, y_bar, z_bar = self.get_interp_xyz_std_cmf()

        white_e = np.array([1/3, 1/3])

        if print_results:
            print("wl de2000 xyz_cie xyz_my")

        Y_norm = 100 # нормировка на яркость для стандартного наблюдателя

        de2000_arr = []

        for i in range(len(self.wavelength)):
            xyz_cie = np.array([x_bar[i], y_bar[i], z_bar[i]])
            xyz_cie_normed = xyz_cie / y_bar[i] * Y_norm

            eps = 1e-7 
            xyz_my = np.clip(np.array([self.cmf_X[i], self.cmf_Y[i], self.cmf_Z[i]]), eps, None)
            xyz_my_normed = xyz_my / y_bar[i] * Y_norm

            
            lab_cie = colour.XYZ_to_Lab(xyz_cie_normed, illuminant=white_e)
            lab_my = colour.XYZ_to_Lab(xyz_my_normed, illuminant=white_e)
        
            de2000 = colour.delta_E(lab_cie, lab_my, method='CIE 2000')

            de2000_arr.append(de2000)

            if print_results:
                print(self.wavelength[i], de2000, xyz_cie_normed, xyz_my_normed)

        de2000_arr = np.array(de2000_arr, dtype=float)
        max_index = int(np.argmax(de2000_arr))
        summary = {
            'max': float(np.max(de2000_arr)),
            'max_wavelength': float(self.wavelength[max_index]),
            'average': float(np.mean(de2000_arr)),
            'median': float(np.median(de2000_arr)),
            'values': de2000_arr
        }

        if print_results:
            print("____________________")
            print("max DE", summary['max'])
            print("max wavelength", summary['max_wavelength'])
            print("av DE", summary['average'])
            print("med DE", summary['median'])
            print("_____________________")

        return summary


    def calc_delta_E_color_checker(
        self,
        checker_name=None,
        illuminant_name='D65',
        print_results=True
    ):
        checkers = colour.characterisation.SDS_COLOURCHECKERS
        if checker_name is None:
            checker_name = next(
                (k for k in ['ColorChecker 2005', 'X-Rite ColorChecker Classic', 'ColorChecker N Ohta'] if k in checkers),
                list(checkers.keys())[0]
            )

        checker_data = checkers[checker_name].data
        illuminant_sd = colour.SDS_ILLUMINANTS[illuminant_name]
        illuminant_values = np.interp(
            self.wavelength,
            illuminant_sd.wavelengths,
            illuminant_sd.values
        )

        x_bar, y_bar, z_bar = self.get_interp_xyz_std_cmf()
        cie_cmf = np.vstack([x_bar, y_bar, z_bar])
        my_cmf = np.vstack([self.cmf_X, self.cmf_Y, self.cmf_Z])

        eps = 1e-7
        cie_white = np.trapezoid(cie_cmf * illuminant_values, self.wavelength, axis=1)
        my_white = np.trapezoid(my_cmf * illuminant_values, self.wavelength, axis=1)

        cie_k = 100 / max(cie_white[1], eps)
        my_k = 100 / max(my_white[1], eps)
        illuminant_xy = colour.XYZ_to_xy(cie_white)

        results = []
        for patch_name, sd_factory in checker_data.items():
            sd = sd_factory() if callable(sd_factory) else sd_factory
            reflectance = np.interp(
                self.wavelength,
                sd.wavelengths,
                sd.values
            )
            stimulus = reflectance * illuminant_values

            xyz_cie = np.trapezoid(cie_cmf * stimulus, self.wavelength, axis=1) * cie_k
            xyz_my = np.trapezoid(my_cmf * stimulus, self.wavelength, axis=1) * my_k

            xyz_cie = np.clip(xyz_cie, eps, None)
            xyz_my = np.clip(xyz_my, eps, None)

            lab_cie = colour.XYZ_to_Lab(xyz_cie, illuminant=illuminant_xy)
            lab_my = colour.XYZ_to_Lab(xyz_my, illuminant=illuminant_xy)
            de2000 = float(colour.delta_E(lab_cie, lab_my, method='CIE 2000'))

            results.append({
                'patch': patch_name,
                'delta_E_2000': de2000,
                'xyz_cie': xyz_cie,
                'xyz_my': xyz_my
            })

        de_values = np.array([row['delta_E_2000'] for row in results])
        max_index = int(np.argmax(de_values))
        summary = {
            'checker': checker_name,
            'illuminant': illuminant_name,
            'max': float(np.max(de_values)),
            'max_patch': results[max_index]['patch'],
            'average': float(np.mean(de_values)),
            'median': float(np.median(de_values)),
            'results': results
        }

        if print_results:
            print(f"ColorChecker delta E 2000 ({checker_name}, {illuminant_name})")
            print("patch de2000 xyz_cie xyz_my")
            for row in results:
                print(row['patch'], row['delta_E_2000'], row['xyz_cie'], row['xyz_my'])
            print("____________________")
            print("max DE", summary['max'])
            print("max patch", summary['max_patch'])
            print("av DE", summary['average'])
            print("med DE", summary['median'])
            print("____________________")

        return summary



        

    def plot_color_checker(self):

        # 2. Безопасное получение данных ColorChecker
        checkers = colour.characterisation.SDS_COLOURCHECKERS
        # Ищем подходящий ключ (сначала 2005, потом Classic, потом любой первый)
        key = next((k for k in ['ColorChecker 2005', 'X-Rite ColorChecker Classic'] if k in checkers), 
                list(checkers.keys())[0])
        
        checker_data = checkers[key].data

        real_sds = []
        for name, sd in checker_data.items():
            # Если это partial, он превратится в SpectralDistribution при обращении
            # или при создании копии/интерполяции
            real_sd = sd() if callable(sd) else sd
            real_sd.name = name # Явно задаем имя для легенды графика


            # sd_interp = real_sd.copy().align(
            #     self.wavelength,
            #     interpolator=colour.CubicSplineInterpolator,
            #     extrapolator_kwargs={
            #         'method': 'Constant',
            #         'left': min(self.wavelength),
            #         'right': max(self.wavelength)
            #     }
            # )
            # sds_interp.append(sd_interp)

            real_sds.append(real_sd)


        from colour.plotting import plot_multi_sds  # Новое название функции
        plot_multi_sds(real_sds)

    def interpolate_color_checker(self): # интерполирует спектры color checker на массив self.wavelengths

        # 2. Безопасное получение данных ColorChecker
        checkers = colour.characterisation.SDS_COLOURCHECKERS
        # Ищем подходящий ключ (сначала 2005, потом Classic, потом любой первый)
        key = next((k for k in ['ColorChecker 2005', 'X-Rite ColorChecker Classic'] if k in checkers), 
                list(checkers.keys())[0])
        
        checker_data = checkers[key].data

        # 2. Получаем спектр конкретного патча, например "dark skin"
        dark_skin_sd = checker_data['dark skin']

        # 3. Получаем значения на вашей сетке длин волн (self.wavelengths)
        # Библиотека сама проведет интерполяцию
        S = dark_skin_sd.interpolator(self.wavelengths)

        print(f"Длины волн: {self.wavelengths[:3]}...")
        print(f"Значения спектра: {S[:3]}...")

    
    def get_interp_xyz_std_cmf(self):

        cmf_xyz = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

        interp_values = cmf_xyz[self.wavelength]

        
        x_bar = interp_values[:, 0]
        y_bar = interp_values[:, 1]
        z_bar = interp_values[:, 2]


        
        x_bar_normed = self.norm_y(x_bar, self.wavelength)
        y_bar_normed = self.norm_y(y_bar, self.wavelength)
        z_bar_normed = self.norm_y(z_bar, self.wavelength, k = NORM_COEF)

        return x_bar_normed, y_bar_normed, z_bar_normed


    def get_cie1931_rgb_cmf_by_experiment_primaries(self):
        cmf_xyz = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        whitepoint_E = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E']

        xy_primaries = []
        for wl in [self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE]:
            xy_primaries.append(colour.XYZ_to_xy(cmf_xyz[wl]))

        rgb_to_xyz = colour.normalised_primary_matrix(xy_primaries, whitepoint_E)
        xyz_to_rgb = np.linalg.inv(rgb_to_xyz)

        interp_values = cmf_xyz[self.wavelength]
        xyz_cmf_matrix = interp_values.T
        rgb_cmf_matrix = xyz_to_rgb @ xyz_cmf_matrix

        cmf_r, cmf_g, cmf_b = rgb_cmf_matrix

        cmf_r_normed = self.norm_y(cmf_r, self.wavelength)
        cmf_g_normed = self.norm_y(cmf_g, self.wavelength)
        cmf_b_normed = self.norm_y(cmf_b, self.wavelength, k=NORM_COEF)

        return cmf_r_normed, cmf_g_normed, cmf_b_normed



    def plot_std_cmf(self):

        x_bar_normed, y_bar_normed, z_bar_normed = self.get_interp_xyz_std_cmf()

        # 1. Получаем объект пространства CIE RGB
        cie_rgb = colour.RGB_COLOURSPACES['CIE RGB']

        # 2. Матрица перехода в XYZ
        #Точка белого для этой системы исторически — Illuminant E
        # M_cie_xyz_to_rgb = cie_rgb.matrix_XYZ_to_RGB

        # CMF_MATRIX = np.vstack([x_bar_normed, y_bar_normed, z_bar_normed])

        # CMF_RGB_MATRIX = M_cie_xyz_to_rgb @ CMF_MATRIX

        # r_bar, g_bar, b_bar = CMF_RGB_MATRIX

        # x_bar_normed = self.norm_y(r_bar, self.wavelength)
        # y_bar_normed = self.norm_y(g_bar, self.wavelength)
        # z_bar_normed = self.norm_y(b_bar, self.wavelength, k = 4.5)

        sam_R = self.calculate_sam(x_bar_normed, self.cmf_X)
        sam_G = self.calculate_sam(y_bar_normed, self.cmf_Y)
        sam_B = self.calculate_sam(z_bar_normed, self.cmf_Z)

        print("SAM R: ", sam_R)
        print("SAM G: ", sam_G)
        print("SAM B: ", sam_B)

        print("average SAM", (sam_R + sam_G + sam_B) / 3)

        nmae_R = self.normalized_l1_spectral_distance(x_bar_normed, self.cmf_X)
        nmae_G = self.normalized_l1_spectral_distance(y_bar_normed, self.cmf_Y)
        nmae_B = self.normalized_l1_spectral_distance(z_bar_normed, self.cmf_Z)

        print("nmae_R", nmae_R)
        print("nmae_G", nmae_G)
        print("nmae_B", nmae_B)

        print("average nmae", (nmae_R + nmae_G + nmae_B) / 3)

        
        self.line_r_std, = self.ax1.plot(self.wavelength, x_bar_normed, color=(1.0, 0.01, 0.01, 0.5),   label = 'CMF R CIE 1931', linewidth = 2)
        self.line_g_std, = self.ax1.plot(self.wavelength, y_bar_normed, color=(0.01, 1.0, 0.01, 0.5), label = 'CMF G CIE 1931', linewidth = 2)
        self.line_b_std, = self.ax1.plot(self.wavelength, z_bar_normed, color=(0.01, 0.01, 1.0, 0.5),  label = 'CMF B CIE 1931', linewidth = 2)

        # self.ax1.legend(
        #     loc='upper left',
        #     frameon=True,
        #     framealpha=0.95,
        #     edgecolor='black',
        #     fancybox=True,
        #     shadow=True,
        #     fontsize=14
        # )  


    def __init_plot(self):

        self.fig = plt.figure(figsize=(10, 8))

        if self.show_trackbars:
        
            gs = self.fig.add_gridspec(2, 2, height_ratios=[20, 1], 
                                    hspace=0.3, wspace=0.3)
            
            # Оси для графиков
            self.ax1 = self.fig.add_subplot(gs[0, :])  # На всю ширину

            
            # Оси для слайдеров (внизу)
            self.ax_slider1 = plt.axes([0.1, 0.15, 0.35, 0.03])  # [left, bottom, width, height]
            self.ax_slider2 = plt.axes([0.1, 0.10, 0.35, 0.03])


            self.L_G_slider = Slider(self.ax_slider1, 'L_G', 0.01, 10, valinit=self.L_G, valstep=0.1)
            self.L_B_slider = Slider(self.ax_slider2, 'L_B', 0.01, 10, valinit=self.L_B, valstep=0.1)

            self.L_G_slider.on_changed(self.update_slider)
            self.L_B_slider.on_changed(self.update_slider)

        else:
            gs = self.fig.add_gridspec(1, 1, height_ratios=[20], 
                        hspace=0.3, wspace=0.3)
            
            self.ax1 = self.fig.add_subplot(gs[0, :])  # На всю ширину
            
        self.ax1.set_ylim(self.ymin, self.ymax)  # [мин, макс] — подберите под ваши данные 




        

        x_smooth_X, cmf_smooth_X = self.wavelength_fine, self.cmf_X
        x_smooth_Y, cmf_smooth_Y = self.wavelength_fine, self.cmf_Y
        x_smooth_Z, cmf_smooth_Z = self.wavelength_fine, self.cmf_Z

        #wavelengths = self.wavelength_fine

        #cmf_X, cmf_Y, cmf_Z = self.interpolate_cmf(self.cmf_X, self.cmf_Y, self.cmf_Z, self.wavelength, self.wavelength_fine)

        wavelengths = self.wavelength 

        cmf_X, cmf_Y, cmf_Z = self.cmf_X, self.cmf_Y, self.cmf_Z
    
        self.line_R, = self.ax1.plot(wavelengths, cmf_X, color='red',   label = 'CMF X', linewidth = 2)
        self.line_G, = self.ax1.plot(wavelengths, cmf_Y, color='green', label = 'CMF Y', linewidth = 2)
        self.line_B, = self.ax1.plot(wavelengths, cmf_Z, color='blue',  label = 'CMF Z', linewidth = 2)

        self.ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Линия x=0

        # вертикальные линии на длине волн primaries
        # self.ax1.axvline(x=self.LAMBDA_RED,   color='red',   linestyle='--', linewidth=2, label = 'red primarie')
        # self.ax1.axvline(x=self.LAMBDA_GREEN, color='green', linestyle='--', linewidth=2, label = 'green primarie')
        # self.ax1.axvline(x=self.LAMBDA_BLUE,  color='blue',  linestyle='--', linewidth=2, label = 'blue primarie')

        # self.ax1.axvline(x=LAMBDA_RED_STD,   color='red',   linestyle='--', linewidth=2, label = 'red primarie')
        # self.ax1.axvline(x=LAMBDA_GREEN_STD, color='green', linestyle='--', linewidth=2, label = 'green primarie')
        # self.ax1.axvline(x=LAMBDA_BLUE_STD,  color='blue',  linestyle='--', linewidth=2, label = 'blue primarie')


        plt.xlabel("Длина волны (нм)")
        plt.ylabel("значение CMF")
        

        self.ax1.xaxis.label.set_size(12)
        self.ax1.yaxis.label.set_size(12)

        self.ax1.set_title(
        "Сравнение функций соответствия цветов XYZ",
        fontsize=14,              # Размер    
        )

        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

        self.ax1.yaxis.labelpad = 0  # установить отступ

        self.ax1.legend(
            loc='upper right',
            frameon=True,
            framealpha=0.95,
            edgecolor='black',
            fancybox=True,
            shadow=True,
            fontsize=14
        )   

        plt.xticks(np.arange(470, 690, 50))


    def update_slider(self, val):

        if not self.show_trackbars:
            return 

        self.L_G  = self.L_G_slider.val
        self.L_B  = self.L_B_slider.val

        self.cmf_R, self.cmf_G, self.cmf_B = self.get_norm_cmf_by_integration()
        self.cmf_X, self.cmf_Y, self.cmf_Z = self.get_cmf_XYZ()

        self.save_cmf_to_txt()
        
        self.line_R.set_ydata(self.cmf_X)
        self.line_G.set_ydata(self.cmf_Y)
        self.line_B.set_ydata(self.cmf_Z)

        self.fig.canvas.draw_idle()


    def show_plot(self):
        plt.show()



if __name__ == "__main__":

    data = read_data_from_file(FILENAME)

    plot = Plot(data, show_trackbars=False)

    
    plot.plot_rg_diagram()
    plot.plot_std_cmf()

    plot.plot_color_checker()
    #print(plot.interp_color_checker())

    print(plot.calc_delta_E_cmf_equal_energy())

    plot.calc_delta_E_mono_color_equal_energy()

    plot.calc_delta_E_color_checker()
    
    plot.show_plot()
