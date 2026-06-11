import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg-cache"))

import colour
import matplotlib.pyplot as plt
import matplotlib

#from scripts.render_max_xyY_aoc import find_max_Y

import csv
import pickle
import numpy as np
import sys
from pathlib import Path
from scipy.interpolate import interp1d

current_dir = Path(__file__).resolve().parent
target_dir = current_dir.parent / "ao-system"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(target_dir) not in sys.path:
    sys.path.insert(1, str(target_dir))

from ao.spectral_converter import SpectralConverter, xyY_to_XYZ, XYZ_to_xyY
from ao.ao_converter import AoConverter
from ao.ao_device import AoDevice, Channel, ChannelsValidator


from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.patches import Wedge
from matplotlib.axes import Axes

from ao_color_setter import AoColorSetterStatic

from typing import Tuple, Optional, Union, List

from constants import LAMBDA_RED, LAMBDA_GREEN, LAMBDA_BLUE, LAMBDA_M_START, Y_STEP_START, CALIBRATION_WAVELENGTH_RANGE
from constants import L_R, L_G, L_B 
from constants import EPS_INT
from ui_constants import (
    FIGURE_BG,
    PANEL_BG,
    PANEL_EDGE,
    TEXT_COLOR,
    MUTED_TEXT,
    GRID_COLOR,
    GAMUT_X_LIMITS,
    GAMUT_Y_LIMITS,
)



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

        plot_kwargs = {
            "marker": "o",
            "linestyle": "None",
            "markersize": 8,
            "markerfacecolor": color,
            "markeredgecolor": "white",
            "markeredgewidth": 1.4,
            "color": color,
            "zorder": 5,
        }

        if legend != 'no text':
            plot_kwargs["label"] = legend

        self.point, = self.ax.plot(self.xy[0], self.xy[1], **plot_kwargs)
        
        self.annotation = self.ax.annotate(
            text, 
            xy=self.xy, 
            xytext=self.annotation_offset,
            textcoords='offset points',
            fontsize=10,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": PANEL_EDGE,
                "alpha": 0.86,
            },
            zorder=6,
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


def find_intersection_or_midpoint(p1: Point, p2: Point, p3: Point, p4: Point):
    intersection = find_intersection(p1, p2, p3, p4)

    if isinstance(intersection, tuple) and len(intersection) == 2:
        return intersection

    x1, y1 = p1.xy
    x2, y2 = p2.xy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def get_Y_ratio(point1, point2, x_sum, y_sum): # функция для получения соотношения яркостей от координат xy в пространстве цветности
    x1, y1 = point1.xy
    x2, y2 = point2.xy
    eps = 1e-10

    if abs(x_sum - x1) < eps:
        return np.inf

    if abs(y2) < eps:
        return np.inf
    
    Y_ratio = ((x2 - x_sum) / (x_sum - x1)) * (y1/y2)

    return Y_ratio


def split_Y_between_points(point1, point2, x_sum, y_sum, Y_sum):
    x1, _ = point1.xy
    x2, _ = point2.xy
    eps = 1e-10

    if abs(x_sum - x1) < eps:
        return Y_sum, 0

    if abs(x_sum - x2) < eps:
        return 0, Y_sum

    Y_ratio = get_Y_ratio(point1, point2, x_sum, y_sum)

    if not np.isfinite(Y_ratio):
        return Y_sum, 0

    if Y_ratio <= 0:
        return 0, Y_sum

    Y2 = Y_sum / (1 + Y_ratio)
    Y1 = Y2 * Y_ratio

    return Y1, Y2


class SpectralConverterMOD(SpectralConverter): # добавлен метод для получения координат xyY из Y и lambda
    def __init__(self, *args, **kwargs):
        self.spectra_table = {}
        super().__init__(*args, **kwargs)  # Вызов конструктора родителя

        CMF = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

        self.WAVELENGTHS = CMF.wavelengths
        self.V_LAMBDA    = CMF.values[:, 1]
    
        self.K_M = 683.0 / 1000 # лм/Вт


    def _clip_table_wavelength(self, wavelength):
        if self.model != "table" or not hasattr(self, "wvs") or len(self.wvs) == 0:
            return wavelength

        return float(np.clip(wavelength, min(self.wvs), max(self.wvs)))


    def _table_sd(self, wv, intensity=1.0):
        return super()._table_sd(self._clip_table_wavelength(wv), intensity)


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
 


 



class LocalAoConverter:
    def __init__(self, calibration_path_csv):
        rows = []
        with open(calibration_path_csv, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                intensity = row.get("intensity", row.get("channel1_intensity"))
                rows.append(
                    {
                        "frequency": float(row["frequency"]),
                        "wavelength": float(row["wavelength"]),
                        "amplitude": float(row["amplitude"]),
                        "intensity": float(intensity),
                    }
                )

        import pandas as pd

        self._calibration_data = pd.DataFrame(rows)
        self._frequency_array = np.array(sorted({row["frequency"] for row in rows}), dtype=float)
        self._wavelength_array = np.array(
            [
                np.mean([row["wavelength"] for row in rows if row["frequency"] == frequency])
                for frequency in self._frequency_array
            ],
            dtype=float,
        )
        self._table = {
            frequency: np.array(
                [row["intensity"] for row in rows if row["frequency"] == frequency],
                dtype=float,
            )
            for frequency in self._frequency_array
        }

    def _get_frequency(self, wavelength):
        wavelength_arr = np.asarray(wavelength, dtype=float)
        order = np.argsort(self._wavelength_array)
        result = np.interp(
            wavelength_arr,
            self._wavelength_array[order],
            self._frequency_array[order],
        )
        if np.isscalar(wavelength):
            return float(result)
        return result


def get_table_spectra_path_for_current_ao(table_spectra_path):
    table_spectra_path = Path(table_spectra_path)
    with open(table_spectra_path, "rb") as file:
        table = pickle.load(file)

    if "wavelenths" not in table:
        return table_spectra_path

    cleaned_table = dict(table)
    cleaned_table["wavelengths"] = cleaned_table.pop("wavelenths")
    fixed_path = current_dir / ".runtime" / f"{table_spectra_path.name}_numeric_keys"
    fixed_path.parent.mkdir(exist_ok=True)

    with open(fixed_path, "wb") as file:
        pickle.dump(cleaned_table, file)

    return fixed_path


class Gamut():

    def __init__(self, visualize_spectra = True, visualize_legend = False, full_screen = False):
        
        
        from constants import CALIBRATION_PATH, TABLE_SPECTRA_PATH
     
        # calibration_path =   os.path.join(dir_path, "..", "ao-system",   "ao", "calibration", "2026-03-23", "amplitude_intensity_calibration.csv")        
        # table_spectra_path = os.path.join(dir_path, "..", "ao-system",    "ao", "calibration", "2026-03-23", "wv_intens_spectra")

        table_spectra_path = get_table_spectra_path_for_current_ao(TABLE_SPECTRA_PATH)
        self.converter = SpectralConverterMOD(observer="1931_2", model="table", table_spectra_path=table_spectra_path)
        try:
            self.ao_converter = AoConverter(CALIBRATION_PATH, table_spectra_path=table_spectra_path)
        except Exception as exc:
            print(f"ao converter loading failed, fallback to local csv converter: {exc}")
            self.ao_converter = LocalAoConverter(CALIBRATION_PATH)

        self.visualize_spectra = visualize_spectra

        self.fig = plt.figure(figsize=(18, 9), facecolor=FIGURE_BG)
        self.fig.canvas.manager.set_window_title("AO Color Matching Experiment")

        # Левая половина - гамут (ax)
        self.ax = self.fig.add_axes([0.05, 0.1, 0.5, 0.82])  # [left, bottom, width, height]

        # Правая половина - график (ax2)
        self.ax2 = self.fig.add_axes([0.6, 0.08, 0.36, 0.39])  # начинается с 0.5 по ширине

        self.ax3 = self.fig.add_axes([0.6, 0.7, 0.36, 0.23])
        self.ax2_mode = "spectra"
        
        if full_screen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()  # переключить в полноэкранный режим

        if not self.visualize_spectra:
            self.ax2.axis('off')

        plt.rcParams.update({
            'font.size': 11,              # базовый размер шрифта
            'axes.labelsize': 11,         # подписи осей
            'axes.titlesize': 13,         # заголовок
            'xtick.labelsize': 10,        # метки оси X
            'ytick.labelsize': 10,        # метки оси Y
            'legend.fontsize': 10,        # легенда
        })


        # Первый подграфик — CIE 1931
        colour.plotting.plot_chromaticity_diagram_CIE1931(
            axes=self.ax,
            standalone=False,
            show_diagram_colours=True,
            show_spectral_locus=True,
            title='CIE 1931 Chromaticity',
            bounding_box=(-0.05, 0.85, -0.05, 0.9)
        )

        self.__style_chromaticity_axes()

        from constants import Y_R_START, Y_G_START, Y_B_START, Y_M_START

        # яркости для каждой из primaries и для монохроматической волны
        self.Y_R = Y_R_START
        self.Y_G = Y_G_START
        self.Y_B = Y_B_START
        self.Y_m = Y_M_START
        # шаг изменения яркости
        self.Y_STEP = Y_STEP_START
        self.Y_scale_coeff = 1.2
        # Use 90% of the maximum usable summed luminance
        self.Y_MAX_USAGE_COEFF = 0.9

        # длины волн для primaries - не меняются, для монохроматичной волны - меняются
        self.LAMBDA_RED   = LAMBDA_RED
        self.LAMBDA_GREEN = LAMBDA_GREEN
        self.LAMBDA_BLUE  = LAMBDA_BLUE

        self.LAMBDA_M     = LAMBDA_M_START
        self.MIN_LAMBDA, self.MAX_LAMBDA = CALIBRATION_WAVELENGTH_RANGE

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

        self.slider = self.__init_slider()
        self.scale_slider = self.__init_scale_slider()

        self.__init_spectral_visualizer()

        

        #self.__init_color_patch() # инициализация полукругов происходит в самой функции потому что сразу вызывается update_color_patch(self)

    @staticmethod
    def __style_panel(ax):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PANEL_EDGE)
            spine.set_linewidth(1.0)

    def __style_chromaticity_axes(self):
        self.ax.set_facecolor(PANEL_BG)
        self.ax.set_title("CIE 1931 Chromaticity", fontsize=14, color=TEXT_COLOR, pad=12)
        self.ax.set_xlabel("x", color=MUTED_TEXT)
        self.ax.set_ylabel("y", color=MUTED_TEXT)
        self.ax.set_xlim(*GAMUT_X_LIMITS)
        self.ax.set_ylim(*GAMUT_Y_LIMITS)
        self.ax.tick_params(colors=MUTED_TEXT, labelsize=10)
        self.ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.75)
        for spine in self.ax.spines.values():
            spine.set_color(PANEL_EDGE)
            spine.set_linewidth(1.0)

    def __style_spectra_axes(self):
        self.__style_panel(self.ax2)
        self.ax2.set_title("Spectra", fontsize=13, color=TEXT_COLOR, pad=10)
        self.ax2.set_xlabel("Wavelength, nm", color=MUTED_TEXT)
        self.ax2.set_ylabel("Intensity", color=MUTED_TEXT)
        self.ax2.tick_params(colors=MUTED_TEXT, labelsize=10)
        self.ax2.grid(True, color=GRID_COLOR, linewidth=0.8)
        self.ax2.legend(loc="upper right", frameon=False)

    def __style_cmf_axes(self):
        self.__style_panel(self.ax2)
        self.ax2.set_title("RGB CMF from data.txt", fontsize=13, color=TEXT_COLOR, pad=10)
        self.ax2.set_xlabel("Wavelength, nm", color=MUTED_TEXT)
        self.ax2.set_ylabel("CMF value", color=MUTED_TEXT)
        self.ax2.tick_params(colors=MUTED_TEXT, labelsize=10)
        self.ax2.grid(True, color=GRID_COLOR, linewidth=0.8)
        self.ax2.axhline(y=0, color="#111827", linewidth=1.0)
        self.ax2.legend(loc="upper right", frameon=False)

    def __style_info_axes(self):
        self.ax3.clear()
        self.__style_panel(self.ax3)
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.ax3.set_xlim(0, 1)
        self.ax3.set_ylim(0, 1)
        self.ax3.text(
            0.04,
            0.9,
            "Experiment state",
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
            transform=self.ax3.transAxes,
        )


    def show_metrics_info(self, lines):
        self.info_panel_mode = "metrics"
        self.ax3.clear()
        self.__style_panel(self.ax3)
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.ax3.set_xlim(0, 1)
        self.ax3.set_ylim(0, 1)
        self.ax3.text(
            0.04,
            0.9,
            "Metrics",
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
            transform=self.ax3.transAxes,
        )

        text_kwargs = {
            "fontsize": 8,
            "color": TEXT_COLOR,
            "transform": self.ax3.transAxes,
            "family": "monospace",
            "va": "top",
        }
        y = 0.76
        for line in lines:
            self.ax3.text(0.04, y, line, **text_kwargs)
            y -= 0.061

    def show_intensity_info(self):
        _, _, _, _, Y_sum1, Y_sum2 = self.get_sum_color_xy_and_Y()
        self.text_info = self.__init_text_info(Y_sum1, Y_sum2)


    def __init_spectral_visualizer(self):

        if not self.visualize_spectra: return
        self.ax2_mode = "spectra"
        self.ax2.clear()
        
        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        sd_red   = self.converter.wavelengths_to_sd(self.LAMBDA_RED, I_R)
        sd_green = self.converter.wavelengths_to_sd(self.LAMBDA_GREEN, I_G)
        sd_blue  = self.converter.wavelengths_to_sd(self.LAMBDA_BLUE, I_B)
        sd_mono  = self.converter.wavelengths_to_sd(self.LAMBDA_M, I_m)

        wavelenghts = sd_red.wavelengths

        #self.fig_spectra, self.ax_spectra = plt.subplots(figsize=(12, 8))

        self.spectra_line_R, = self.ax2.plot(wavelenghts, sd_red.values,    color='#ef4444',   label = 'red', linewidth=2)
        self.spectra_line_G, = self.ax2.plot(wavelenghts, sd_green.values,  color='#16a34a', label = 'green', linewidth=2)
        self.spectra_line_B, = self.ax2.plot(wavelenghts, sd_blue.values,   color='#2563eb',  label = 'blue', linewidth=2)

        self.spectra_line_M, = self.ax2.plot(wavelenghts, sd_mono.values,    color='#9333ea',label = 'mono', linewidth=2)
        self.__style_spectra_axes()

    def show_cmf_rgb(self):
        from plot_cmf import FILENAME, Plot, read_data_from_file

        data = read_data_from_file(FILENAME)
        cmf_plot = Plot(data, show_trackbars=False, build_plot=False)

        self.ax2_mode = "cmf"
        self.ax2.clear()
        cie_r, cie_g, cie_b = cmf_plot.get_cie1931_rgb_cmf_by_experiment_primaries()
        self.ax2.plot(cmf_plot.wavelength, cie_r, color="#ef4444", label="CIE R", linewidth=2, alpha=0.28)
        self.ax2.plot(cmf_plot.wavelength, cie_g, color="#16a34a", label="CIE G", linewidth=2, alpha=0.28)
        self.ax2.plot(cmf_plot.wavelength, cie_b, color="#2563eb", label="CIE B", linewidth=2, alpha=0.28)
        self.ax2.plot(cmf_plot.wavelength, cmf_plot.cmf_R, color="#ef4444", label="R CMF", linewidth=2)
        self.ax2.plot(cmf_plot.wavelength, cmf_plot.cmf_G, color="#16a34a", label="G CMF", linewidth=2)
        self.ax2.plot(cmf_plot.wavelength, cmf_plot.cmf_B, color="#2563eb", label="B CMF", linewidth=2)
        self.__style_cmf_axes()
        self.fig.canvas.draw_idle()

    def show_spectra(self):
        self.__init_spectral_visualizer()
        self.fig.canvas.draw_idle()


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
        line, = self.ax.plot(x_coords, y_coords, color='#7c3aed', linestyle='--',
                        linewidth=2.0, alpha=0.85, label=label, zorder=4)
        

        return line


    def __init_slider(self):

        #if self.visualize_spectra: return 

        slider_ax = plt.axes([0.66, 0.535, 0.25, 0.025], facecolor=PANEL_BG)
        slider = Slider(
            slider_ax,
            'Y_STEP',
            0.05,
            1,
            valinit=0.05,
            valstep=0.05,
            color="#2563eb",
            track_color="#dbeafe",
            handle_style={"facecolor": "#1d4ed8", "edgecolor": "white", "size": 9},
        )
        slider.label.set_fontsize(10)
        slider.label.set_color(MUTED_TEXT)
        slider.valtext.set_fontsize(10)
        slider.valtext.set_color(TEXT_COLOR)
        for spine in slider_ax.spines.values():
            spine.set_color(PANEL_EDGE)
        slider.on_changed(self.update_slider)

        return slider


    def __init_scale_slider(self):

        slider_ax = plt.axes([0.66, 0.495, 0.25, 0.025], facecolor=PANEL_BG)
        slider = Slider(
            slider_ax,
            'Y_scale_coeff',
            1.01,
            2,
            valinit=self.Y_scale_coeff,
            valstep=0.01,
            color="#0f766e",
            track_color="#ccfbf1",
            handle_style={"facecolor": "#0f766e", "edgecolor": "white", "size": 9},
        )
        slider.label.set_fontsize(10)
        slider.label.set_color(MUTED_TEXT)
        slider.valtext.set_fontsize(10)
        slider.valtext.set_color(TEXT_COLOR)
        for spine in slider_ax.spines.values():
            spine.set_color(PANEL_EDGE)
        slider.on_changed(self.update_scale_slider)

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

        point_R = Point(self.ax, self.XY_RED,   "red",   color='#ef4444', legend = 'primaries', annotation_offset=(-30, -5))
        point_G = Point(self.ax, self.XY_GREEN, "green", color='#16a34a')
        point_B = Point(self.ax, self.XY_BLUE,  "blue",  color='#2563eb')
        point_m = Point(self.ax, xy_m,          "mono",  legend = 'mono', color = '#9333ea', annotation_offset=(-30, 5))

        

        xy_sum1, xy_sum2, text1, text2, Y_sum1, Y_sum2 = self.get_sum_color_xy_and_Y()

        point_sum_1 = Point(self.ax, xy_sum1, text=text1, legend = 'sum 2 primaries'      , color = '#f97316', annotation_offset=(10, -5))
        point_sum_2 = Point(self.ax, xy_sum2, text=text2, legend = 'sum primarie and mono', color = '#db2777', annotation_offset=(10, 5))


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
        
        self.info_panel_mode = "experiment"
        self.__style_info_axes()

        text_kwargs = {
            "fontsize": 11,
            "color": TEXT_COLOR,
            "transform": self.ax3.transAxes,
            "family": "monospace",
        }

        text_info_1 = self.ax3.text(0.04, 0.72, f'lambda_m = {self.LAMBDA_M} nm', **text_kwargs)
        text_info_2 = self.ax3.text(0.04, 0.56, f'Y_R = {self.Y_R}, Y_B = {self.Y_B}, Y_G = {self.Y_G}, Y_m = {self.Y_m}', **text_kwargs)
        text_info_3 = self.ax3.text(0.04, 0.4, f'Y_sum1 = {round(Y_sum1, self.N_ROUND)} Y_sum2 = {round(Y_sum2, self.N_ROUND)}', **text_kwargs)
        text_info_4 = self.ax3.text(0.04, 0.24, f'Y_step = {self.Y_STEP}, Y_scale = {self.Y_scale_coeff}', **text_kwargs)

        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        text_info_5 = self.ax3.text(0.04, 0.08, f"I_R = {I_R} I_G = {I_G} I_B = {I_B} I_m =  {I_m}", **text_kwargs)

        return [text_info_1, text_info_2, text_info_3, text_info_4, text_info_5]


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


    def get_max_intensity_for_wavelength(self, wavelength):
        spectra_table = getattr(self.converter, "spectra_table", None)
        if not spectra_table:
            return np.inf

        table_wavelengths = np.array(
            sorted(float(wv) for wv in spectra_table.keys() if not isinstance(wv, str)),
            dtype=float,
        )
        if len(table_wavelengths) == 0:
            return np.inf

        idx = np.searchsorted(table_wavelengths, wavelength)
        if idx <= 0:
            candidate_wavelengths = [table_wavelengths[0]]
        elif idx >= len(table_wavelengths):
            candidate_wavelengths = [table_wavelengths[-1]]
        elif np.isclose(table_wavelengths[idx], wavelength):
            candidate_wavelengths = [table_wavelengths[idx]]
        else:
            candidate_wavelengths = [table_wavelengths[idx - 1], table_wavelengths[idx]]

        max_intensities = []
        for candidate_wavelength in candidate_wavelengths:
            max_intensities.append(np.max(spectra_table[np.float64(candidate_wavelength)][0]))

        return float(min(max_intensities))


    def get_max_Y_for_wavelength(self, wavelength):
        max_intensity = self.get_max_intensity_for_wavelength(wavelength)
        return max_intensity * self.converter.get_v_lambda(wavelength) * self.converter.K_M


    def get_Y_sum_close_to_max(self, channel_fractions):
        max_Y_sum = np.inf
        eps = 1e-10

        for fraction, wavelength in channel_fractions:
            if fraction <= eps:
                continue

            max_channel_Y = self.get_max_Y_for_wavelength(wavelength)
            max_Y_sum = min(max_Y_sum, max_channel_Y / fraction)

        if not np.isfinite(max_Y_sum):
            return 0

        return max_Y_sum * self.Y_MAX_USAGE_COEFF


    def update_Y_s(self): # при изменении длины волны меняет яркости primaries, считая яркость монохроматической волны фиксированной
        
        self.Y_R = 5

        self.Y_B = self.Y_R * L_B 
        
        self.Y_G = self.Y_R * L_G

        self.Y_m = self.Y_R * self.converter.get_v_lambda(self.LAMBDA_M) / self.converter.get_v_lambda(self.LAMBDA_RED)

        self.update_points_and_text_info()

        abs_lambda = 7 # если монохроматический цвет близок к primaries то выставляем суммарную яркость на этот primaries и монохроматический цвет
        
        for i in range(5):
            if abs(self.LAMBDA_M - self.LAMBDA_RED)    < abs_lambda:
                Y_sum = self.get_Y_sum_close_to_max([
                    (1, self.LAMBDA_M),
                    (1, self.LAMBDA_RED),
                ])
                self.Y_m = Y_sum
                self.Y_R = Y_sum
                self.Y_G = 0
                self.Y_B = 0

            elif abs(self.LAMBDA_M - self.LAMBDA_GREEN) < abs_lambda:
                Y_sum = self.get_Y_sum_close_to_max([
                    (1, self.LAMBDA_M),
                    (1, self.LAMBDA_GREEN),
                ])
                self.Y_m = Y_sum
                self.Y_R = 0
                self.Y_G = Y_sum
                self.Y_B = 0

            elif abs(self.LAMBDA_M - self.LAMBDA_BLUE)  < abs_lambda:
                Y_sum = self.get_Y_sum_close_to_max([
                    (1, self.LAMBDA_M),
                    (1, self.LAMBDA_BLUE),
                ])
                self.Y_m = Y_sum
                self.Y_R = 0
                self.Y_G = 0
                self.Y_B = Y_sum        

            elif self.LAMBDA_M <= self.LAMBDA_BLUE or self.LAMBDA_M >= self.LAMBDA_RED:
        
                x_sum, y_sum = find_intersection_or_midpoint(self.point_R, self.point_B, self.point_G, self.point_m)

                Y_R_fraction, Y_B_fraction = split_Y_between_points(self.point_R, self.point_B, x_sum, y_sum, 1)
                Y_G_fraction, Y_m_fraction = split_Y_between_points(self.point_G, self.point_m, x_sum, y_sum, 1)
                Y_sum = self.get_Y_sum_close_to_max([
                    (Y_R_fraction, self.LAMBDA_RED),
                    (Y_B_fraction, self.LAMBDA_BLUE),
                    (Y_G_fraction, self.LAMBDA_GREEN),
                    (Y_m_fraction, self.LAMBDA_M),
                ])

                self.Y_R = Y_R_fraction * Y_sum
                self.Y_B = Y_B_fraction * Y_sum
                self.Y_G = Y_G_fraction * Y_sum
                self.Y_m = Y_m_fraction * Y_sum


            elif self.LAMBDA_M <= self.LAMBDA_GREEN:
        
                x_sum, y_sum = find_intersection_or_midpoint(self.point_B, self.point_G, self.point_R, self.point_m)

                Y_B_fraction, Y_G_fraction = split_Y_between_points(self.point_B, self.point_G, x_sum, y_sum, 1)
                Y_R_fraction, Y_m_fraction = split_Y_between_points(self.point_R, self.point_m, x_sum, y_sum, 1)
                Y_sum = self.get_Y_sum_close_to_max([
                    (Y_B_fraction, self.LAMBDA_BLUE),
                    (Y_G_fraction, self.LAMBDA_GREEN),
                    (Y_R_fraction, self.LAMBDA_RED),
                    (Y_m_fraction, self.LAMBDA_M),
                ])

                self.Y_B = Y_B_fraction * Y_sum
                self.Y_G = Y_G_fraction * Y_sum
                self.Y_R = Y_R_fraction * Y_sum
                self.Y_m = Y_m_fraction * Y_sum

            elif self.LAMBDA_M <= self.LAMBDA_RED:

                x_sum, y_sum = find_intersection_or_midpoint(self.point_R, self.point_G, self.point_B, self.point_m)

                Y_R_fraction, Y_G_fraction = split_Y_between_points(self.point_R, self.point_G, x_sum, y_sum, 1)
                Y_B_fraction, Y_m_fraction = split_Y_between_points(self.point_B, self.point_m, x_sum, y_sum, 1)
                Y_sum = self.get_Y_sum_close_to_max([
                    (Y_R_fraction, self.LAMBDA_RED),
                    (Y_G_fraction, self.LAMBDA_GREEN),
                    (Y_B_fraction, self.LAMBDA_BLUE),
                    (Y_m_fraction, self.LAMBDA_M),
                ])

                self.Y_R = Y_R_fraction * Y_sum
                self.Y_G = Y_G_fraction * Y_sum
                self.Y_B = Y_B_fraction * Y_sum
                self.Y_m = Y_m_fraction * Y_sum


            self.Y_R = round(self.Y_R, self.N_ROUND)   
            self.Y_G = round(self.Y_G, self.N_ROUND)
            self.Y_B = round(self.Y_B, self.N_ROUND)
            self.Y_m = round(self.Y_m, self.N_ROUND)

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
        
        powers = self._get_power_from_calibration(frequencies, intensities)

        return frequencies, powers

    def _get_power_from_calibration(self, frequencies, intensities):
        powers = []
        freq_array = np.array(self.ao_converter._frequency_array)

        for frequency, intensity in zip(frequencies, intensities):
            idx = np.searchsorted(freq_array, frequency)
            idx = min(max(idx, 0), len(freq_array) - 2)

            fr_down = freq_array[idx]
            fr_up = freq_array[idx + 1]

            table_down = self.ao_converter._table[fr_down]
            table_up = self.ao_converter._table[fr_up]
            intens_array = table_down + (frequency - fr_down) / (fr_up - fr_down) * (table_up - table_down)

            rows = self.ao_converter._calibration_data[self.ao_converter._calibration_data["frequency"] == fr_down]
            amplitude_array = rows["amplitude"].to_numpy(dtype=float)

            if len(intens_array) != len(amplitude_array):
                raise ValueError(
                    f"Calibration table mismatch for frequency {fr_down}: "
                    f"{len(intens_array)} intensities and {len(amplitude_array)} amplitudes."
                )

            i_max = np.argmax(intens_array)
            intens_array = intens_array.copy()
            intens_array[i_max:] = intens_array[i_max]

            power_interp = interp1d(
                intens_array,
                amplitude_array,
                fill_value="extrapolate",
            )
            powers.append(float(np.clip(power_interp(intensity), amplitude_array[0], amplitude_array[i_max])))

        return powers

    
    def dump_info(self):            # выводит информацию о яркостях и частотах колориметра
        print("Y_R",self.Y_R, "Y_B", self.Y_B, "Y_G", self.Y_G, "Y_m", self.Y_m)
        print("I_R", round(self.converter.luminance_to_intensity(self.Y_R, self.LAMBDA_RED), self.N_ROUND), "I_B", round(self.converter.luminance_to_intensity(self.Y_B, self.LAMBDA_BLUE), self.N_ROUND), "I_G", round(self.converter.luminance_to_intensity(self.Y_G, self.LAMBDA_GREEN), self.N_ROUND), round(self.converter.luminance_to_intensity(self.Y_m, self.LAMBDA_M), self.N_ROUND))
        print("LAMBDA_M", self.LAMBDA_M)


    def update_spectral_visualizer(self):
        if self.ax2_mode != "spectra" or not self.visualize_spectra:
            return

        if not hasattr(self, "spectra_line_R"):
            return
        
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
        self.ax2.relim()
        self.ax2.autoscale_view()



    def update_text_info(self, Y_sum1=None, Y_sum2=None):

        #if self.visualize_spectra: return
        if getattr(self, "info_panel_mode", "experiment") != "experiment":
            if Y_sum1 is None or Y_sum2 is None:
                _, _, _, _, Y_sum1, Y_sum2 = self.get_sum_color_xy_and_Y()
            self.text_info = self.__init_text_info(Y_sum1, Y_sum2)

        self.text_info[0].set_text(f'lambda_m = {self.LAMBDA_M} nm')
        self.text_info[1].set_text(f'Y_R = {self.Y_R}, Y_B = {self.Y_B}, Y_G = {self.Y_G}, Y_m = {self.Y_m}')
        if Y_sum1 != None or Y_sum2 != None:

            Y_sum1 = round(Y_sum1, self.N_ROUND)
            Y_sum2 = round(Y_sum2, self.N_ROUND)

            self.text_info[2].set_text(f'Y_sum1 = {Y_sum1} Y_sum2 = {Y_sum2}')
        
        
        self.text_info[3].set_text(f'Y_step = {self.Y_STEP}, Y_scale = {self.Y_scale_coeff}')

        I_R, I_G, I_B, I_m = self.get_intensities_from_Y()

        self.text_info[4].set_text(f"I_R = {I_R} I_G = {I_G} I_B = {I_B} I_m =  {I_m}")

         
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
        except Exception as exc:
            print(f"failed to update gamut points: {exc}")
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


    def update_scale_slider(self, Y_scale_coeff):
        
        self.Y_scale_coeff = round(Y_scale_coeff, self.N_ROUND)
        self.update_text_info()


    def scale_luminances(self, scale_coeff):
        prev_values = [self.Y_R, self.Y_G, self.Y_B, self.Y_m, self.LAMBDA_M]

        self.Y_R = round(max(0, self.Y_R * scale_coeff), self.N_ROUND)
        self.Y_G = round(max(0, self.Y_G * scale_coeff), self.N_ROUND)
        self.Y_B = round(max(0, self.Y_B * scale_coeff), self.N_ROUND)
        self.Y_m = round(max(0, self.Y_m * scale_coeff), self.N_ROUND)

        try:
            self.redraw_gamut(prev_values)
        except Exception as exc:
            self.Y_R, self.Y_G, self.Y_B, self.Y_m, self.LAMBDA_M = prev_values
            self.update_points_and_text_info()
            self.update_primaries_triangle()
            self.update_spectral_visualizer()
            self.fig.canvas.draw_idle()
            print(f"brightness scaling failed: {exc}")
            return False

        return True


    def redraw_gamut(self, prev_values = None):
        is_error = self.update_points_and_text_info()

        if is_error:
            self.Y_R, self.Y_G, self.Y_B, self.Y_m, self.LAMBDA_M = prev_values
            self.update_points_and_text_info()

        self.update_primaries_triangle()
        self.update_spectral_visualizer()
        
        self.ax.set_xlim(*GAMUT_X_LIMITS)
        self.ax.set_ylim(*GAMUT_Y_LIMITS)
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
            
            if self.LAMBDA_M > self.MAX_LAMBDA: self.LAMBDA_M = self.MAX_LAMBDA
            else:           
                Point(self.ax, self.point_m.xy, "",  legend = 'mono', color = 'blue')        
                is_changed = True
                n_changed_channel = 3
                self.update_Y_s()


        elif event.key == 'M':        
            self.LAMBDA_M -= 1

            if self.LAMBDA_M in [self.LAMBDA_RED, self.LAMBDA_GREEN, self.LAMBDA_BLUE]: # чтобы длина монохроматической волны не была равна primaires
                self.LAMBDA_M -= 1

            if self.LAMBDA_M < self.MIN_LAMBDA: self.LAMBDA_M = self.MIN_LAMBDA
            else:                   
                is_changed = True
                n_changed_channel = 3
                self.update_Y_s()


        elif event.key in ('y', '1'):
            update_color_setter_mode = 1

        elif event.key in ('Y', 'shift+y', '2'):
            update_color_setter_mode = 2

        
        elif event.key == 'q':             # Закрываем окно
            plt.close(self.fig)
            
                
        if is_changed:

            self.redraw_gamut(prev_values)


        return is_changed, n_changed_channel, update_color_setter_mode
