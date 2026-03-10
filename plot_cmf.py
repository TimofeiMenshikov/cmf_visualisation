import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import colour
from matplotlib.widgets import Slider

from ao.spectral_converter import SpectralConverter

from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline
from scipy.ndimage import gaussian_filter1d
from scipy import integrate


FILENAME = "data.txt"

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


def smooth_cmf(x, cmf, method='cubic', num_points=300):
    """
    Сглаживание кумулятивной функции распределения
    
    Parameters:
    -----------
    x : array-like
        Значения по оси X
    cmf : array-like
        Значения CMF (должны быть в [0, 1])
    method : str
        'cubic' - кубические сплайны
        'pchip' - монотонная интерполяция
        'bspline' - B-сплайны
        'gaussian' - гауссовское сглаживание
    num_points : int
        Количество точек в сглаженной кривой
    
    Returns:
    --------
    x_smooth, cmf_smooth : ndarray
        Сглаженные данные
    """
    x = np.array(x)
    cmf = np.array(cmf)
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    
    if method == 'cubic':
        cs = CubicSpline(x, cmf)
        cmf_smooth = cs(x_smooth)
    
    elif method == 'pchip':
        pchip = PchipInterpolator(x, cmf)
        cmf_smooth = pchip(x_smooth)
    
    elif method == 'bspline':
        spl = make_interp_spline(x, cmf, k=3)
        cmf_smooth = spl(x_smooth)
    
    elif method == 'gaussian':
        # Для gaussian нужна равномерная сетка
        cmf_smooth = gaussian_filter1d(cmf, sigma=2)
        x_smooth = x
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Гарантируем, что значения в [-1, 1]
    cmf_smooth = np.clip(cmf_smooth, -1, 1)
    
    return x_smooth, cmf_smooth

    
    
class Plot():

    def __init__(self, data, show_trackbars = True):

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

        self.ymin = -0.1
        self.ymax =  1


        self.data = data

        append_primaries_lambda_to_data(data)

        self.data.sort(key = lambda x: x[0])
        self.get_real_data()
        self.wavelength = [row[0] for row in self.data]

        self.V_lambda_dict = self.get_V_lambda_dict()

        self.normed_R, self.normed_G, self.normed_B = self.get_norm_cmf_by_integration()

        self.show_trackbars = show_trackbars
        self.__init_plot()
        
       
    @staticmethod
    def get_V_lambda_dict():
        # Get V(λ) at 1 nm intervals from 380-780 nm
        wavelengths = np.arange(380, 781, 1)
        V_lambda = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].values[:, 1]  # y̅ = V(λ)

        V_lambda_dict = dict(zip(wavelengths, V_lambda))

        print(V_lambda_dict)

        return V_lambda_dict
    

    @staticmethod
    def norm_y(y, x, target_area = 30, k = 1): # k - коэффициент на который делится target_area (учитывает что не вся ось x доступна)
        area = integrate.simpson(y, x)

        return (y / area) * (target_area / k)

    
    def get_norm_cmf_by_integration(self, mode = 'default'):  # вызывает функцию получающую cmf, нормирует cmf по площади и сохраняет отдельные масивы с каналами
        
        R, G, B = self.get_cmf()

        normed_R = self.norm_y(R, self.wavelength)
        normed_G = self.norm_y(G, self.wavelength)
        normed_B = self.norm_y(B, self.wavelength, k = 1.5)

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


    def get_cmf(self):

        R = []
        G = []
        B = []

        for i in range(len(data)):
            lam = data[i][LAMBDA_INDEX]

            int_r = data[i][I_R_INDEX]
            int_g = data[i][I_G_INDEX]
            int_b = data[i][I_B_INDEX]
            int_m = data[i][I_m_INDEX]

            cmf_r, cmf_g, cmf_b = self.norm_int(int_r, int_g, int_b, int_m, lam)

            R.append(cmf_r)
            G.append(cmf_g)
            B.append(cmf_b)

        return R, G, B


    def norm_int(self, int_r, int_g, int_b, int_m, lam): 

        sum_int = (int_r + int_g + int_b) * int_m

        if sum_int == 0: return 0, 0, 0

        int_r /= sum_int
        int_g /= sum_int
        int_b /= sum_int

        L = int_r * self.L_R + int_g * self.L_G + int_b * self.L_B

        cmf_r = int_r * self.V_lambda_dict[lam] / L
        cmf_g = int_g * self.V_lambda_dict[lam] / L
        cmf_b = int_b * self.V_lambda_dict[lam] / L

        return cmf_r, cmf_g, cmf_b


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


        x_smooth_R, cmf_smooth_R = smooth_cmf(self.wavelength, self.normed_R, method = 'cubic')
        x_smooth_G, cmf_smooth_G = smooth_cmf(self.wavelength, self.normed_G, method = 'cubic')
        x_smooth_B, cmf_smooth_B = smooth_cmf(self.wavelength, self.normed_B, method = 'cubic')

        # x_smooth_R, cmf_smooth_R = self.wavelength, self.normed_R
        # x_smooth_G, cmf_smooth_G = self.wavelength, self.normed_G
        # x_smooth_B, cmf_smooth_B = self.wavelength, self.normed_B

    
        self.line_R, = self.ax1.plot(x_smooth_R, cmf_smooth_R, color='red',   label = 'CMF red')
        self.line_G, = self.ax1.plot(x_smooth_G, cmf_smooth_G, color='green', label = 'CMF green')
        self.line_B, = self.ax1.plot(x_smooth_B, cmf_smooth_B, color='blue',  label = 'CMF blue')

        self.ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Линия x=0

        # вертикальные линии на длине волн primaries
        self.ax1.axvline(x=self.LAMBDA_RED,   color='red',   linestyle='--', linewidth=2, label = 'red primarie')
        self.ax1.axvline(x=self.LAMBDA_GREEN, color='green', linestyle='--', linewidth=2, label = 'green primarie')
        self.ax1.axvline(x=self.LAMBDA_BLUE,  color='blue',  linestyle='--', linewidth=2, label = 'blue primarie')

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("CMF value")

        self.ax1.xaxis.label.set_size(12)
        self.ax1.yaxis.label.set_size(12)

        self.ax1.set_title(
        "RGB Color Matching Functions",       
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

        self.normed_R, self.normed_G, self.normed_B = self.get_norm_cmf_by_integration()
        
        self.line_R.set_ydata(self.normed_R)
        self.line_G.set_ydata(self.normed_G)
        self.line_B.set_ydata(self.normed_B)

        self.fig.canvas.draw_idle()


    def show_plot(self):
        plt.show()



if __name__ == "__main__":

    data = read_data_from_file(FILENAME)

    plot = Plot(data, show_trackbars=False)

    #print(data)

    
    plot.show_plot()

