import colour
import matplotlib.pyplot as plt
import numpy as np
from functools import partial



LAMBDA_RED   = 620
LAMBDA_GREEN = 530
LAMBDA_BLUE  = 470

# Пример использования:
RGB_PRIMARIES = [
    [LAMBDA_RED,   255, 0, 0],    # Красный
    [LAMBDA_GREEN, 0, 255, 0],    # Зелёный
    [LAMBDA_BLUE,  0, 0, 255]     # Синий
]

REAL_COLORS_RATIO = 0.25
ANTI_COLOR_RATIO = 0.3

REAL_COLORS_RATIO_STEP = 0.05
ANTI_COLOR_RATIO_STEP  = 0.05

EPS = 0.001 # для сравнения вещественных чисел


def is_bigger(x1, x2): return x1 > x2 - EPS
def is_lower(x1, x2):  return x1 < x2 + EPS



def get_xy_values(data_log):
    # 1. Нормализуем нелинейные RGB значения (диапазон [0, 1])
    rgb_nonlinear = np.array([
        [entry[1] / 255, entry[2] / 255, entry[3] / 255]
        for entry in data_log
    ])
    
    # 2. Преобразуем нелинейный sRGB → линейный RGB
    # Рекомендуемый способ для версий ≥0.4:
    rgb_linear = colour.RGB_COLOURSPACES['sRGB'].cctf_decoding(rgb_nonlinear)
    
    # 3. Преобразуем линейный RGB → XYZ
    # ВАЖНО: не передаём illuminant как строку — он берётся из цветового пространства
    xyz_primaries = colour.RGB_to_XYZ(
        rgb_linear,
        colourspace=colour.RGB_COLOURSPACES['sRGB']  # illuminant уже содержится в цветовом пространстве
    )
    
    # 4. Преобразуем XYZ → координаты цветности xy
    xy_primaries = colour.XYZ_to_xy(xyz_primaries)

    return xy_primaries


def wavelength_to_xy(wavelength):
    """
    Преобразует длину волны (нм) в координаты цветности CIE xy.
    
    Параметры:
        wavelength (float): Длина волны в нанометрах (380–780)
    
    Возвращает:
        np.ndarray: [x, y] координаты на диаграмме CIE 1931
    """
    # 1. Создаём "дельта-спектр" — монохроматический свет заданной длины волны
    #    {длина_волны: интенсивность}
    sd = colour.SpectralDistribution({wavelength: 1.0}, name=f'{wavelength}nm')
    
    # 2. Конвертируем спектр в XYZ с использованием стандартного наблюдателя CIE 1931
    xyz = colour.sd_to_XYZ(
        sd,
        cmfs=colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    )
    
    # 3. Преобразуем XYZ → xy координаты цветности
    xy = colour.XYZ_to_xy(xyz)
    return xy


def plot_cie1931_gamut():

    global RGB_PRIMARIES

    xy_primaries = get_xy_values(RGB_PRIMARIES)

    # 5. Строим диаграмму CIE 1931
    fig, ax = colour.plotting.plot_chromaticity_diagram_CIE1931(
        standalone=False,
        show_diagram_colours=False,
        show_spectral_locus=True,
        title='Цветовой гамут на диаграмме CIE 1931',
        bounding_box=(-0.1, 0.8, -0.1, 0.9)
    )
    
    
    # 7. Отмечаем вершины с подписями
    labels = ['R (Красный)', 'G (Зелёный)', 'B (Синий)']
    for (x, y), label in zip(xy_primaries, labels):
        ax.plot(x, y, 'ro', markersize=3, color='black')
        ax.annotate(label, (x, y), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='black')
    
    # 8. Добавляем белую точку D65 (берём из цветового пространства)
    white_point = colour.RGB_COLOURSPACES['sRGB'].whitepoint
    ax.plot(white_point[0], white_point[1], 'wo', markersize=12, 
            markeredgecolor='black', markeredgewidth=1.5, label='Белая точка D65')
    
    ax.legend(loc='upper right')

    return fig, ax, xy_primaries


def draw_lines_to_lambda(ax, lam, xy_primaries):

    if lam < 390 or lam > 700:
        print("incorrect lambda")
        return
    

    xy_red   = xy_primaries[0]
    xy_green = xy_primaries[1]
    xy_blue  = xy_primaries[2]

    if lam <= LAMBDA_BLUE or lam >= LAMBDA_RED:
        point1 = [xy_red[0], xy_blue[0]]
        point2 = [xy_red[1], xy_blue[1]]

    elif lam <= LAMBDA_GREEN:
        point1 = [xy_green[0], xy_blue[0]]
        point2 = [xy_green[1], xy_blue[1]]

    elif lam <= LAMBDA_RED:
        point1 = [xy_blue[0], xy_red[0]]
        point2 = [xy_blue[1], xy_red[1]]


    
    ax.plot(
        point1,  # x-координаты: от зелёного к синему
        point2,  # y-координаты: от зелёного к синему
        color='cyan',               # Цвет линии (циан для контраста)
        linewidth=3,
        linestyle='-',
        marker='o',                 # Кружки в точках
        markersize=8,        
    )


def draw_monochroma_point(ax, lam):
    mon_xy = wavelength_to_xy(lam)

    ax.plot(mon_xy[0], mon_xy[1], 'ro', markersize=3, color='black') 


def get_real_and_anti_color_points(lam, xy_primaries):
    if lam < 390 or lam > 700:
        print("incorrect lambda")
        return
    
    xy_red   = xy_primaries[0]
    xy_green = xy_primaries[1]
    xy_blue  = xy_primaries[2]

    if lam <= LAMBDA_BLUE or lam >= LAMBDA_RED:
        return xy_red, xy_blue, xy_green

    elif lam <= LAMBDA_GREEN:
        return xy_green, xy_blue, xy_red

    elif lam <= LAMBDA_RED:
        return xy_blue, xy_red, xy_green
    

def get_point_xy(real_and_anti_color_points, real_colors_ratio, anti_color_ratio):

    real_color1, real_color2, anti_color = real_and_anti_color_points

    real_color1_x, real_color1_y = real_color1
    real_color2_x, real_color2_y = real_color2
    anti_color_x, anti_color_y   = anti_color

    point_x = real_color1_x * real_colors_ratio + real_color2_x * (1 - real_colors_ratio)
    point_y = real_color1_y * real_colors_ratio + real_color2_y * (1 - real_colors_ratio)

    delta_anti_color_x = point_x - anti_color_x
    delta_anti_color_y = point_y - anti_color_y

    final_point_x = point_x + delta_anti_color_x * anti_color_ratio
    final_point_y = point_y + delta_anti_color_y * anti_color_ratio

    return final_point_x, final_point_y, point_x, point_y
    

def draw_point(fig, ax, lam, xy_primaries, real_colors_ratio, anti_color_ratio, line, point):

    real_and_anti_color_points = get_real_and_anti_color_points(lam, xy_primaries)

    final_x, final_y, x, y = get_point_xy(real_and_anti_color_points, real_colors_ratio, anti_color_ratio)

    
    anti_color = real_and_anti_color_points[-1]
    anti_color_x = anti_color[0]
    anti_color_y = anti_color[1]

    

    point.set_data([x], [y])

    line.set_ydata([final_y, anti_color_y])          # Меняем только y-данные
    line.set_xdata([final_x, anti_color_x])


    

def on_key(event, ax, fig, line, lam, xy_primaries, point):

    global REAL_COLORS_RATIO, ANTI_COLOR_RATIO

    is_changed = False

    if event.key == 'r':  # Обновляем данные
        
        line.set_color('red')            
        REAL_COLORS_RATIO += REAL_COLORS_RATIO_STEP

        if is_lower(REAL_COLORS_RATIO, 1): is_changed = True
        else:                      REAL_COLORS_RATIO -= REAL_COLORS_RATIO_STEP
            
        print("REAL_COLORS_RATIO", REAL_COLORS_RATIO)
        
        
    elif event.key == 'R':

        line.set_color('yellow')          
        REAL_COLORS_RATIO -= REAL_COLORS_RATIO_STEP
        is_changed = True
        
    elif event.key == 'a':

        line.set_color('green')          
        ANTI_COLOR_RATIO += ANTI_COLOR_RATIO_STEP
        is_changed = True

    elif event.key == 'A':

        line.set_color('blue')
        ANTI_COLOR_RATIO -= ANTI_COLOR_RATIO_STEP
        is_changed =True

    
    elif event.key == 'q':             # Закрываем окно
        plt.close(fig)

    if is_changed:
        draw_point(fig, ax, lam, xy_primaries, REAL_COLORS_RATIO, ANTI_COLOR_RATIO, line, point)

        ax.relim()                     # Пересчитываем границы осей
        ax.autoscale_view()            # Применяем новые границы
        fig.canvas.draw_idle()         # Запрашиваем перерисовку [[7]]

    
def main():
    fig, ax, xy_primaries = plot_cie1931_gamut()

    lam = 500



    draw_lines_to_lambda(ax, lam, xy_primaries)

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
    #draw_monochroma_point(ax, lam)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
