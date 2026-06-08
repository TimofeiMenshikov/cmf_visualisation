import numpy as np
import colour
from scipy.optimize import nnls
from scipy import integrate
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, Slider

def get_y_bar_auto():
    """
    Извлекает стандартную функцию яркости Y (CIE 1931).
    Использует ту же сетку длин волн, что и стандартные CMF в colour-science.
    """
    cmf_cie = colour.colorimetry.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    # values[:, 1] — это столбец y_bar (соответствует V(lambda))
    y_bar = cmf_cie.values[:, 1]
    return y_bar

def normalize_rgb_cmf(rgb_values):
    """
    Нормирует кривые CMF (R, G, B) так, чтобы площадь под каждой была равна 1.0.
    Это гарантирует, что равноэнергетический белый (1,1,1) будет 
    находиться в центре хроматического графика (1/3, 1/3).
    
    Parameters:
    rgb_values (np.ndarray): Массив формы (N, 3), где столбцы - это R, G, B.
    
    Returns:
    np.ndarray: Нормированные значения CMF.
    """
    # Создаем копию, чтобы не менять исходные данные
    norm_values = rgb_values.copy()
    
    for i in range(3):
        # Вычисляем площадь под кривой i-го канала (методом трапеций)
        area = np.trapezoid(norm_values[:, i])
        
        if area != 0:
            norm_values[:, i] = norm_values[:, i] / area
        else:
            print(f"Предупреждение: Площадь канала {i} равна нулю!")
            
    return norm_values

def get_cmf_rgb_values():
    """
    Возвращает матрицу RGB CMF и соответствующие длины волн.
    """
    # Матрица перехода для системы CIE RGB (1931)
    
    M_RGB_to_XYZ = colour.models.RGB_COLOURSPACE_CIE_RGB.matrix_RGB_to_XYZ

    print("FORWARD MATRIX")
    print(M_RGB_to_XYZ)

    M_XYZ_to_RGB = np.linalg.inv(M_RGB_to_XYZ)

    print("BACKWARD MATRIX")
    print(M_XYZ_to_RGB)

    cmfs_xyz = colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    wavelengths = cmfs_xyz.wavelengths
    
    # Переводим XYZ в RGB: (N, 3) @ (3, 3).T -> (N, 3)
    rgb_values = np.dot(cmfs_xyz.values, M_XYZ_to_RGB.T)

    #rgb_values = normalize_rgb_cmf(rgb_values)

    return rgb_values, wavelengths

def get_cmf_xyz_values(rgb_values, M_RGB_to_XYZ):
    return np.dot(rgb_values, M_RGB_to_XYZ.T)

def get_luminance_coefficients(y_bar, cmf_rgb_basis):
    """
    Находит коэффициенты L_R, L_G, L_B.
    Матрица cmf_rgb_basis должна иметь форму (N, 3).
    """
    # Решаем систему: CMF_RGB * [L_R, L_G, L_B]^T = y_bar
    coeffs, _ = nnls(cmf_rgb_basis, y_bar)
    return coeffs[0], coeffs[1], coeffs[2]


def find_vertex_X(A, B, C, L_R, L_G, L_B):
    """
    Находит X как пересечение касательной и линии Алихны.
    """
    # Уравнение 1 (Касательная): A*r + B*g = -C
    # Уравнение 2 (Алихна): (L_R - L_B)*r + (L_G - L_B)*g = -L_B
    
    M = np.array([
        [A, B],
        [L_R - L_B, L_G - L_B]
    ])
    
    Y = np.array([-C, -L_B])
    
    try:
        coords = np.linalg.solve(M, Y)
        r_x, g_x = coords[0], coords[1]
        b_x = 1 - r_x - g_x
        
        return (r_x, g_x), np.array([r_x, g_x, b_x])
    except np.linalg.LinAlgError:
        return None



def get_line_coeffs(p1, p2):
    """Находит A, B, C для прямой Ax + Bg + C = 0 через две точки."""
    r1, g1 = p1
    r2, g2 = p2
    A = g1 - g2
    B = r2 - r1
    C = r1 * g2 - r2 * g1
    return [A, B, C]

def adjust_line_coeffs(line_coeffs, delta_C):
    line_coeffs[-1] += delta_C
    return line_coeffs

def find_intersection(line1, line2):
    """Решает систему для двух линий: A*r + B*g + C = 0"""
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    W = A1 * B2 - A2 * B1
    if abs(W) < 1e-12: return None
    r = (B1 * C2 - B2 * C1) / W
    g = (A2 * C1 - A1 * C2) / W
    return np.array([r, g, 1 - r - g])

def get_area(target_wl_X, target_wl_Z, rgb_values, wavelengths):

    r_bar, g_bar, b_bar = rgb_values.T
    sum_rgb = r_bar + g_bar + b_bar
    r_locus = r_bar / sum_rgb
    g_locus = g_bar / sum_rgb

    # 2. Вспомогательная функция для касательных
    def get_tangent_data(target_wl):
        idx = np.abs(wavelengths - target_wl).argmin()
        r0, g0 = r_locus[idx], g_locus[idx]
        dr = r_locus[idx+1] - r_locus[idx-1]
        dg = g_locus[idx+1] - g_locus[idx-1]
        # Вторая точка для построения линии
        r1, g1 = r0 + dr, g0 + dg
        return get_line_coeffs((r0, g0), (r1, g1))
    
    # Получаем коэффициенты линий
    line_XY = get_tangent_data(target_wl_X)
    line_ZY = get_tangent_data(target_wl_Z)

    #############################
    #Небольшое перемещение по нормали 

    
    line_XY = adjust_line_coeffs(line_XY, 0.2)

    



    ###########################

    line_XZ = (L_R - L_B, L_G - L_B, L_B) # Алихна

    # 3. Находим вершины XYZ
    vec_X = find_intersection(line_XY, line_XZ)
    vec_Z = find_intersection(line_ZY, line_XZ)
    vec_Y = find_intersection(line_XY, line_ZY)

    if vec_X is None or vec_Y is None or vec_Z is None:
        return np.inf

    def calculate_triangle_area(vec_X, vec_Y, vec_Z):
        """
        Считает площадь треугольника XYZ на плоскости (r, g).
        Принимает векторы вида [r, g, b].
        """
        # Берем только координаты r и g
        x1, y1 = vec_X[0], vec_X[1]
        x2, y2 = vec_Y[0], vec_Y[1]
        x3, y3 = vec_Z[0], vec_Z[1]
        
        # Формула Гаусса (через определитель 2x2)
        area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        
        return area
    
    area = calculate_triangle_area(vec_X, vec_Y, vec_Z)

    return area


def get_min_area_target_wl(rgb_values, wavelengths):

    min_area = 1000000000
    min_target_wl_Z = 480
    min_target_wl_X = 700


    ###############
    # для стандартных cmf range(480, 515) для z и range(695, 705) для x

    for target_wl_Z in range(480, 515):
        for target_wl_X in range(600, 620):
            area = get_area(target_wl_X, target_wl_Z, rgb_values, wavelengths)

            print("area:", area, "target_wl_Z:", target_wl_Z)

            if not np.isfinite(area):
                continue

            if area < min_area:
                min_area = area
                min_target_wl_Z, min_target_wl_X = target_wl_Z, target_wl_X

    if not np.isfinite(min_area):
        raise ValueError("No valid target wavelength pair found for gamut construction.")

    print("TARGET WL Z", min_target_wl_Z)
    print("TARGET WL X", min_target_wl_X)
    return min_target_wl_X, min_target_wl_Z


def plot_full_construction(L_R, L_G, L_B, rgb_values, wavelengths):
    r_bar, g_bar, b_bar = rgb_values.T
    sum_rgb = r_bar + g_bar + b_bar
    r_locus = r_bar / sum_rgb
    g_locus = g_bar / sum_rgb

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.49)

    # 1. Спектральный локус
    ax.plot(r_locus, g_locus, 'k-', linewidth=2, label='Спектральный локус', zorder=1)

    # 2. Вспомогательная функция для касательных
    def get_tangent_data(target_wl):
        idx = np.abs(wavelengths - target_wl).argmin()
        r0, g0 = r_locus[idx], g_locus[idx]
        dr = r_locus[idx + 1] - r_locus[idx - 1]
        dg = g_locus[idx + 1] - g_locus[idx - 1]
        # Вторая точка для построения линии
        r1, g1 = r0 + dr, g0 + dg
        return get_line_coeffs((r0, g0), (r1, g1)), np.array([r0, g0])

    target_wl_X, target_wl_Z = get_min_area_target_wl(rgb_values, wavelengths)

    # если известна длина волны с наименьшей площадью
    target_wl_Z = 486
    target_wl_X = 611

    # Получаем базовые коэффициенты линий. Дальше ползунки двигают их
    # через shift_line() и rotate_line().
    base_line_XY, pivot_XY = get_tangent_data(target_wl_X)
    base_line_YZ, pivot_YZ = get_tangent_data(target_wl_Z)
    base_line_XZ = (L_R - L_B, L_G - L_B, L_B) # Алихна

    def shift_line(line_coeffs, distance):
        """
        Сдвигает прямую Ax + By + C = 0 на расстояние distance по нормали.
        """
        A, B, C = line_coeffs
        norm_length = math.sqrt(A ** 2 + B ** 2)
        new_C = C - distance * norm_length
        return A, B, new_C

    def rotate_line(line_coeffs, pivot, angle_degrees):
        """
        Поворачивает прямую вокруг pivot на angle_degrees.
        """
        A, B, C = line_coeffs
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        new_A = A * cos_a - B * sin_a
        new_B = A * sin_a + B * cos_a
        new_C = -(new_A * pivot[0] + new_B * pivot[1])

        return new_A, new_B, new_C

    base_vec_X = find_intersection(base_line_XY, base_line_XZ)
    base_vec_Z = find_intersection(base_line_YZ, base_line_XZ)

    if base_vec_X is not None and base_vec_Z is not None:
        pivot_XZ = (base_vec_X[:2] + base_vec_Z[:2]) / 2
    else:
        pivot_XZ = np.array([1/3, 1/3])

    # 3. Интерактивная отрисовка линий, вершин и треугольника.
    r_ext = np.linspace(-0.8, 2.0, 200)

    def line_points(line_coeffs):
        A, B, C = line_coeffs

        if abs(B) > 1e-12:
            return r_ext, -(A * r_ext + C) / B

        # На случай вертикальной прямой.
        r = np.full_like(r_ext, -C / A)
        return r, np.linspace(-1.0, 1.8, len(r_ext))

    ax_shift_XY = plt.axes([0.18, 0.32, 0.68, 0.025])
    ax_shift_YZ = plt.axes([0.18, 0.27, 0.68, 0.025])
    ax_shift_XZ = plt.axes([0.18, 0.22, 0.68, 0.025])
    ax_rotate_XY = plt.axes([0.18, 0.16, 0.68, 0.025])
    ax_rotate_YZ = plt.axes([0.18, 0.11, 0.68, 0.025])
    ax_rotate_XZ = plt.axes([0.18, 0.06, 0.68, 0.025])

    slider_XY = Slider(ax_shift_XY, 'shift line_XY', -1.0, 1.0, valinit=0.0)
    slider_YZ = Slider(ax_shift_YZ, 'shift line_YZ', -1.0, 1.0, valinit=0.0)
    slider_XZ = Slider(ax_shift_XZ, 'shift line_XZ', -1.0, 1.0, valinit=0.0)
    rotate_XY = Slider(ax_rotate_XY, 'rotate line_XY', -45.0, 45.0, valinit=0.0)
    rotate_YZ = Slider(ax_rotate_YZ, 'rotate line_YZ', -45.0, 45.0, valinit=0.0)
    rotate_XZ = Slider(ax_rotate_XZ, 'rotate line_XZ', -45.0, 45.0, valinit=0.0)
    fig.shift_sliders = [slider_XY, slider_YZ, slider_XZ]
    fig.rotate_sliders = [rotate_XY, rotate_YZ, rotate_XZ]

    ax_save = plt.axes([0.18, 0.38, 0.16, 0.04])
    ax_reset = plt.axes([0.37, 0.38, 0.16, 0.04])
    save_button = Button(ax_save, 'save config')
    reset_button = Button(ax_reset, 'reset')
    fig.control_buttons = [save_button, reset_button]

    status_text = fig.text(0.56, 0.392, '', fontsize=9, va='center')

    def get_control_values():
        return {
            'shift_line_XY': slider_XY.val,
            'shift_line_YZ': slider_YZ.val,
            'shift_line_XZ': slider_XZ.val,
            'rotate_line_XY': rotate_XY.val,
            'rotate_line_YZ': rotate_YZ.val,
            'rotate_line_XZ': rotate_XZ.val,
        }

    def save_config(_):
        config_path = os.path.join(os.path.dirname(__file__), 'config.txt')

        with open(config_path, 'w', encoding='utf-8') as config_file:
            config_file.write('# XYZ gamut interactive controls\n')

            for key, value in get_control_values().items():
                config_file.write(f'{key} = {value:.10f}\n')

            config_file.write('\n')
            config_file.write('# Current M_xyz_to_rgb\n')

            if fig.current_M_xyz_to_rgb is None:
                config_file.write('M_xyz_to_rgb = None\n')
            else:
                for row in fig.current_M_xyz_to_rgb:
                    values = ' '.join(f'{value:.10f}' for value in row)
                    config_file.write(f'M_xyz_to_rgb_row = {values}\n')

        status_text.set_text(f'saved: {os.path.basename(config_path)}')
        fig.canvas.draw_idle()

    def reset_controls(_):
        for control in [
            slider_XY,
            slider_YZ,
            slider_XZ,
            rotate_XY,
            rotate_YZ,
            rotate_XZ,
        ]:
            control.reset()

        status_text.set_text('reset')
        fig.canvas.draw_idle()

    def get_shifted_lines():
        line_XY = rotate_line(base_line_XY, pivot_XY, rotate_XY.val)
        line_YZ = rotate_line(base_line_YZ, pivot_YZ, rotate_YZ.val)
        line_XZ = rotate_line(base_line_XZ, pivot_XZ, rotate_XZ.val)

        line_XY = shift_line(line_XY, slider_XY.val)
        line_YZ = shift_line(line_YZ, slider_YZ.val)
        line_XZ = shift_line(line_XZ, slider_XZ.val)
        return line_XY, line_YZ, line_XZ

    def get_vertices(line_XY, line_YZ, line_XZ):
        vec_X = find_intersection(line_XY, line_XZ)
        vec_Y = find_intersection(line_XY, line_YZ)
        vec_Z = find_intersection(line_YZ, line_XZ)
        return vec_X, vec_Y, vec_Z

    def get_matrix(vec_X, vec_Y, vec_Z):
        if any(v is None for v in [vec_X, vec_Y, vec_Z]):
            return None
        return np.column_stack([vec_X, vec_Y, vec_Z])

    line_XY, line_YZ, line_XZ = get_shifted_lines()
    vec_X, vec_Y, vec_Z = get_vertices(line_XY, line_YZ, line_XZ)
    M_xyz_to_rgb = get_matrix(vec_X, vec_Y, vec_Z)

    x_xy, y_xy = line_points(line_XY)
    x_yz, y_yz = line_points(line_YZ)
    x_xz, y_xz = line_points(line_XZ)

    line_XY_artist, = ax.plot(x_xy, y_xy, 'r:', alpha=0.8, label='line_XY')
    line_YZ_artist, = ax.plot(x_yz, y_yz, 'b:', alpha=0.8, label='line_YZ')
    line_XZ_artist, = ax.plot(x_xz, y_xz, 'g--', alpha=0.8, label='line_XZ')

    triangle_artist, = ax.plot([], [], linewidth=2, color='magenta',
                               label='Базис XYZ', zorder=3)
    triangle_fill = Polygon(np.empty((0, 2)), closed=True, color='magenta',
                            alpha=0.15, zorder=2)
    ax.add_patch(triangle_fill)

    vertex_artists = [
        ax.scatter([], [], color='magenta', s=20,
                   edgecolors='black', zorder=5)
        for _ in range(3)
    ]
    vertex_labels = [
        ax.annotate(name, (0, 0), textcoords="offset points",
                    xytext=(-15, -15), fontweight='bold', fontsize=12)
        for name in ['X', 'Y', 'Z']
    ]

    matrix_text = ax.text(
        0.02, 0.02, '', transform=ax.transAxes, fontsize=8,
        family='monospace', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    def update_triangle(vec_X, vec_Y, vec_Z):
        if any(v is None for v in [vec_X, vec_Y, vec_Z]):
            triangle_artist.set_data([], [])
            triangle_fill.set_xy(np.empty((0, 2)))
            matrix_text.set_text('Линии параллельны: матрица не считается')
            fig.current_M_xyz_to_rgb = None
            callback = getattr(fig, "on_matrix_change", None)
            if callback is not None:
                callback(None)

            for artist, label in zip(vertex_artists, vertex_labels):
                artist.set_offsets(np.empty((0, 2)))
                label.set_visible(False)
            return

        pts = np.array([vec_X[:2], vec_Y[:2], vec_Z[:2], vec_X[:2]])
        triangle_artist.set_data(pts[:, 0], pts[:, 1])
        triangle_fill.set_xy(pts[:-1])
        fig.current_M_xyz_to_rgb = get_matrix(vec_X, vec_Y, vec_Z)
        matrix_text.set_text(np.array2string(fig.current_M_xyz_to_rgb,
                                             precision=4))
        callback = getattr(fig, "on_matrix_change", None)
        if callback is not None:
            callback(fig.current_M_xyz_to_rgb)

        for artist, label, point in zip(
            vertex_artists, vertex_labels, [vec_X, vec_Y, vec_Z]
        ):
            artist.set_offsets([point[:2]])
            label.xy = point[:2]
            label.set_visible(True)

    def update(_):
        line_XY, line_YZ, line_XZ = get_shifted_lines()

        for artist, line in [
            (line_XY_artist, line_XY),
            (line_YZ_artist, line_YZ),
            (line_XZ_artist, line_XZ),
        ]:
            x_line, y_line = line_points(line)
            artist.set_data(x_line, y_line)

        vec_X, vec_Y, vec_Z = get_vertices(line_XY, line_YZ, line_XZ)
        update_triangle(vec_X, vec_Y, vec_Z)
        fig.canvas.draw_idle()

    update_triangle(vec_X, vec_Y, vec_Z)
    slider_XY.on_changed(update)
    slider_YZ.on_changed(update)
    slider_XZ.on_changed(update)
    rotate_XY.on_changed(update)
    rotate_YZ.on_changed(update)
    rotate_XZ.on_changed(update)
    save_button.on_clicked(save_config)
    reset_button.on_clicked(reset_controls)

    # 4. Аннотации длин волн
    for wl in [450, 480, 520, 550, 600]:
        idx = np.abs(wavelengths - wl).argmin()
        ax.scatter(r_locus[idx], g_locus[idx], c='gray', s=20)
        ax.annotate(f'{wl}', (r_locus[idx], g_locus[idx]), fontsize=8, alpha=0.7)

    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlim(-0.8, 2)
    ax.set_ylim(-1, 1.8)
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_title('Гамут в пространстве rg')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    return M_xyz_to_rgb

def normalize_to_white_point(M_xyz_to_rgb):
    """
    Масштабирует столбцы матрицы XYZ -> RGB так, чтобы 
    белая точка (1,1,1) в XYZ переходила в (1,1,1) в RGB.
    """
    # 1. Целевой вектор белого в RGB
    white_rgb = np.array([1.0, 1.0, 1.0])
    
    # 2. Решаем систему уравнений M * S = white_rgb
    # S — это вектор масштабирующих коэффициентов для [X, Y, Z]
    S = np.linalg.solve(M_xyz_to_rgb, white_rgb)
    
    # 3. Умножаем каждый столбец матрицы на соответствующий коэффициент
    # Это меняет длину базисных векторов, не меняя их направления (геометрии)
    M_normalized = M_xyz_to_rgb * S
    
    return M_normalized


def calculate_cmf_from_gamut_matrix(rgb_matrix, M_xyz_to_rgb):
    """
    Пересчитывает XYZ CMF из RGB CMF и текущих вершин XYZ на гамуте.
    """
    if M_xyz_to_rgb is None:
        return None, None

    try:
        M_normalized = normalize_to_white_point(M_xyz_to_rgb)
        M_rgb_to_xyz = np.linalg.inv(M_normalized)
    except np.linalg.LinAlgError:
        return None, None

    cmf_xyz = get_cmf_xyz_values(rgb_matrix, M_rgb_to_xyz)
    max_y = max(cmf_xyz[:, 1])

    if abs(max_y) < 1e-12:
        return None, None

    cmf_xyz /= max_y
    return cmf_xyz, M_rgb_to_xyz


def plot_cmf_on_axes(wavelengths, values, labels=None, colors=None, ax=None, linestyle='-', transparency = 1):
    """
    Отрисовка функций сложения на существующих осях.
    """
    if ax is None:
        ax = plt.gca() # Если оси не переданы, берем текущие
        
    if colors is None:
        colors = [(1, 0, 0, transparency), (0.01, 1, 0.01, transparency), (0, 0, 1, transparency)]
        
    if labels is None:
        labels = [r'$\bar{x}(\lambda)$', r'$\bar{y}(\lambda)$', r'$\bar{z}(\lambda)$']

    
    
    line_artists = []

    for i in range(3):
        area = np.trapezoid(values[:, i], x=wavelengths)
        
        
        # Рисуем линию
        line, = ax.plot(wavelengths, values[:, i], 
                        color=colors[i], linewidth=2, 
                        label=f'{labels[i]}', linestyle = linestyle)
        line_artists.append(line)
        
        # Заливка
        #ax.fill_between(wavelengths, values[:, i], color=colors[i], alpha=0.1)

    ax.set_xlabel('Длина волны (нм)')
    ax.set_ylabel('Интенсивность')

    ax.set_title(
        "Сравнение функций соответствия цветов XYZ",       
        fontsize=14,              # Размер    
        )
    ax.axhline(0, color='black', lw=1, alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    return line_artists


def get_standard_xyz_coordinates():
    """
    Содержит стандартную матрицу и вычисляет r, g координаты вершин XYZ.
    """
    # 1. Стандартная матрица пересчета из XYZ в CIE RGB (700, 546.1, 435.8 нм)
    # Эти значения фиксированы стандартом 1931 года.
    M_xyz_to_rgb = np.array([
        [ 0.41847, -0.15866, -0.082835],
        [-0.091169, 0.25243,  0.015708],
        [ 0.0009209, -0.0025498, 0.17860]
    ])

    # 2. Вершины X, Y, Z в системе RGB — это столбцы этой матрицы.
    # Но чтобы получить их в нормированном виде (r + g + b = 1), 
    # нам нужно отнормировать каждый столбец отдельно.
    
    vertices_rgb = []
    names = ['X', 'Y', 'Z']
    
    for i in range(3):
        # Берем i-й столбец (это вектор базиса XYZ в координатах RGB)
        vec = M_xyz_to_rgb[:, i]
        
        # Нормируем вектор на сумму его компонент, чтобы получить r, g, b
        s = np.sum(vec)
        r = vec[0] / s
        g = vec[1] / s
        b = vec[2] / s
        
        vertices_rgb.append([r, g, b])
        print(f"Вершина {names[i]}: r = {r:.4f}, g = {g:.4f}")

    return np.array(vertices_rgb)

def plot_xyz_on_current_gamut(vertices_rgb):
    """
    Отрисовывает полученные координаты поверх текущего графика.
    """
    ax = plt.gca()
    
    r_coords = vertices_rgb[:, 0]
    g_coords = vertices_rgb[:, 1]
    names = ['X', 'Y', 'Z']

    # Отрисовка точек
    ax.scatter(r_coords, g_coords, color='red', s=20, marker='X', zorder=20, label='Std XYZ')

    # Аннотации
    for i, name in enumerate(names):
        ax.annotate(f"{name}_std", (r_coords[i], g_coords[i]), 
                    xytext=(-20, 10), textcoords='offset points',
                    color='red', fontweight='bold')

# Вызов (используем ранее найденные L)
# plot_full_construction(0.1769, 0.8124, 0.0107, rgb_matrix, wavelengths)


def calc_delta_E_mono_color_equal_energy(cmfs_xyz, my_cmf, wavelengths):

    x_bar, y_bar, z_bar = cmfs_xyz.T
    my_x_bar, my_y_bar, my_z_bar = my_cmf.T

    white_e = np.array([1/3, 1/3])

    print("wl de2000 xyz_cie xyz_my")

    Y_norm = 100 # нормировка на яркость для стандартного наблюдателя

    de2000_arr = []

    for i in range(len(wavelengths)):
        xyz_cie = np.array([x_bar[i], y_bar[i], z_bar[i]])
        xyz_cie_normed = xyz_cie / y_bar[i] * Y_norm

        eps = 1e-7 
        xyz_my = np.clip(np.array([my_x_bar[i], my_y_bar[i], my_z_bar[i]]), eps, None)
        xyz_my_normed = xyz_my / y_bar[i] * Y_norm

        
        lab_cie = colour.XYZ_to_Lab(xyz_cie_normed, illuminant=white_e)
        lab_my = colour.XYZ_to_Lab(xyz_my_normed, illuminant=white_e)
    
        de2000 = colour.delta_E(lab_cie, lab_my, method='CIE 2000')

        de2000_arr.append(de2000)

        print(wavelengths[i], de2000, xyz_cie_normed, xyz_my_normed)

    print("____________________")
    de2000_arr.sort()
    print("max DE", max(de2000_arr))
    print("av DE", sum(de2000_arr) / len(de2000_arr))
    print("med DE", de2000_arr[len(de2000_arr) // 2])
    print("____________________")


def calculate_delta_E_values(cmfs_xyz, my_cmf, wavelengths):
    """
    Считает delta E 2000 для текущих CMF без вывода в консоль.
    """
    x_bar, y_bar, z_bar = cmfs_xyz.T
    my_x_bar, my_y_bar, my_z_bar = my_cmf.T

    white_e = np.array([1/3, 1/3])
    Y_norm = 100
    eps = 1e-7
    de2000_arr = []

    for i in range(len(wavelengths)):
        if abs(y_bar[i]) < eps:
            de2000_arr.append(np.nan)
            continue

        xyz_cie = np.array([x_bar[i], y_bar[i], z_bar[i]])
        xyz_cie_normed = xyz_cie / y_bar[i] * Y_norm

        xyz_my = np.clip(
            np.array([my_x_bar[i], my_y_bar[i], my_z_bar[i]]),
            eps,
            None
        )
        xyz_my_normed = xyz_my / y_bar[i] * Y_norm

        lab_cie = colour.XYZ_to_Lab(xyz_cie_normed, illuminant=white_e)
        lab_my = colour.XYZ_to_Lab(xyz_my_normed, illuminant=white_e)
        de2000 = colour.delta_E(lab_cie, lab_my, method='CIE 2000')
        de2000_arr.append(float(de2000))

    return np.array(de2000_arr)


def format_delta_E_summary(de2000_arr, wavelengths):
    """
    Формирует короткую сводку delta E для вывода рядом с графиком.
    """
    valid_mask = np.isfinite(de2000_arr)

    if not np.any(valid_mask):
        return "delta E\nнет данных"

    valid_de = de2000_arr[valid_mask]
    valid_wl = wavelengths[valid_mask]
    max_idx = np.argmax(valid_de)

    return (
        "delta E 2000\n"
        f"max: {valid_de[max_idx]:.2f} @ {valid_wl[max_idx]:.0f} nm\n"
        f"avg: {np.mean(valid_de):.2f}\n"
        f"med: {np.median(valid_de):.2f}"
    )


# --- Исполняемый блок ---

# 1. Получаем данные
y_bar = get_y_bar_auto()

###################################
# # для стандартных CMF
# # касательные в точках 503 (505) и 703 (700)
 
# # Важно: распаковываем кортеж на матрицу и массив длин волн
# rgb_matrix, wavelengths = get_cmf_rgb_values()

# # 2. Считаем коэффициенты яркости (L)
# L_R, L_G, L_B = get_luminance_coefficients(y_bar, rgb_matrix)

#################################################


#####################
# для полученных CMF

import os

#L_R, L_G, L_B = 0.34028889, 0.64125798, 0.06360552
L_R, L_G, L_B = 0.28560719640179916, 0.646176911544228, 0.06821589205397302


def load_cmf_from_txt(filename="normalized_cmf.txt"):
    """
    Читает файл и возвращает кортеж из 4 массивов.
    """
    if not os.path.isabs(filename):
        filename = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден.")

    # Загружаем данные
    data = np.loadtxt(filename)
    
    # Распаковываем колонки
    w = data[:, 0]
    r = data[:, 1]
    g = data[:, 2]
    b = data[:, 3]
    
    return w, r, g, b

def save_cmf_to_txt(wavelength, cmf_R, cmf_G, cmf_B, filename="normalized_cmf.txt"):
    """
    Сохраняет массивы длин волн и CMF в текстовый файл.
    
    Параметры:
    wavelength, cmf_R, cmf_G, cmf_B : array_like (массивы одинаковой длины)
    filename : str (имя файла для сохранения)
    """
    if not os.path.isabs(filename):
        filename = os.path.join(os.path.dirname(__file__), filename)

    # 1. Объединяем массивы в столбцы (N строк, 4 колонки)
    # Используем column_stack, чтобы каждый массив стал отдельным столбцом
    data = np.column_stack((wavelength, cmf_R, cmf_G, cmf_B))
    
    # 2. Формируем заголовок (header)
    # По умолчанию np.savetxt добавит '#' перед этой строкой
    header = "Wavelength(nm)\tR_CMF\tG_CMF\tB_CMF"
    
    # 3. Сохраняем данные
    # fmt='%d %.8f %.8f %.8f' означает:
    # %d — целое число для длин волн
    # %.8f — число с плавающей точкой и 8 знаками после запятой для значений CMF
    # delimiter='\t' — разделение табуляцией для красоты
    np.savetxt(filename, data, header=header, fmt='%d %.8f %.8f %.8f', delimiter='\t')
    
    print(f"Данные успешно сохранены в файл: {os.path.abspath(filename)}")




def plot_only_points_rg(M_rgb_to_xyz): # нужно для отрисовки точек на гамуте для другой матрицы преобразования
    ax = plt.gca()
    
    # 1. Основные цвета (Primaries) в пространстве rg
    # По определению системы rg, они всегда в этих точках:
    r_coords = [1, 0, 0]
    g_coords = [0, 1, 0]
    colors = ['red', 'green', 'blue']
    labels = ['Red Primary', 'Green Primary', 'Blue Primary']
    
    # Рисуем основные цвета
    for i in range(3):
        ax.scatter(r_coords[i], g_coords[i], c=colors[i], s=20, 
                   edgecolors='black', label=labels[i], zorder=5)

    # 2. Точка белого (White Point)
    # В RGB пространстве точка белого — это сумма вкладов R, G и B (1, 1, 1)
    # В координатах rg это всегда:
    # r = 1 / (1 + 1 + 1) = 1/3
    # g = 1 / (1 + 1 + 1) = 1/3
    r_w, g_w = 1/3, 1/3
    
    ax.scatter(r_w, g_w, c='white', s=20, edgecolors='black', 
               marker='X', label='White Point (1/3, 1/3)', zorder=6)
    







wavelengths, cmf_r, cmf_g, cmf_b = load_cmf_from_txt()

rgb_matrix = np.array([cmf_r, cmf_g, cmf_b]).T

########################







M_xyz_to_rgb = plot_full_construction(L_R, L_G, L_B, rgb_matrix, wavelengths)
gamut_fig = plt.gcf()
vertices_rgb = get_standard_xyz_coordinates()
#plot_xyz_on_current_gamut(vertices_rgb)

M_rgb_to_xyz_std = np.array([[  7.37045780e-01  , 1.15732670e-01  , 1.47221550e-01],
 [  3.28649394e-01 ,  6.02788891e-01  , 6.85617149e-02],
 [  1.63893399e-04  ,  2.94821109e-02  , 9.70353996e-01]])

#plot_only_points_rg(M_rgb_to_xyz_std)



print("получившеяся матрица")
print(M_xyz_to_rgb)


my_cmf_xyz, M_rgb_to_xyz = calculate_cmf_from_gamut_matrix(
    rgb_matrix, M_xyz_to_rgb
)

cmfs_xyz = colour.colorimetry.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

cmfs_xyz = cmfs_xyz[wavelengths]

if my_cmf_xyz is not None:
    calc_delta_E_mono_color_equal_energy(cmfs_xyz, my_cmf_xyz, wavelengths)

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.78)





my_cmf_lines = plot_cmf_on_axes(
    wavelengths,
    my_cmf_xyz,
    labels=['CMF X', 'CMF Y', 'CMF Z'],
    ax=ax
)
plot_cmf_on_axes(
    wavelengths,
    cmfs_xyz,
    labels=['CMF X CIE 1931', 'CMF Y CIE 1931', 'CMF Z CIE 1931'],
    transparency=0.5,
    ax=ax
)

delta_E_text = ax.text(
    1.03,
    0.98,
    '',
    transform=ax.transAxes,
    va='top',
    ha='left',
    fontsize=10,
    family='monospace',
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.8')
)


def update_cmf_plot(M_xyz_to_rgb):
    my_cmf_xyz, M_rgb_to_xyz = calculate_cmf_from_gamut_matrix(
        rgb_matrix, M_xyz_to_rgb
    )

    if my_cmf_xyz is None:
        for line in my_cmf_lines:
            line.set_ydata(np.full_like(wavelengths, np.nan, dtype=float))
        ax.set_title("CMF не считается: матрица вырождена")
        delta_E_text.set_text("delta E\nматрица\nвырождена")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        return

    for i, line in enumerate(my_cmf_lines):
        line.set_ydata(my_cmf_xyz[:, i])

    ax.set_title("Сравнение функций соответствия цветов XYZ")
    de2000_arr = calculate_delta_E_values(cmfs_xyz, my_cmf_xyz, wavelengths)
    delta_E_text.set_text(format_delta_E_summary(de2000_arr, wavelengths))
    ax.relim()
    ax.autoscale_view()
    fig.current_cmf_xyz = my_cmf_xyz
    fig.current_M_rgb_to_xyz = M_rgb_to_xyz
    fig.current_delta_E = de2000_arr
    fig.canvas.draw_idle()


gamut_fig.on_matrix_change = update_cmf_plot
update_cmf_plot(M_xyz_to_rgb)
plt.show()


print(M_rgb_to_xyz)
