import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg-cache"))

import colour
import matplotlib.pyplot as plt

import numpy as np

current_dir = Path(__file__).resolve().parent

target_dir = current_dir.parent / "ao-system"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(target_dir) not in sys.path:
    sys.path.insert(1, str(target_dir))

from matplotlib.widgets import Button, Slider, TextBox
from ao.ao_device import AoDevice, Channel
from ao_color_setter import AoColorSetter
from ui_constants import TEXT_BOX_COORDINATES, PANEL_BG, PANEL_EDGE, TEXT_COLOR, MUTED_TEXT
from gamut import Gamut
from constants import CALIBRATION_NAME, CALIBRATION_WAVELENGTH_RANGE

BUTTON_COLOR = "#2563eb"
BUTTON_HOVER_COLOR = "#1d4ed8"
RESET_COLOR = "#475569"
RESET_HOVER_COLOR = "#334155"


class AoDeviceExperiment():
    def __init__(self, read_output = False, simulate = True, logger = None, visualize_spectra = True):

        self.DATA_FILENAME = current_dir / "data.txt"
        self.IS_SAVED_DATA = False

        self.MIN_LAMBDA, self.MAX_LAMBDA = CALIBRATION_WAVELENGTH_RANGE

        if logger is None:
            self.ao = AoDevice(read_output=read_output, simulate=simulate)
        else:
            self.ao = AoDevice(read_output=read_output, simulate=simulate, logger=logger)

        self.ao_color_setter = AoColorSetter(self.ao, self.get_frequency_and_power, period=1.5)


        self.visualize_spectra = visualize_spectra
        self.gamut = Gamut(visualize_spectra=self.visualize_spectra)
        print(f"calibration: {CALIBRATION_NAME}, lambda range: {self.MIN_LAMBDA:g}-{self.MAX_LAMBDA:g} nm")

        self._shutdown_started = False
        self.gamut.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)
        self.gamut.fig.canvas.mpl_connect('close_event', self.safe_shutdown)

        self.textbox, self.textbox_info_text = self.__init_text_box()
        (
            self.save_button,
            self.reset_button,
            self.show_cmf_button,
            self.show_metrics_button,
            self.scale_up_button,
            self.scale_down_button,
        ) = self.__init_button()

        self.start_experiment()

        plt.show()


    def safe_shutdown(self, event=None):
        if self._shutdown_started:
            return

        self._shutdown_started = True
        try:
            self.ao_color_setter.stop()
            if self.ao.is_connected():
                self.ao.turn_off_preamp()
        except Exception as exc:
            print(f"AO safe shutdown failed: {exc}")


    def start_experiment(self):
        port = self.ao.find_device()
        print(f"ao connected: {self.ao.is_connected()}, port: {port}")

        if self.ao.is_connected():
            self.ao.turn_on_preamp()

        self.ao_color_setter.start()


    def __init_text_box(self):                        # текстовое поле - аналог нажатия клавишами поэтому его инициализация находится вне гамута

        #if self.visualize_spectra: return None, None

        #plt.figure(self.gamut.fig.number)  # Делаем fig1 активной
        ax_textbox = plt.axes(TEXT_BOX_COORDINATES, facecolor=PANEL_BG)
        for spine in ax_textbox.spines.values():
            spine.set_color(PANEL_EDGE)

        textbox = TextBox(ax_textbox, 'Lambda mono', initial=str(self.gamut.LAMBDA_M), color=PANEL_BG, hovercolor="#eef2ff")
        textbox.text_disp.set_fontsize(12)
        textbox.text_disp.set_color(TEXT_COLOR)

        textbox.label.set_horizontalalignment("right")
        textbox.label.set_fontsize(12)
        textbox.label.set_color(MUTED_TEXT)

        textbox.on_submit(self.get_lambda_text_box)

        textbox_info_text = self.gamut.ax3.text(
            0.04,
            0.04,
            '',
            fontsize=10,
            color=MUTED_TEXT,
            transform=self.gamut.ax3.transAxes,
        )

        return textbox, textbox_info_text


    def __init_button(self):

        #if self.visualize_spectra: return None, None

        plt.figure(self.gamut.fig.number)  # Делаем fig1 активной
        ax_cmf_button = plt.axes([0.60, 0.585, 0.095, 0.04], facecolor=BUTTON_COLOR)
        ax_metrics_button = plt.axes([0.707, 0.585, 0.112, 0.04], facecolor=BUTTON_COLOR)
        ax_save_button = plt.axes([0.831, 0.585, 0.057, 0.04], facecolor=BUTTON_COLOR)
        ax_reset_button = plt.axes([0.90, 0.585, 0.06, 0.04], facecolor=RESET_COLOR)
        ax_scale_up_button = plt.axes([0.60, 0.525, 0.045, 0.035], facecolor=BUTTON_COLOR)
        ax_scale_down_button = plt.axes([0.60, 0.485, 0.045, 0.035], facecolor=RESET_COLOR)
    

        show_cmf_button = Button(ax_cmf_button, 'Show CMF', color=BUTTON_COLOR, hovercolor=BUTTON_HOVER_COLOR)
        show_metrics_button = Button(ax_metrics_button, 'show_metrics', color=BUTTON_COLOR, hovercolor=BUTTON_HOVER_COLOR)
        save_button = Button(ax_save_button, 'Save', color=BUTTON_COLOR, hovercolor=BUTTON_HOVER_COLOR)
        reset_button = Button(ax_reset_button, 'Reset', color=RESET_COLOR, hovercolor=RESET_HOVER_COLOR)
        scale_up_button = Button(ax_scale_up_button, 'Y+', color=BUTTON_COLOR, hovercolor=BUTTON_HOVER_COLOR)
        scale_down_button = Button(ax_scale_down_button, 'Y-', color=RESET_COLOR, hovercolor=RESET_HOVER_COLOR)

        for button in (show_cmf_button, show_metrics_button, save_button, reset_button, scale_up_button, scale_down_button):
            button.label.set_fontsize(10)
            button.label.set_color("white")
            button.label.set_fontweight("bold")

        for ax_button in (ax_cmf_button, ax_metrics_button, ax_save_button, ax_reset_button, ax_scale_up_button, ax_scale_down_button):
            for spine in ax_button.spines.values():
                spine.set_visible(False)

        
        show_cmf_button.on_clicked(self.show_cmf)
        show_metrics_button.on_clicked(self.show_metrics)
        save_button.on_clicked(self.save_to_file)
        reset_button.on_clicked(self.reset_experiment)
        scale_up_button.on_clicked(self.scale_luminances_up)
        scale_down_button.on_clicked(self.scale_luminances_down)

        return save_button, reset_button, show_cmf_button, show_metrics_button, scale_up_button, scale_down_button

    def scale_luminances_up(self, event):
        if self.gamut.scale_luminances(self.gamut.Y_scale_coeff):
            self.IS_SAVED_DATA = False
            self.update_ao_device_all()
            self.__sync_metrics_button_label()
            self.dump_info()

    def scale_luminances_down(self, event):
        if self.gamut.scale_luminances(1 / self.gamut.Y_scale_coeff):
            self.IS_SAVED_DATA = False
            self.update_ao_device_all()
            self.__sync_metrics_button_label()
            self.dump_info()

    def show_cmf(self, event):
        if self.gamut.ax2_mode == "cmf":
            self.gamut.show_spectra()
            self.show_cmf_button.label.set_text("Show CMF")
        else:
            self.gamut.show_cmf_rgb()
            self.show_cmf_button.label.set_text("Show Spectra")
        self.gamut.fig.canvas.draw_idle()

    def show_metrics(self, event):
        from plot_cmf import FILENAME, Plot, read_data_from_file

        if getattr(self.gamut, "info_panel_mode", "experiment") == "metrics":
            self.gamut.show_intensity_info()
            self.show_metrics_button.label.set_text("show_metrics")
            self.gamut.fig.canvas.draw_idle()
            return

        try:
            cmf_plot = Plot(read_data_from_file(FILENAME), show_trackbars=False, build_plot=False)
            xyz_cie_white, xyz_my_white, white_de = cmf_plot.calc_delta_E_cmf_equal_energy()
            mono_summary = cmf_plot.calc_delta_E_mono_color_equal_energy(print_results=False)
            checker_summary = cmf_plot.calc_delta_E_color_checker(print_results=False)
            spectral_summary = cmf_plot.calc_spectral_metrics(print_results=False)
        except Exception as exc:
            self.gamut.show_metrics_info(["Metrics error", str(exc)])
            self.show_metrics_button.label.set_text("show_intensity")
            self.gamut.fig.canvas.draw_idle()
            return

        lines = (
            "delta E 2000",
            f"white E: {float(white_de):.2f}",
            f"CIE XYZ {self.__format_xyz(xyz_cie_white)}",
            f"my  XYZ {self.__format_xyz(xyz_my_white)}",
            f"mono max: {mono_summary['max']:.2f} @ {mono_summary['max_wavelength']:.0f} nm",
            f"mono avg/med: {mono_summary['average']:.2f} / {mono_summary['median']:.2f}",
            f"CC {checker_summary['illuminant']} max: {checker_summary['max']:.2f}",
            f"CC patch: {checker_summary['max_patch']}",
            f"CC avg/med: {checker_summary['average']:.2f} / {checker_summary['median']:.2f}",
            f"SAM XYZ: {self.__format_channels(spectral_summary['sam'])}",
            f"SAM avg: {spectral_summary['sam']['average']:.2f} deg",
            f"NMAE XYZ: {self.__format_channels(spectral_summary['nmae'])}",
            f"NMAE avg: {spectral_summary['nmae']['average']:.3f}",
        )
        self.gamut.show_metrics_info(lines)
        self.show_metrics_button.label.set_text("show_intensity")
        self.gamut.fig.canvas.draw_idle()

    @staticmethod
    def __format_xyz(xyz):
        return f"{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}"

    @staticmethod
    def __format_channels(values):
        return f"{values['X']:.2f}, {values['Y']:.2f}, {values['Z']:.2f}"

    def __sync_metrics_button_label(self):
        if getattr(self.gamut, "info_panel_mode", "experiment") == "metrics":
            self.show_metrics_button.label.set_text("show_intensity")
        else:
            self.show_metrics_button.label.set_text("show_metrics")


    def get_lambda_text_box(self, text):
        
        # Преобразуем текст в float

        try:
            lam = float(text)
        except ValueError:
            self.textbox_info_text.set_text(f"lam must be float and <= {self.MAX_LAMBDA} and >= {self.MIN_LAMBDA}")
            return
        
        if lam >= self.MIN_LAMBDA and lam <= self.MAX_LAMBDA:

            print(lam)
            prev_values = [
                self.gamut.Y_R,
                self.gamut.Y_G,
                self.gamut.Y_B,
                self.gamut.Y_m,
                self.gamut.LAMBDA_M,
            ]

            n_changed_channel = 3 # так как монохроматический цвет


            # тут должна быть проверка на частоты колориметра (либо можно поправить глобальные максимумы и минимумы lambda)

            try:
                self.gamut.LAMBDA_M = lam
                print("self.gamut.Lambda_m", self.gamut.LAMBDA_M)

                self.gamut.update_Y_s()

                self.update_ao_device(n_changed_channel)
                self.IS_SAVED_DATA = False

                self.textbox_info_text.set_text("succesfully changed lam")

                self.gamut.redraw_gamut()
                self.__sync_metrics_button_label()
                self.dump_info()
            except Exception as exc:
                self.gamut.Y_R, self.gamut.Y_G, self.gamut.Y_B, self.gamut.Y_m, self.gamut.LAMBDA_M = prev_values
                self.gamut.redraw_gamut()
                self.__sync_metrics_button_label()
                self.textbox_info_text.set_text(f"lambda update failed: {exc}")
                print(f"lambda update failed: {exc}")

        else:

            self.textbox_info_text.set_text(f"lam must be <= {self.MAX_LAMBDA} and >= {self.MIN_LAMBDA}")
            
            
    def save_to_file(self, event):

        I_R, I_G, I_B, I_m = self.gamut.get_intensities_from_Y()

        data = f"{self.gamut.LAMBDA_M} {I_R} {I_G} {I_B} {I_m}"

        if not self.IS_SAVED_DATA:

            # Проверяем, нужно ли добавлять перевод строки перед записью
            if os.path.exists(self.DATA_FILENAME) and os.path.getsize(self.DATA_FILENAME) > 0:
                with open(self.DATA_FILENAME, 'rb') as f:
                    f.seek(-1, 2)  # Переходим к последнему байту
                    last_char = f.read(1)
                    needs_newline = last_char != b'\n'
            else:
                needs_newline = False
            
            # Записываем данные
            with open(self.DATA_FILENAME, 'a', encoding='utf-8') as f:
                if needs_newline:
                    f.write('\n')
                f.write(data)
            print("save button is pressed")

            self.IS_SAVED_DATA = True
        
        if self.gamut.ax2_mode == "cmf":
            self.gamut.show_cmf_rgb()

    def reset_experiment(self, event):

        self.gamut.update_Y_s()

        self.gamut.redraw_gamut()
        self.__sync_metrics_button_label()

        self.IS_SAVED_DATA = False

        print("reset button is pressed")


    def handle_key_press(self, event):
        is_changed, n_changed_channel, update_color_setter_mode = self.gamut.update_gamut(event)

        if is_changed:
            
            self.IS_SAVED_DATA = False

            self.update_ao_device(n_changed_channel)
            self.__sync_metrics_button_label()
            self.dump_info()

        elif update_color_setter_mode in (1, 2):
            if update_color_setter_mode == 1:
                self.ao_color_setter.set_mode_1()
            else:
                self.ao_color_setter.set_mode_2()

            if self.textbox_info_text is not None:
                self.textbox_info_text.set_text("2 colors active")


    def get_frequency_and_power(self): # получает значения частоты и мощности из параметров яркости и длин волн
        Y_s     = np.array([self.gamut.Y_R, self.gamut.Y_G, self.gamut.Y_B, self.gamut.Y_m])
        lambdas = np.array([self.gamut.LAMBDA_RED, self.gamut.LAMBDA_GREEN, self.gamut.LAMBDA_BLUE, self.gamut.LAMBDA_M]) 

        return self.gamut.get_frequency_and_power_from_Y_wavelength(Y_s, lambdas)
    

    def start_ao_device(self):
        
        frequencies, powers = self.get_frequency_and_power()
        
        # 0 канал - красный цвет 1 канал - зеленый 2 канал - синий 3 канал - монохроматическая волна

        red_channel   = Channel(frequencies[0], powers[0])
        green_channel = Channel(frequencies[1], powers[1])
        blue_channel  = Channel(frequencies[2], powers[2])
        mono_channel  = Channel(frequencies[3], powers[3]) 

        self.ao.set_channels(red_channel, green_channel, blue_channel, mono_channel)
        print(f"выставлены каналы {red_channel} {green_channel} {blue_channel} {mono_channel}")
        

    def update_ao_device(self, n_channel):

        frequencies, powers = self.get_frequency_and_power()

        self.ao_color_setter.update(frequencies, powers)

        #self.ao._send_single_channel_if_changed(n_channel, Channel(frequencies[n_channel], powers[n_channel]))

        print(f"выставлен канал {n_channel}, {frequencies[n_channel]} {powers[n_channel]}")


    def update_ao_device_all(self):

        frequencies, powers = self.get_frequency_and_power()
        self.ao_color_setter.update(frequencies, powers)

        print("обновлены все каналы")
        print(frequencies, powers)


    def dump_info(self):

        self.gamut.dump_info()
        print("frequency and power")
        print(self.get_frequency_and_power())

            
def main():

    ao_experiment = AoDeviceExperiment(read_output=True, simulate=False)
    

if __name__ == "__main__":
    main()
