import colour
import matplotlib.pyplot as plt

import numpy as np
import os

from matplotlib.widgets import Button, Slider, TextBox
from ao.ao_device import AoDevice, Channel
from ao_color_setter import AoColorSetter
from gamut import Gamut


class AoDeviceExperiment():
    def __init__(self, read_output = False, simulate = False, logger = None):

        self.DATA_FILENAME = "data.txt"
        self.IS_SAVED_DATA = False

        self.MIN_LAMBDA = 443.4
        self.MAX_LAMBDA = 743.6

        self.ao = AoDevice(read_output, simulate, logger)

        self.ao_color_setter = AoColorSetter(self.ao, self.get_frequency_and_power)

        self.gamut = Gamut()

        self.gamut.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)

        self.textbox, self.textbox_info_text = self.__init_text_box()
        self.save_button, self.reset_button = self.__init_button()

        self.start_experiment()

        plt.tight_layout()
        plt.show()


    def start_experiment(self):
        self.ao.find_device()
        #self.start_ao_device()
        print(self.ao.is_connected)

        self.ao_color_setter.start()


    def __init_text_box(self):                        # текстовое поле - аналог нажатия клавишами поэтому его инициализация находится вне гамута
        ax_textbox = plt.axes([0.68, 0.7, 0.05, 0.05]) 

        textbox = TextBox(ax_textbox, 'Lambda mono', initial=str(self.gamut.LAMBDA_M))
        textbox.text_disp.set_fontsize(12)

        x, y = textbox.label.get_position()

        textbox.label.set_position((x - 0.5, y))  # больше отрицательное — дальше от поля
        textbox.label.set_fontsize(12)

        textbox.on_submit(self.get_lambda_text_box)

        textbox_info_text = self.gamut.ax2.text(0.1, 0.65, '', fontsize = 12)

        return textbox, textbox_info_text


    def __init_button(self):
        ax_save_button = plt.axes([0.55, 0.15, 0.1, 0.04], facecolor='red') 
        ax_reset_button = plt.axes([0.75, 0.15, 0.1, 0.04], facecolor='red')
    

        save_button = Button(ax_save_button, 'Save to file') 
        reset_button = Button(ax_reset_button, 'Reset experiment')
        
        save_button.on_clicked(self.save_to_file)
        reset_button.on_clicked(self.reset_experiment)

        return save_button, reset_button


    def get_lambda_text_box(self, text):
        
        # Преобразуем текст в float

        try:
            lam = float(text)
        except ValueError:
            self.textbox_info_text.set_text(f"lam must be float and <= {self.MAX_LAMBDA} and >= {self.MIN_LAMBDA}")
            return
        
        if lam >= self.MIN_LAMBDA and lam <= self.MAX_LAMBDA:

            print(lam)
            prev_lam = self.gamut.LAMBDA_M # если ошибка в колориметре то возвращаем на место
            self.gamut.LAMBDA_M = lam
            print("self.gamut.Lambda_m", self.gamut.LAMBDA_M)

            n_changed_channel = 3 # так как монохроматический цвет


            # тут должна быть проверка на частоты колориметра (либо можно поправить глобальные максимумы и минимумы lambda)

            self.gamut.update_Y_s()

            self.update_ao_device(n_changed_channel)
            self.IS_SAVED_DATA = False

            self.textbox_info_text.set_text("succesfully changed lam")

            self.gamut.redraw_gamut()
            self.dump_info()

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

    def reset_experiment(self, event):

        

        self.gamut.update_Y_s()

        self.gamut.redraw_gamut()

        self.IS_SAVED_DATA = False

        


        print("reset button is pressed")


    def handle_key_press(self, event):
        is_changed, n_changed_channel = self.gamut.update_gamut(event)

        if is_changed:
            
            self.IS_SAVED_DATA = False

            self.update_ao_device(n_changed_channel)
            self.dump_info()


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


    def dump_info(self):

        self.gamut.dump_info()
        print("frequency and power")
        print(self.get_frequency_and_power())

            
def main():

    ao_experiment = AoDeviceExperiment(read_output=True, simulate=True)
    

if __name__ == "__main__":
    main()