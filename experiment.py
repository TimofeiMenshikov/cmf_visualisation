import colour
import matplotlib.pyplot as plt

import numpy as np
import os



from ao.ao_device import AoDevice, Channel


from typing import Tuple, Optional, Union, List
from matplotlib.axes import Axes

from gamut import Gamut


# Тип для координат: список, кортеж или массив numpy
CoordType = Union[Tuple[float, float], List[float]]


               
   


class AoDeviceExperiment():
    def __init__(self, read_output = False, simulate = False, logger = None):
        self.ao = AoDevice(read_output, simulate, logger)
        self.gamut = Gamut()

        self.gamut.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)

        self.start_experiment()

        plt.tight_layout()
        plt.show()


    def start_experiment(self):
        self.ao.find_device()
        self.start_ao_device()
        print(self.ao.is_connected)


    def handle_key_press(self, event):
        is_changed, n_changed_channel = self.gamut.update_gamut(event)

        if is_changed:
            
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

        self.ao._send_single_channel_if_changed(n_channel, Channel(frequencies[n_channel], powers[n_channel]))

        print(f"выставлен канал {n_channel}, {frequencies[n_channel]} {powers[n_channel]}")


    def dump_info(self):

        self.gamut.dump_info()
        print("frequency and power")
        print(self.get_frequency_and_power())

            
def main():

    ao_experiment = AoDeviceExperiment(read_output=True, simulate=True)
    

if __name__ == "__main__":
    main()