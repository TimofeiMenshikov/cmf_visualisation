import threading
import time
from ao.ao_device import AoDevice, Channel
from datetime import datetime

import winsound

LOG_FILENAME = "log.txt"

LAMBDA_RED   = 620
LAMBDA_GREEN = 530
LAMBDA_BLUE  = 470

class AoColorSetter:
    def __init__(self, ao_device, get_frequency_and_power_func, LAMBDA_M, period=1.5):
        """
        ao_device — объект с методом set_channels(...)
        get_frequency_and_power_func — функция получения (frequencies, powers)
        period — период переключения в секундах
        """
        self.ao = ao_device
        self.get_frequency_and_power = get_frequency_and_power_func
        self.period = period

        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._log_lock = threading.Lock()

        self.LAMBDA_M = LAMBDA_M

        self._current_settings = None

        

        
        

    # -------------------------
    # Запуск
    # -------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            print("AOColorSetter уже запущен")
            return

        frequencies, powers = self.get_frequency_and_power()
        self._current_settings = (frequencies, powers)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("AOColorSetter запущен")

    # -------------------------
    # Основной поток
    # -------------------------
    def _worker(self):
        empty = Channel(0, 0)

        while not self._stop_event.is_set():

            with self._lock:
                frequencies, powers = self._current_settings

            red_channel   = Channel(frequencies[0], powers[0])
            green_channel = Channel(frequencies[1], powers[1])
            blue_channel  = Channel(frequencies[2], powers[2])
            mono_channel  = Channel(frequencies[3], powers[3])

            


            
            if self.LAMBDA_M <= LAMBDA_BLUE or self.LAMBDA_M >= LAMBDA_RED:

                first_channel    = red_channel
                second_channel   = blue_channel

                first_channel_2  = green_channel
                second_channel_2 = mono_channel

                text1 = "red + blue"
                text2 = "green + mono"

            elif self.LAMBDA_M <= LAMBDA_GREEN:

                first_channel    = blue_channel
                second_channel   = green_channel

                first_channel_2  = red_channel
                second_channel_2 = mono_channel
        
                text1 = "blue + green"
                text2 = "red + mono"

            elif self.LAMBDA_M <= LAMBDA_RED:
            
                first_channel    = red_channel
                second_channel   = green_channel

                first_channel_2  = blue_channel
                second_channel_2 = mono_channel 
            
                text1 = "red + green"
                text2 = "blue + mono"

            self._log_channels(
                text1,
                [first_channel, second_channel]
            )

            # Первая группа
            self.ao.set_channels(first_channel, second_channel)
            print(text1)
            winsound.Beep(5000, 200)
            

            if self._stop_event.wait(self.period):
                break

            # Вторая группа
            self.ao.set_channels(first_channel_2, second_channel_2)
            print(text2)
            winsound.Beep(6000, 200)
            

            self._log_channels(
                text2,
                [first_channel_2, second_channel_2]
            )


            if self._stop_event.wait(self.period):
                break

    # -------------------------
    # Обновление частот
    # -------------------------
    def update(self, frequencies, powers, LAMBDA_M):
        with self._lock:
            self.LAMBDA_M = LAMBDA_M
            self._current_settings = (frequencies, powers)
        print("Частоты обновлены")

    # -------------------------
    # Остановка
    # -------------------------
    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            print("AOColorSetter остановлен")


    def _log_channels(self, mode_name, active_channels):
        """
        mode_name — строка (например: 'RED+GREEN')
        active_channels — список активных каналов
        inactive_channels — список выключенных каналов
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        active_info = ", ".join(str(ch) for ch in active_channels)
        

        log_line = (
            f"[{timestamp}] MODE: {mode_name} | "
            f"ACTIVE: [{active_info}]\n"
            
        )

        with self._log_lock:
            with open("log.txt", "a", encoding="utf-8") as f:
                f.write(log_line)


class AoColorSetterStatic:
    def __init__(self, ao_device, get_frequency_and_power_func, LAMBDA_M):
        """
        ao_device — объект с методом set_channels(...)
        get_frequency_and_power_func — функция получения (frequencies, powers)
        period — период переключения в секундах
        """
        self.ao = ao_device
        self.get_frequency_and_power = get_frequency_and_power_func
        
        self.LAMBDA_M = LAMBDA_M

        self._current_settings = None  

        self.color_mode = 1 

    def start(self):


        frequencies, powers = self.get_frequency_and_power()
        self._current_settings = (frequencies, powers)

        print("AOColorSetterStatic запущен") 

    def update(self, frequencies, powers, LAMBDA_M):
        
        self.LAMBDA_M = LAMBDA_M
        self._current_settings = (frequencies, powers)
        self.set_color_mode(self.color_mode)
        print("Частоты обновлены")


    def _log_channels(self, mode_name, active_channels):
        """
        mode_name — строка (например: 'RED+GREEN')
        active_channels — список активных каналов
        inactive_channels — список выключенных каналов
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        active_info = ", ".join(str(ch) for ch in active_channels)
        

        log_line = (
            f"[{timestamp}] MODE: {mode_name} | "
            f"ACTIVE: [{active_info}]\n"
            
        )

        
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)


    def set_color_mode(self, mode = 1):
        

        self.color_mode = mode        

            
        frequencies, powers = self._current_settings

        red_channel   = Channel(frequencies[0], powers[0])
        green_channel = Channel(frequencies[1], powers[1])
        blue_channel  = Channel(frequencies[2], powers[2])
        mono_channel  = Channel(frequencies[3], powers[3])

        


        
        if self.LAMBDA_M <= LAMBDA_BLUE or self.LAMBDA_M >= LAMBDA_RED:

            first_channel    = red_channel
            second_channel   = blue_channel

            first_channel_2  = green_channel
            second_channel_2 = mono_channel

            text1 = "red + blue"
            text2 = "green + mono"

        elif self.LAMBDA_M <= LAMBDA_GREEN:

            first_channel    = blue_channel
            second_channel   = green_channel

            first_channel_2  = red_channel
            second_channel_2 = mono_channel
    
            text1 = "blue + green"
            text2 = "red + mono"

        elif self.LAMBDA_M <= LAMBDA_RED:
        
            first_channel    = red_channel
            second_channel   = green_channel

            first_channel_2  = blue_channel
            second_channel_2 = mono_channel 
        
            text1 = "red + green"
            text2 = "blue + mono"

        if mode == 1:

            self._log_channels(
                text1,
                [first_channel, second_channel]
            )

            # Первая группа
            self.ao.set_channels(first_channel, second_channel)
            print(text1)

        elif mode == 2:
        

            self._log_channels(
                text2,
                [first_channel_2, second_channel_2]
            )
        



            # Вторая группа
            self.ao.set_channels(first_channel_2, second_channel_2)
            print(text2)
            winsound.Beep(6000, 200)

        
