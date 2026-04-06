import threading
import time
from ao.ao_device import AoDevice, Channel
from datetime import datetime

LOG_FILENAME = "log.txt"


class AoColorSetter:
    def __init__(self, ao_device, get_frequency_and_power_func, period=1.5):
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

            # Первая группа
            self.ao.set_channels(red_channel, green_channel)
            self._log_channels(
                "RED+GREEN",
                [red_channel, green_channel]
            )

            if self._stop_event.wait(self.period):
                break

            # Вторая группа
            self.ao.set_channels(blue_channel, mono_channel)
            self._log_channels(
                "BLUE+MONO",
                [blue_channel, mono_channel]
            )

            if self._stop_event.wait(self.period):
                break

    # -------------------------
    # Обновление частот
    # -------------------------
    def update(self, frequencies, powers):
        with self._lock:
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
    def __init__(self, ao_device, get_frequency_and_power_func):
        """
        ao_device — объект с методом set_channels(...)
        get_frequency_and_power_func — функция получения (frequencies, powers)
        period — период переключения в секундах
        """
        self.ao = ao_device
        self.get_frequency_and_power = get_frequency_and_power_func
        
        self._current_settings = None 


    def start(self):

        frequencies, powers = self.get_frequency_and_power()
        self._current_settings = (frequencies, powers)

        print("AOColorSetterStatic запущен")


    def update(self, frequencies, powers):
        
        self._current_settings = (frequencies, powers)
        print("Частоты обновлены")

    def set_mode_1(self):
            
        frequencies, powers = self._current_settings

        red_channel   = Channel(frequencies[0], powers[0])
        green_channel = Channel(frequencies[1], powers[1])
        blue_channel  = Channel(frequencies[2], powers[2])
        mono_channel  = Channel(frequencies[3], powers[3])

        self.ao.set_channels(red_channel, green_channel)
        self._log_channels(
            "RED+GREEN",
            [red_channel, green_channel]
        )


    def set_mode_2(self):
        frequencies, powers = self._current_settings

        red_channel   = Channel(frequencies[0], powers[0])
        green_channel = Channel(frequencies[1], powers[1])
        blue_channel  = Channel(frequencies[2], powers[2])
        mono_channel  = Channel(frequencies[3], powers[3])

        self.ao.set_channels(blue_channel, mono_channel)
        self._log_channels(
            "BLUE+MONO",
            [blue_channel, mono_channel]
        )


    def _log_channels(self, mode_name, active_channels):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        active_info = ", ".join(str(ch) for ch in active_channels)
        
        log_line = (
            f"[{timestamp}] MODE: {mode_name} | "
            f"ACTIVE: [{active_info}]\n"
            
        )

        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(log_line)