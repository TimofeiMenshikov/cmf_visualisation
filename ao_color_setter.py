import threading
from datetime import datetime

from ao.ao_device import Channel
from ao.configs import ao_config


LOG_FILENAME = "log.txt"
OFF_CHANNEL = Channel(100, 0)
OFF_CHANNELS = (OFF_CHANNEL, OFF_CHANNEL, OFF_CHANNEL, OFF_CHANNEL)


def _make_channels(frequencies, powers):
    return (
        Channel(float(frequencies[0]), float(powers[0])),
        Channel(float(frequencies[1]), float(powers[1])),
        Channel(float(frequencies[2]), float(powers[2])),
        Channel(float(frequencies[3]), float(powers[3])),
    )


def _set_four_channels(ao_device, channels):
    if len(channels) != 4:
        raise ValueError(f"Expected exactly 4 AO channels, got {len(channels)}")

    ao_device.set_channels(*channels)


def _scale_channels(channels, scale):
    return tuple(
        Channel(channel.frequency_mhz, channel.amplitude_perc * scale)
        for channel in channels
    )


def _zero_channel(channel):
    return Channel(channel.frequency_mhz, 0)


def _limit_channels_to_max_el_power(ao_device, channels):
    validator = getattr(ao_device, "_validator", None)
    if validator is None or getattr(ao_device, "_no_validation", False):
        return channels, 1.0

    sum_power = validator._get_sum_el_power(*channels)
    if sum_power <= ao_config.MAX_EL_POWER:
        return channels, 1.0

    scale = ao_config.MAX_EL_POWER / sum_power
    limited_channels = _scale_channels(channels, scale)

    for _ in range(8):
        if validator._get_sum_el_power(*limited_channels) <= ao_config.MAX_EL_POWER:
            return limited_channels, scale

        scale *= 0.98
        limited_channels = _scale_channels(channels, scale)

    return limited_channels, scale


class AoColorSetter:
    def __init__(self, ao_device, get_frequency_and_power_func, period=1.5):
        self.ao = ao_device
        self.get_frequency_and_power = get_frequency_and_power_func
        self.period = period

        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._ao_lock = threading.Lock()
        self._log_lock = threading.Lock()

        self._current_settings = None
        self._current_mode = 1

    def start(self):
        if self._thread and self._thread.is_alive():
            print("AOColorSetter already started")
            return

        self._current_settings = self.get_frequency_and_power()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("AOColorSetter started")

    def _worker(self):
        while not self._stop_event.is_set():
            self.set_mode_1()

            if self._stop_event.wait(self.period):
                break

            self.set_mode_2()

            if self._stop_event.wait(self.period):
                break

    def update(self, frequencies, powers):
        with self._lock:
            self._current_settings = (frequencies, powers)
            current_mode = self._current_mode

        self._set_mode(current_mode)
        print("AO frequencies updated")

    def set_mode_1(self):
        self._set_mode(1)

    def set_mode_2(self):
        self._set_mode(2)

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
            print("AOColorSetter stopped")

        with self._ao_lock:
            _set_four_channels(self.ao, OFF_CHANNELS)

    def _set_mode(self, mode):
        red_channel, green_channel, blue_channel, mono_channel = self._get_channels()

        if mode == 1:
            channels = (
                red_channel,
                green_channel,
                _zero_channel(blue_channel),
                _zero_channel(mono_channel),
            )
            mode_name = "RED+GREEN"
        elif mode == 2:
            channels = (
                _zero_channel(red_channel),
                _zero_channel(green_channel),
                blue_channel,
                mono_channel,
            )
            mode_name = "BLUE+MONO"
        else:
            raise ValueError(f"Unknown AO color setter mode: {mode}")

        with self._lock:
            self._current_mode = mode

        with self._ao_lock:
            channels, power_scale = _limit_channels_to_max_el_power(self.ao, channels)
            _set_four_channels(self.ao, channels)

        self._log_channels(mode_name, channels, power_scale)

    def _get_channels(self):
        with self._lock:
            if self._current_settings is None:
                raise RuntimeError("AOColorSetter is not started")
            frequencies, powers = self._current_settings

        return _make_channels(frequencies, powers)

    def _log_channels(self, mode_name, active_channels, power_scale=1.0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_info = ", ".join(str(ch) for ch in active_channels)
        scale_info = "" if power_scale == 1.0 else f" | POWER_SCALE: {power_scale:.3f}"

        log_line = (
            f"[{timestamp}] MODE: {mode_name} | "
            f"ACTIVE: [{active_info}]{scale_info}\n"
        )

        with self._log_lock:
            with open(LOG_FILENAME, "a", encoding="utf-8") as file:
                file.write(log_line)


class AoColorSetterStatic:
    def __init__(self, ao_device, get_frequency_and_power_func):
        self.ao = ao_device
        self.get_frequency_and_power = get_frequency_and_power_func
        self._current_settings = None

    def start(self):
        self._current_settings = self.get_frequency_and_power()
        print("AOColorSetterStatic started")

    def update(self, frequencies, powers):
        self._current_settings = (frequencies, powers)
        print("AO frequencies updated")

    def set_mode_1(self):
        red_channel, green_channel, blue_channel, mono_channel = self._get_channels()
        channels = _limit_channels_to_max_el_power(
            self.ao,
            (
                red_channel,
                green_channel,
                _zero_channel(blue_channel),
                _zero_channel(mono_channel),
            ),
        )[0]

        _set_four_channels(self.ao, channels)
        self._log_channels("RED+GREEN", channels)

    def set_mode_2(self):
        red_channel, green_channel, blue_channel, mono_channel = self._get_channels()
        channels = _limit_channels_to_max_el_power(
            self.ao,
            (
                _zero_channel(red_channel),
                _zero_channel(green_channel),
                blue_channel,
                mono_channel,
            ),
        )[0]

        _set_four_channels(self.ao, channels)
        self._log_channels("BLUE+MONO", channels)

    def stop(self):
        _set_four_channels(self.ao, OFF_CHANNELS)

    def _get_channels(self):
        if self._current_settings is None:
            raise RuntimeError("AOColorSetterStatic is not started")

        frequencies, powers = self._current_settings
        return _make_channels(frequencies, powers)

    def _log_channels(self, mode_name, active_channels):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_info = ", ".join(str(ch) for ch in active_channels)

        log_line = (
            f"[{timestamp}] MODE: {mode_name} | "
            f"ACTIVE: [{active_info}]\n"
        )

        with open(LOG_FILENAME, "a", encoding="utf-8") as file:
            file.write(log_line)
