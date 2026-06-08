PARITY_EVEN = "E"
STOPBITS_ONE = 1


class SerialException(Exception):
    pass


class SerialTimeoutException(SerialException):
    pass


class Serial:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.is_open = True
        self._last_written = b""

    def write(self, data):
        self._last_written = bytes(data)
        return len(data)

    def read(self, size=1):
        if size <= 0:
            return b""
        return bytes([0xAA]) + bytes(max(0, size - 1))

    def readline(self):
        return b"temperature: 0\n"

    def reset_input_buffer(self):
        return None

    def close(self):
        self.is_open = False

