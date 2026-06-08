class Configuration:
    def __init__(
        self,
        width,
        polynomial,
        init_value,
        final_xor_value,
        reverse_input=False,
        reverse_output=False,
    ):
        self.width = width
        self.polynomial = polynomial
        self.init_value = init_value
        self.final_xor_value = final_xor_value
        self.reverse_input = reverse_input
        self.reverse_output = reverse_output


class Calculator:
    def __init__(self, config):
        self.config = config

    def checksum(self, data):
        crc = self.config.init_value
        top_bit = 1 << (self.config.width - 1)
        mask = (1 << self.config.width) - 1

        for byte in data:
            crc ^= byte << (self.config.width - 8)
            for _ in range(8):
                if crc & top_bit:
                    crc = ((crc << 1) ^ self.config.polynomial) & mask
                else:
                    crc = (crc << 1) & mask

        return crc ^ self.config.final_xor_value

