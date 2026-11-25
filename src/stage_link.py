import os
import select
import termios
import threading
import time


class ModbusError(Exception):
    pass


def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


class ModbusRtu:
    def __init__(self, path, baud=115200):
        self.path = path
        self.baud = baud
        self._fd = None
        self._lock = threading.Lock()

    def open(self):
        if self._fd is not None:
            return
        fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        baud_const = getattr(termios, "B%d" % self.baud)
        attrs[4] = baud_const
        attrs[5] = baud_const
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIOFLUSH)
        self._fd = fd

    def close(self):
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None

    def read_holding(self, slave, address, count=1):
        payload = bytes([
            slave,
            0x03,
            (address >> 8) & 0xFF,
            address & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        ])
        response = self._xfer(payload)
        if len(response) < 5 or response[0] != slave or response[1] & 0x80:
            raise ModbusError("вісь %s не відповіла" % slave)
        if crc16(response[:-2]) != (response[-2] | (response[-1] << 8)):
            raise ModbusError("помилка контрольної суми")
        data = response[3:-2]
        if response[2] != len(data) or len(data) != count * 2:
            raise ModbusError("коротка відповідь осі %s" % slave)
        return [(data[index] << 8) | data[index + 1] for index in range(0, len(data), 2)]

    def _xfer(self, payload, timeout=0.2):
        with self._lock:
            return self._xfer_unlocked(payload, timeout)

    def _xfer_unlocked(self, payload, timeout=0.2):
        self.open()
        crc = crc16(payload)
        frame = payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])
        termios.tcflush(self._fd, termios.TCIFLUSH)
        time.sleep(0.008)
        os.write(self._fd, frame)
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready, _, _ = select.select([self._fd], [], [], 0.02)
            if not ready:
                if buf:
                    break
                continue
            chunk = os.read(self._fd, 256)
            if chunk:
                buf += chunk
                deadline = time.time() + 0.025
        return buf
