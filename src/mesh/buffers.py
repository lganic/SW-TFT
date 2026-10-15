
import struct
from collections import deque

DATA_TYPES = {
    "char":      ("b", 1),
    "uchar":     ("B", 1),
    "short":     ("h", 2),
    "ushort":    ("H", 2),
    "int":       ("i", 4),
    "uint":      ("I", 4),
    "long":      ("l", 4),
    "ulong":     ("L", 4),
    "long_long": ("q", 8),
    "ulong_long":("Q", 8),
    "float":     ("f", 4),
    "double":    ("d", 8),
}

class ReadableBuffer:

    def __init__(self, buffer: bytes, endian: str = "little"):
        self.queue = deque(buffer)
        self.endian = self._get_endian(endian)

    @staticmethod
    def _get_endian(endian: str) -> str:
        if endian == "little":
            return "<"
        if endian == "big":
            return ">"
        raise ValueError("Endian must be 'little' or 'big'")

    def __len__(self):
        return len(self.queue)

    def get(self, num_bytes: int) -> bytes:
        if num_bytes < 0:
            raise ValueError("num_bytes must be nonnegative")
        if len(self.queue) < num_bytes:
            raise EOFError("Not enough bytes remaining")

        return bytes(self.queue.popleft() for _ in range(num_bytes))

    def _parse(self, datatype: str):
        fmt, size = DATA_TYPES[datatype]
        return struct.unpack(
            self.endian + fmt,
            self.get(size)
        )[0]

    def parse_char(self):
        
        return self._parse("char")

    def parse_uchar(self):
        
        return self._parse("uchar")

    def parse_short(self):
        
        return self._parse("ushort")

    def parse_ushort(self):
        
        return self._parse("ushort")

    def parse_int(self):
        
        return self._parse("uint")

    def parse_uint(self):
        
        return self._parse("uint")

    def parse_long(self):
        
        return self._parse("long")

    def parse_ulong(self):
        
        return self._parse("ulong")

    def parse_long_long(self):
        
        return self._parse("long_long")

    def parse_ulong_long(self):
        
        return self._parse("ulong_long")

    def parse_float(self):
        
        return self._parse("float")

    def parse_double(self):
        
        return self._parse("double")

    # Explicit signed alternatives
    def parse_signed_char(self):

        return self._parse("char")

    def parse_signed_short(self): 

        return self._parse("short")

    def parse_signed_int(self):   

        return self._parse("int")

    def parse_signed_long(self):  

        return self._parse("long")

    def parse_byte(self):
        return self.parse_uchar()

    def parse_string(self, length: int):

        return ''.join([self.parse_char() for _ in range(length)])


class WritableBuffer:

    def __init__(self, endian: str = "little"):
        self.buffer = bytearray()
        self.endian = ParseableBuffer._get_endian(endian)

    def __len__(self):
        return len(self.buffer)

    def get(self) -> bytes:
        return bytes(self.buffer)

    def write(self, data: bytes):
        self.buffer.extend(data)

    def _write(self, datatype: str, value):
        fmt, _ = DATA_TYPES[datatype]
        self.buffer.extend(
            struct.pack(self.endian + fmt, value)
        )

    def write_char(self, value):
        
        self._write("char", value)

    def write_uchar(self, value):
        
        self._write("uchar", value)

    def write_short(self, value):
        
        self._write("ushort", value)

    def write_ushort(self, value):
        
        self._write("ushort", value)

    def write_int(self, value):
        
        self._write("uint", value)

    def write_uint(self, value):
        
        self._write("uint", value)

    def write_long(self, value):
        
        self._write("long", value)

    def write_ulong(self, value):
        
        self._write("ulong", value)

    def write_long_long(self, value):
        
        self._write("long_long", value)

    def write_ulong_long(self, value):
        
        self._write("ulong_long", value)

    def write_float(self, value):
        
        self._write("float", value)

    def write_double(self, value):
        
        self._write("double", value)

    # Explicit signed alternatives
    def write_signed_char(self, value):
        
        self._write("char", value)

    def write_signed_short(self, value):
        
        self._write("short", value)

    def write_signed_int(self, value):
        
        self._write("int", value)

    def write_signed_long(self, value):
        
        self._write("long", value)

    def write_byte(self, value):
        
        self.write_uchar(value)

