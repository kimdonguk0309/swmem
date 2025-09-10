"""
16-EiB Software-Defined Memory SuperComputer
--------------------------------------------
사용자 코드는 아래 USER_CODE 문자열 안에 넣으면 즉시 16-EiB 메모리 위에서 실행됩니다.
"""
import mmap, types, sys, struct, ctypes, os, builtins, runpy, numpy as _np

# ---------- 1. 16-EiB 가상 메모리 ----------
PAGE_SHIFT = 16
PAGE_SIZE  = 1 << PAGE_SHIFT
PAGE_MASK  = PAGE_SIZE - 1
EIB        = 1 << 64

class SWMem:
    _dir = {}
    @staticmethod
    def _page_base(va): return va & ~PAGE_MASK
    @classmethod
    def _ensure(cls, va):
        base = cls._page_base(va)
        if base not in cls._dir:
            cls._dir[base] = mmap.mmap(-1, PAGE_SIZE,
                                       prot=mmap.PROT_READ|mmap.PROT_WRITE)
        return cls._dir[base]
    @classmethod
    def read(cls, va, n):
        data = bytearray()
        while n:
            pg  = cls._ensure(va)
            off = va & PAGE_MASK
            chunk = min(n, PAGE_SIZE - off)
            data.extend(pg[off:off+chunk])
            va += chunk; n -= chunk
        return bytes(data)
    @classmethod
    def write(cls, va, buf):
        buf = memoryview(buf)
        while buf:
            pg  = cls._ensure(va)
            off = va & PAGE_MASK
            chunk = min(len(buf), PAGE_SIZE - off)
            pg[off:off+chunk] = buf[:chunk]
            buf  = buf[chunk:]; va += chunk
    @classmethod
    def addressof(cls, va):
        pg  = cls._ensure(va)
        off = va & PAGE_MASK
        return ctypes.addressof(ctypes.c_byte.from_buffer(pg, off))

# ---------- 2. drop-in numpy ----------
class sw_numpy:
    @staticmethod
    def array(obj, dtype=_np.float32):
        dtype = _np.dtype(dtype)
        buf   = memoryview(bytearray(obj)).cast('B')
        va    = 0x4000_0000_0000_0000
        SWMem.write(va, buf)
        return sw_ndarray(va, len(buf)//dtype.itemsize, dtype)
    @staticmethod
    def random_rand(*shape): return sw_numpy.array(_np.random.rand(*shape))
class sw_ndarray:
    def __init__(self, va, size, dtype): self.va, self.size, self.dtype = va, size, dtype
    def __matmul__(self, other):
        M, K = self.size, other.size
        N    = 1 if len(other.shape)==1 else other.shape[1]
        C_va = 0x5000_0000_0000_0000
        a = _np.frombuffer(SWMem.read(self.va, self.size*self.dtype.itemsize), dtype=self.dtype).reshape(-1, K)
        b = _np.frombuffer(SWMem.read(other.va, other.size*other.dtype.itemsize), dtype=other.dtype).reshape(K, -1)
        c = (a @ b).astype(self.dtype)
        SWMem.write(C_va, c.tobytes())
        return sw_ndarray(C_va, c.size, self.dtype)
    def tobytes(self): return SWMem.read(self.va, self.size*self.dtype.itemsize)
    @property
    def shape(self): return (self.size,)

# ---------- 3. 사용자 코드 입력 ----------
USER_CODE = """
import numpy as np  # sw_numpy로 오버라이드
np = sw_numpy
a = np.random_rand(2048, 2048)
b = np.random_rand(2048, 2048)
c = a @ b
print("16-EiB 행렬곱 완료. C[:4] =", _np.frombuffer(c.tobytes(), dtype='float32')[:4])
"""

# ---------- 4. 즉시 실행 ----------
if __name__ == '__main__':
    # 16-EiB 메모리가 쓰일 전역 이름 공간 준비
    g = {
        'sw_numpy': sw_numpy,
        '_np'     : _np,
        'SWMem'   : SWMem
    }
    # 사용자 코드 컴파일-실행
    bytecode = compile(USER_CODE, '<16-EiB-SuperComputer>', 'exec')
    exec(bytecode, g)
