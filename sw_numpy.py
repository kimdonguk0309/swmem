import numpy as _np
from .swmem import SWMem

class sw_numpy:
    """drop-in replacement for numpy"""
    @staticmethod
    def array(obj, dtype=None):
        dtype = _np.dtype(dtype or _np.float32)
        buf   = memoryview(bytearray(obj)).cast('B')
        va    = 0x4000_0000_0000_0000          # 4 EiB 경계부터 할당
        SWMem.write(va, buf)
        return sw_ndarray(va, len(buf)//dtype.itemsize, dtype)

class sw_ndarray:
    def __init__(self, va, size, dtype):
        self.va   = va
        self.size = size
        self.dtype= dtype
        self.shape = (size,)
    def __matmul__(self, other):
        from . import sw_numpy as np
        M, K = self.shape[0], self.shape[1] if len(self.shape)>1 else 1
        N    = other.shape[1] if len(other.shape)>1 else other.size
        C_va = 0x5000_0000_0000_0000
        # GEMM 호출 (아래는 간단히 넘파이로 대체, 실제는 BLAS 연결 가능)
        a = _np.frombuffer(SWMem.read(self.va, self.size*self.dtype.itemsize),
                           dtype=self.dtype).reshape(M,K)
        b = _np.frombuffer(SWMem.read(other.va, other.size*other.dtype.itemsize),
                           dtype=other.dtype).reshape(K,N)
        c = (a @ b).astype(self.dtype)
        SWMem.write(C_va, c.tobytes())
        return sw_ndarray(C_va, c.size, self.dtype)
    def tobytes(self):
        return SWMem.read(self.va, self.size*self.dtype.itemsize)
