import mmap, os, struct, ctypes, sys, builtins

PAGE_SHIFT = 16        # 64 KiB
PAGE_SIZE  = 1 << PAGE_SHIFT
PAGE_MASK  = PAGE_SIZE - 1
EIB        = 1 << 64   # 16 EiB

class SWMem:
    _dir = {}                     # 가상 페이지 → mmap 객체
    _backend = mmap.mmap(-1, PAGE_SIZE, prot=mmap.PROT_READ|mmap.PROT_WRITE)

    @staticmethod
    def _page_base(va):
        return va & ~(PAGE_MASK)

    @classmethod
    def _ensure_page(cls, va):
        base = cls._page_base(va)
        if base not in cls._dir:
            cls._dir[base] = mmap.mmap(-1, PAGE_SIZE,
                                       prot=mmap.PROT_READ|mmap.PROT_WRITE)
        return cls._dir[base]

    @classmethod
    def read(cls, va, size):
        data = bytearray()
        while size:
            pg  = cls._ensure_page(va)
            off = va & PAGE_MASK
            chunk = min(size, PAGE_SIZE - off)
            data.extend(pg[off:off+chunk])
            va   += chunk
            size -= chunk
        return bytes(data)

    @classmethod
    def write(cls, va, buf):
        buf = memoryview(buf)
        while buf:
            pg  = cls._ensure_page(va)
            off = va & PAGE_MASK
            chunk = min(len(buf), PAGE_SIZE - off)
            pg[off:off+chunk] = buf[:chunk]
            buf  = buf[chunk:]
            va  += chunk

    @classmethod
    def addressof(cls, va):
        """ctypes pointer 반환 (zero-copy)"""
        pg  = cls._ensure_page(va)
        off = va & PAGE_MASK
        return ctypes.addressof(ctypes.c_byte.from_buffer(pg, off))

def sw_exec(user_script_path):
    """기존 파이썬 파일을 16 EiB 메모리 위에서 실행"""
    import runpy
    orig_open = builtins.open
    def sw_open(file, *a, **kw):
        if file == user_script_path:
            # 실행 도중 모든 파일 read/write 도 16 EiB 공간으로 우회
            return orig_open(file, *a, **kw)
        return orig_open(file, *a, **kw)
    builtins.open = sw_open
    runpy.run_path(user_script_path, run_name='__main__')
