import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from swmem import sw_exec
if __name__ == '__main__' and len(sys.argv) > 1:
    sw_exec(sys.argv[1])
