import collections.abc
import math
import numpy as np

def create_and_modify_array(length: int, s: int, e: int) -> np.ndarray | None:
    if not (0 <= s < length and 0 <= e < length):
        return None
    if s > e:
        return None

    arr = np.ones(length, dtype=int)
    
    if e == 0:
        return arr
    
    arr[s : e + 1] = 2
    
    return arr

def find_intersection(s1, e1, s2, e2):
    intersection_start = max(s1, s2)
    intersection_end = min(e1, e2)
    if intersection_start > intersection_end:
        return None
    else:
        return intersection_start, intersection_end


def get_length(length, hop_length=128):
    return 1 + length // hop_length

class AudioMaskReader(collections.abc.Mapping):
    def __init__(self, path):
        # 处理 path 的mask
        self.data = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                if len(line) == 3:
                    # self.data[line[0]] = [int(line[1]), 0, 0]
                    self.data[line[0]] = [get_length(int(line[1])), 0, 0]
                else:
                    s1 = float(line[3].split(",")[0]) * int(line[2])
                    s2 = float(line[4].split(",")[0]) * int(line[2])
                    dur1 = float(line[3].split(",")[1]) * int(line[2])
                    dur2 = float(line[4].split(",")[1]) * int(line[2])
                    e1 = min(math.ceil(s1 + dur1), int(line[1]))
                    e2 = min(math.ceil(s2 + dur2), int(line[1]))
                    s1 = max(math.floor(s1), 0)
                    s2 = max(math.floor(s2), 0)
                    
                    start, end = find_intersection(s1, e1, s2, e2)
                    # self.data[line[0]] = [int(line[1]), start, end]
                    self.data[line[0]] = [get_length(int(line[1])), get_length(start), get_length(end)]
                    

    def __getitem__(self, key):
        info = self.data[key]
        length = info[0]
        s = info[1]
        e = info[2]
        
        return create_and_modify_array(length, s, e)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
    
    def keys(self):
        return self.data.keys()