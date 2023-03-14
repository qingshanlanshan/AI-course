import numpy as np
from typing import Tuple

def sin_sample_gen()->Tuple[float,float]:
    x = np.random.uniform(-np.pi, np.pi)
    y = np.sin(x)
    y += np.random.normal(-0.01, 0.01)
    return (x, y)    

def sin_sample_gen_array(size:int,sort:bool=True)->Tuple[np.ndarray,np.ndarray]:
    x = np.random.uniform(-np.pi, np.pi, size)
    if sort:
        x.sort()
    y = np.sin(x)
    y += np.random.normal(-0.01, 0.01, size)
    return (x, y)

if __name__=="__main__":
    file=open("sample.txt","w")
    for i in range(1000):
        x,y = sin_sample_gen()
        file.write("{} {}\n".format(x,y))