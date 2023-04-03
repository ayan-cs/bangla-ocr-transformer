import os, mlconfig
from pathlib import Path
from utils import load_function
import numpy as np

parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]
data = os.path.join(parent, 'raw_data', 'Bongabdo')

all_imgs = [i.rsplit('.', maxsplit=1)[0] for i in os.listdir(os.path.join(data, 'Images'))]
all_gt = [i.rsplit('.', maxsplit=1)[0] for i in os.listdir(os.path.join(data, 'Annotations'))]

print(len(all_imgs)==len(all_gt))
print(all_imgs == all_gt)

print(np.random.permutation(10))