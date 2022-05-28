import os
import sys
import imageio
import imageio.v3 as iio
from pathlib import Path

path = os.path.abspath('.') + '/plots/'
print()
images = list()
for file in Path(path).iterdir():
    if not file.is_file():
        continue

    images.append(iio.imread(file))

imageio.mimsave(os.path.abspath('.') + '/' + sys.argv[1] , images)