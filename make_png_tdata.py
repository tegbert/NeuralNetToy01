# make_png_tdata.py
#
# Reads original text data file and generates a PNG image.
#
import sys
import numpy as np
import scipy.misc


def run(data_fname):
    img_ll = []
    with open(data_fname) as fin:
        for rlin in fin:
            if len(rlin) < 30:
                continue
            lin = rlin.strip()
            row_ll = []
            for ch in lin:
                val = 0.0
                if '.' == ch:
                    val = 1.0
                row_ll.append(val)
            img_ll.append(row_ll)
    img_np = np.array(img_ll)
    return img_np


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("USE: make_orig <dataFname> <outputFname>")
        sys.exit()
    g_img_np = run(sys.argv[1])
    scipy.misc.imsave(sys.argv[2], g_img_np)
