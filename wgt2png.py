# wgt2png.py
#
# Reads model and weight data to create a single PNG image file.
#
import sys
import numpy as np
import scipy.misc
from keras.models import model_from_json
from make_png_sequence import exercise_ann


def run_run(model_fname, wgt_fpname, out_fpname):
    with open(model_fname) as fin:
        mjson = fin.read()
        model = model_from_json(mjson)
    wgts = np.load(wgt_fpname)
    model.set_weights(wgts)
    img_np = exercise_ann(model, xdim=50, ydim=50)
    scipy.misc.imsave(out_fpname, img_np)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("USE: implement <modelFname> <wgtFpname> <outFpname>")
        sys.exit()
    g_model_fname = sys.argv[1]
    g_wgt_fpname = sys.argv[2]
    g_out_fpname = sys.argv[3]
    run_run(g_model_fname, g_wgt_fpname, g_out_fpname)
