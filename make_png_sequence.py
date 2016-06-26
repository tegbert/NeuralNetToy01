# make_png_sequence.py
#
# Reads model and weight data to create a PNG image sequence.
#
import sys
import os
import math
import numpy as np
import scipy.misc
from keras.models import model_from_json


def exercise_ann(model, xdim=200, ydim=200):
    xincr = 2.0 / float(xdim - 1)
    yincr = 2.0 / float(ydim - 1)
    x = -1.0
    img_ll = []
    for ix in range(xdim):
        pix_ll = []
        y = -1.0
        for iy in range(ydim):
            inp = np.array([(x, y)])
            outp = model.predict(inp, batch_size=1)
            opix = outp[0][0]
            pix_ll.append(1.0 - opix)
            y += yincr
        x += xincr
        img_ll.append(np.array(pix_ll))
    return np.transpose(np.array(img_ll, dtype=np.float32))


def run_run(model_fname, wgt_dir, movie_dir):
    with open(model_fname) as fin:
        mjson = fin.read()
        model = model_from_json(mjson)
    # iterate the epochs
    ix = 0
    iy = 0
    skip = 0
    while True:
        ix += 1

        # --- BEGIN accelerating training movie
        if skip < int(math.sqrt(float(ix)/10.0)):
            skip += 1
            continue
        else:
            skip = 0
            iy += 1
        # --- END accelerating training movie

        wfname = "%05d.npy" % ix
        wfpname = os.path.join(wgt_dir, wfname)
        if not os.path.exists(wfpname):
            break
        print("Generating '%s' as %d..." % (wfname, iy))
        ofpname = os.path.join(movie_dir, "%05d.png" % iy)
        wgts = np.load(wfpname)
        model.set_weights(wgts)
        img_np = exercise_ann(model, xdim=200, ydim=200)
        scipy.misc.imsave(ofpname, img_np)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("USE: make_png_sequence <modelFname> <wgtDir> <movieDir>")
        sys.exit()
    g_model_fname = sys.argv[1]
    g_wgt_dir = sys.argv[2]
    g_movie_dir = sys.argv[3]
    if not os.path.exists(g_movie_dir):
        os.mkdir(g_movie_dir, 0o755)
    run_run(g_model_fname, g_wgt_dir, g_movie_dir)
