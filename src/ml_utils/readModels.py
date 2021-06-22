#
# MUSIC𝄞NTWRK
#
# A python library for pitch class set and rhythmic sequences classification and manipulation,
# the generation of networks in generalized music and sound spaces, and the sonification of arbitrary data
#
# Copyright (C) 2018 Marco Buongiorno Nardelli
# http://www.materialssoundmusic.com, mbn@unt.edu
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import os, glob, tarfile
from .modelLoad import modelLoad

def readModels(path,filename):

    def extract_files(members):
        for tarinfo in members:
            if os.path.splitext(tarinfo.name)[1] == ".h5": 
                yield tarinfo
            elif os.path.splitext(tarinfo.name)[1] == ".normal":
                yield tarinfo
            elif os.path.splitext(tarinfo.name)[1] == ".scaler":
                yield tarinfo
            elif os.path.splitext(tarinfo.name)[1] == ".dict":
                yield tarinfo

    # extract data from tar files
    tar_files = list(glob.glob(os.path.join(path,filename)))
    for file in tar_files:
        tar = tarfile.open(file)
        member=extract_files(tar)
        tar.extractall(members=member)
        tar.close()

    # load model parameters, scaler and normalizer for each model
    modelfiles = list(glob.glob(os.path.join(path,'*.h5')))
    ynew = []
    models = {}
    scalers = {}
    normals = {}
    trdicts = {}
    n = 0 
    for file in modelfiles:
        try:
            models[str(n)],scalers[str(n)],normals[str(n)],trdicts[str(n)] = modelLoad(str(file[+2:-3]))
        except:
            models[str(n)],scalers[str(n)],normals[str(n)] = modelLoad(str(file[+2:-3]))
            trdicts[str(n)] = None
        n += 1
        os.system('rm '+str(file[+2:-3])+'.h5 '+str(file[+2:-3])+'.scaler '+str(file[+2:-3])+'.normal '+str(file[+2:-3])+'.train.dict')
    return(models,scalers,normals,trdicts,modelfiles)

