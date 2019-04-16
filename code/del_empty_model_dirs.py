# deletes models with no weights saved to them.
import shutil
import os
for x in os.listdir('models'):
    subpath = os.path.join('models', x)
    if "encoder.pth" not in os.listdir(subpath):
        print("bad", subpath, os.listdir(subpath))
        shutil.rmtree(subpath)