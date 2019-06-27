import shutil
import re
import os

SOURCE_PATH = '../sourcegeo/future'
filepat = re.compile(r'(?P<model>[a-z][a-z])(?P<ind>\d\d)(?P<var>[a-z]{1,3})(?P<year>\d\d)')


for dir_, dirnames, filenames in os.walk(SOURCE_PATH, followlinks=True):
    for file in filenames:
        fname = os.path.basename(file)
        if filepat.match(fname):
            dct = filepat.match(fname).groupdict()
            dst = os.path.join(SOURCE_PATH, dct['model'], dct['ind'],
                         dct['year'], dct['var'])
            os.makedirs(dst, exist_ok=True)
            shutil.copyfile(os.path.join(dir_, fname), os.path.join(dst, fname),
                            follow_symlinks=True)

