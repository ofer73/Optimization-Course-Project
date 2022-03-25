import glob
from distutils.dir_util import copy_tree
import shutil
import os

outs_folder = "outs"
figs_folder = "figs"

def main():
    shutil.rmtree(figs_folder)
    os.makedirs(figs_folder)
    folders = glob.glob(f'{outs_folder}/*')
    for folder in folders:
        if os.path.exists(f'{folder}/figs'):
            folder_parts = os.path.basename(folder).split("_")
            new_parent, new_folder = "_".join(folder_parts[:3]), "_".join(folder_parts[3:])
            os.makedirs(f'{figs_folder}/{new_parent}',exist_ok=True)
            copy_tree(f'{folder}/figs', f'{figs_folder}/{new_parent}/{new_folder}')

if __name__ == '__main__':
    main()