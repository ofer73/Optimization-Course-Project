import glob
import shutil
import sys
import os

exclude = ["args_iterate", "*.jpg", "figs"]
exclude_ext = [i[i.find(".")+1:] for i in exclude if i.startswith("*.")]
exclude = [i for i in exclude if not i.startswith("*.")]

def filter_exclude(folder):
    folderbase = os.path.basename(folder)
    return folderbase in exclude or True in [folderbase.endswith(i) for i in exclude_ext]

def main():
    assert len(sys.argv) == 3
    
    src,dst = sys.argv[1:3]
    
    src_folders = glob.glob(f'{src}/*')
    src_folders = [f for f in src_folders if not filter_exclude(f)]
    
    for folder in src_folders:
        shutil.rmtree(f'{dst}/{os.path.basename(folder)}', ignore_errors=True)
        shutil.move(folder, dst)
        
    shutil.rmtree(src)
    
if __name__ == '__main__':
    main()