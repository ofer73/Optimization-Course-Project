import glob
import os

def main():
    files = glob.glob("*")
    for f in files:
        if f.endswith(".slurm") and not f.startswith("main."):
            os.remove(f)
    
if __name__ == '__main__':
    main()