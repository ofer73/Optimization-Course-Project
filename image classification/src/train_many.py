import os,sys
for eta0 in [0.05, 0.07, 0.09]:
    for tail_epochs in [20, 30, 40, 50]:
        for batchsize in [128, 64, 32, 12]:
            args = ' '.join(sys.argv[1:])
            os.system(f"python main.py {args} --tail_epochs {tail_epochs} --eta0 {eta0} --batchsize {batchsize}")
