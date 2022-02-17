import os
import sys
from glob import glob
from matplotlib import pyplot as plt
import shutil

folder_name = sys.argv[1]

exclude = ["args_iterate", "*.jpg", "figs"]

exclude_ext = [i[i.find(".")+1:] for i in exclude if i.startswith("*.")]
exclude = [i for i in exclude if not i.startswith("*.")]

training_loss_line = "Training running losses:"
test_acc_line = "Test running accuracies:"

def str_to_list(s):
    return [elem.strip() for elem in s.strip().strip('][').split(',')]

def main():
    folder_base = f'outs/{folder_name}'
    figs_folder = os.path.join(folder_base,'figs')
    schedulers_dir = os.path.join(figs_folder, "schedulers")
    shutil.rmtree(figs_folder, ignore_errors=True)
    os.makedirs(schedulers_dir, exist_ok=True)
    folders = glob(f'{folder_base}/*')
    
    test_acc_dic = {}
    training_loss_dic = {}
    
    with open(f'{folder_base}/args_iterate', 'r') as argsh:
        iterate_over = [i.strip().replace("-","_") for i in argsh.readlines()]
    
    print(f'parsing folder {folder_base} for args {iterate_over}')
    c = 0
    crashed = []
    for folder in folders:
        folderbase = os.path.basename(folder)
        if folderbase in exclude or True in [folderbase.endswith(i) for i in exclude_ext]:
            continue
            
        c += 1
            
        training_loss, test_acc = None, None
        training_loss_toggle, test_acc_toggle = False, False
        if not os.path.exists(f'{folder}/log.log'):
            crashed.append(folder)
            continue
            
        with open(f'{folder}/log.log','r') as logh:
            for l in logh.readlines():
                if training_loss_toggle:
                    training_loss = str_to_list(l)
                elif test_acc_toggle:
                    test_acc = str_to_list(l)
                training_loss_toggle = False
                test_acc_toggle = False
                
                if l.startswith(training_loss_line):
                    training_loss_toggle = True
                if l.startswith(test_acc_line):
                    test_acc_toggle = True
        
        assert test_acc != None and training_loss != None
        
        args = {}
        with open(f'{folder}/args','r') as logh:
            for l in logh.readlines():
                key,value = l.strip().split(":")
                args[key] = value
                
        folder_key = tuple([args[key] for key in iterate_over])
        
        test_acc = list(map(float, test_acc))
        training_loss = list(map(float, training_loss))
        epochs = int(args['train_epochs'])
        test_acc_dic[folder_key] = [range(epochs), test_acc]
        training_loss_dic[folder_key] = [range(epochs), training_loss]
        
        shutil.copy(f'{folder}/scheduler.jpg', f'{schedulers_dir}/{"_".join([f"{arg_name}_{arg_value}" for arg_value, arg_name in zip(folder_key, iterate_over)])}.jpg')
    
    print()
    print("-----------------------------------------------")
    print(f'found {c} folders. {len(crashed)} crashed')
    
    if len(crashed) > 0:
        print("crashed folders:")
        for cfolder in crashed:
            print(cfolder)
            
    print("-----------------------------------------------")
    print()
    
    if len(crashed) == c:
        return
    
    for argi, arg in enumerate(iterate_over):
        arg_values = list(set([i[argi] for i in test_acc_dic.keys()]))
        for arg_value in arg_values:
            for key in test_acc_dic:
                if key[argi] == arg_value:
                    plt.plot(test_acc_dic[key][0], test_acc_dic[key][1], label=key)
            
            plt.legend(loc='best')
            plt.xlabel('epochs')
            plt.ylabel('test acc')
            plt.title(f'test acc for {arg}={arg_value}')
            plt.savefig(f'{figs_folder}/test_acc_{arg}={arg_value}.jpg')
            plt.clf()
            
            for key in test_acc_dic:
                if key[argi] == arg_value:
                    plt.plot(training_loss_dic[key][0], training_loss_dic[key][1], label=key)
            
            plt.legend(loc='best')
            plt.xlabel('epochs')
            plt.ylabel('training loss')
            plt.title(f'training loss for {arg}={arg_value}')
            plt.savefig(f'{figs_folder}/training_loss_{arg}={arg_value}.jpg')
            plt.clf()
    
    for key in test_acc_dic:
        plt.plot(test_acc_dic[key][0], test_acc_dic[key][1], label=key)
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('test acc')
    plt.title(f'test acc {iterate_over}')
    plt.savefig(f'{figs_folder}/test_acc.jpg')
    plt.clf()
    
    for key in training_loss_dic:
        plt.plot(training_loss_dic[key][0], training_loss_dic[key][1], label=key)
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.title(f'training loss {iterate_over}')
    plt.savefig(f'{figs_folder}/training_loss.jpg')
    
    sorted_test_acc_keys = sorted(test_acc_dic.keys(), key=lambda x: test_acc_dic[x][1][-1], reverse=True)
    print('test acc results')
    print("\n".join([f'{key}:{test_acc_dic[key][1][-1]}' for key in sorted_test_acc_keys]))
    print()
    sorted_training_loss_dic = sorted(training_loss_dic.keys(), key=lambda x: training_loss_dic[x][1][-1])
    print('training loss results')
    print("\n".join([f'{key}:{training_loss_dic[key][1][-1]}' for key in sorted_training_loss_dic]))
    print()
    
if __name__ == '__main__':
    main()