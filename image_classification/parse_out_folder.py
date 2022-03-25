import os
import sys
from glob import glob
from matplotlib import pyplot as plt
import shutil
from collections import defaultdict
import csv

convs = [80,90]

folder_name = sys.argv[1]

exclude = ["args_iterate", "*.jpg", "figs"]

exclude_ext = [i[i.find(".")+1:] for i in exclude if i.startswith("*.")]
exclude = [i for i in exclude if not i.startswith("*.")]

training_loss_line = "Training running losses:"
test_acc_line = "Test running accuracies:"
style = '--' #dashed

def average_lists(lists):
    lists = [l for l in lists if l]
    return [sum(i)/len(i) for i in zip(*lists)]

def str_to_list(s):
    return [elem.strip() for elem in s.strip().strip('][').split(',')]

def main():
    folder_base = f'outs/{folder_name}'
    figs_folder = os.path.join(folder_base,'figs')
    schedulers_dir = os.path.join(figs_folder, "schedulers")
    shutil.rmtree(figs_folder, ignore_errors=True)
    os.makedirs(schedulers_dir, exist_ok=True)
    folders = glob(f'{folder_base}/*')

    with open(f'{folder_base}/args_iterate', 'r') as argsh:
        iterate_over = [i.strip().replace("-","_") for i in argsh.readlines()]
        for i, io in enumerate(iterate_over):
            if io == 'alpha':
                pass
                #iterate_over[i] == 'alpha_name'
        
    if iterate_over[0].startswith("TIMES="):
        times = iterate_over[0][len("TIMES="):]
        times = int(times)
        iterate_over = iterate_over[1:]
    else:
        times = 1
        
    test_acc_dic = defaultdict(lambda: {'epochs':0, 'test_acc':[False]*times})
    training_loss_dic = defaultdict(lambda: {'epochs':0, 'training_loss':[False]*times})
    schedulers_fig = []
    
    print(f'parsing folder {folder_base} for args {iterate_over}')
    c = 0
    crashed = []
    crashed_keys = []
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
        
        folder_times_id = 0
        if times > 1:
            folder_times_id = int(folderbase.split("_")[-1])
            
        if training_loss[-1] == "nan" or all(map(lambda x: x=='0.1', test_acc)) or all(map(lambda x: x=='0.1', training_loss)):
            crashed.append(folder)
            crashed_keys.append([folder_key,folder_times_id])
            continue
            
        test_acc = list(map(float, test_acc))
        training_loss = list(map(float, training_loss))
        epochs = int(args['train_epochs'])
        test_acc_dic[folder_key]['test_acc'][folder_times_id] = test_acc
        if test_acc_dic[folder_key]['epochs'] > 0:
            assert test_acc_dic[folder_key]['epochs'] == epochs
        else:
            test_acc_dic[folder_key]['epochs'] = epochs
            
        training_loss_dic[folder_key]['training_loss'][folder_times_id] = training_loss
        if training_loss_dic[folder_key]['epochs'] > 0:
            assert training_loss_dic[folder_key]['epochs'] == epochs
        else:
            training_loss_dic[folder_key]['epochs'] = epochs
        
        if folder_key not in schedulers_fig:
            schedulers_fig.append(folder_key)
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
    
    for key in test_acc_dic:
        test_acc_dic[key]['test_acc'] = average_lists(test_acc_dic[key]['test_acc'])
        
    for key in training_loss_dic:
        training_loss_dic[key]['training_loss'] = average_lists(training_loss_dic[key]['training_loss'])
    
    for argi, arg in enumerate(iterate_over):
        arg_values = list(set([i[argi] for i in test_acc_dic.keys()]))
        for arg_value in arg_values:
            for key in test_acc_dic:
                if key[argi] == arg_value:
                    plt.plot(range(test_acc_dic[key]['epochs']), test_acc_dic[key]['test_acc'], style, label=key)
            
            plt.legend(loc='best')
            plt.xlabel('epochs')
            plt.ylabel('test acc')
            plt.title(f'test acc for {arg}={arg_value}')
            plt.savefig(f'{figs_folder}/test_acc_{arg}={arg_value}.jpg')
            plt.clf()
            
            for key in test_acc_dic:
                if key[argi] == arg_value:
                    plt.plot(range(training_loss_dic[key]['epochs']), training_loss_dic[key]['training_loss'], style, label=key)
            
            plt.legend(loc='best')
            plt.xlabel('epochs')
            plt.ylabel('training loss')
            plt.title(f'training loss for {arg}={arg_value}')
            plt.savefig(f'{figs_folder}/training_loss_{arg}={arg_value}.jpg')
            plt.clf()
    
    for key in test_acc_dic:
        plt.plot(range(test_acc_dic[key]['epochs']), test_acc_dic[key]['test_acc'], style, label=key)
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('test acc')
    plt.title(f'test acc {iterate_over}')
    plt.savefig(f'{figs_folder}/test_acc.jpg')
    plt.clf()
    
    for key in training_loss_dic:
        plt.plot(range(training_loss_dic[key]['epochs']), training_loss_dic[key]['training_loss'], style, label=key)
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.title(f'training loss {iterate_over}')
    plt.savefig(f'{figs_folder}/training_loss.jpg')
    
    sorted_test_acc_keys = sorted(test_acc_dic.keys(), key=lambda x: test_acc_dic[x]['test_acc'][-1], reverse=True)
    print('test acc results')
    print("\n".join([f'{key}:{test_acc_dic[key]["test_acc"][-1]}' for key in sorted_test_acc_keys]))
    print()
    sorted_training_loss_dic = sorted(training_loss_dic.keys(), key=lambda x: training_loss_dic[x]['training_loss'][-1])
    print('training loss results')
    print("\n".join([f'{key}:{training_loss_dic[key]["training_loss"][-1]}' for key in sorted_training_loss_dic]))
    print()
    
    print('crashed keys', crashed_keys)
    
    csvfile = open(f'{figs_folder}/{folder_name}.csv', 'w', newline='')
    fieldnames = iterate_over+['test acc']+[f'conv %{conv}' for conv in convs]+['training loss']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    info = {}
    for key in sorted_test_acc_keys:
        info['test acc'] = test_acc_dic[key]["test_acc"][-1]
        info['training loss'] = training_loss_dic[key]['training_loss'][-1]
        for key_arg, arg in zip(key,iterate_over):
            info[arg] = key_arg
        
        for conv in convs:
            info[f'conv %{conv}'] = [i >= info['test acc']*conv/100. for i in test_acc_dic[key]["test_acc"]].index(True)
        writer.writerow(info)
    
    csvfile.close()
    
if __name__ == '__main__':
    main()