import sys

import matplotlib.pyplot as plt
import numpy

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    import numpy as np
    import os
    import random

    from load_args import load_args
    from data_loader import data_loader
    from mnist_cnn import MNISTConvNet
    from cifar10_resnet import resnet20
    from cifar100_densenet import densenet
    from train import train
    from evaluate import evaluate


    def main():
        args = load_args()
        vargs = vars(args)
        if not os.path.exists(args.log_folder):
              os.makedirs(args.log_folder)
        with open(f'{args.log_folder}/args', 'w+') as f:
            for arg in vargs:
                f.write(f"{arg}:{vargs[arg]}\n")
                
        # Check the availability of GPU.
        use_cuda = args.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Set the random seed for reproducibility.
        if args.reproducible:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            if device != torch.device("cpu"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # Load data, note we will also call the validation set as the test set.
        print('Loading data...')
        dataset = data_loader(dataset_name=args.dataset,
                              dataroot=args.dataroot,
                              batch_size=args.batchsize,
                              val_ratio=(args.val_ratio if args.validation else 0))
        train_loader = dataset[0]
        if args.validation:
            test_loader = dataset[1]
        else:
            test_loader = dataset[2]

        # Define the model and the loss function.
        if args.dataset == 'CIFAR10':
            net = resnet20()
        elif args.dataset == 'CIFAR100':
            net = densenet(depth=100, growthRate=12, num_classes=100)
        elif args.dataset in ['MNIST', 'FashionMNIST']:
            net = MNISTConvNet()
        else:
            raise ValueError("Unsupported dataset {0}.".format(args.dataset))
        net.to(device)
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate the model.
        print(f"Training for train_loader len {len(train_loader)}")
        running_stats = train(args, train_loader, test_loader, net,
                              criterion, device)
        all_train_losses, all_train_accuracies = running_stats[:2]
        all_test_losses, all_test_accuracies, all_learning_rates = running_stats[2:]

        print("Evaluating...")
        final_train_loss, final_train_accuracy = evaluate(train_loader, net,
                                                          criterion, device)
        final_test_loss, final_test_accuracy = evaluate(test_loader, net,
                                                        criterion, device)

        # Logging results.
        print(f'Writing the results to folder {args.log_folder}')
        log_name = 'log.log'
        '''if not os.path.exists(args.log_folder):
            os.makedirs(args.log_folder)
        log_name = (('%s_%s_' % (args.dataset, args.optim_method))
                    + ('Eta0_%g_' % (args.eta0))
                    + ('WD_%g_' % (args.weight_decay))
                    + (('Mom_%g_' % (args.momentum))
                       if args.optim_method.startswith('SGD') else '')
                    + (('alpha_%g_' % (args.alpha))
                       if args.optim_method not in ['Adam', 'SGD'] else '')
                    + (('Milestones_%s_' % ('_'.join(args.milestones)))
                       if args.optim_method == 'SGD_Stage_Decay' else '')
                    + (('Warmup_Steps_%s_' % args.warmup_steps)
                       if '+w' in args.optim_method.lower() else '')
                    + (('Tail_Steps_%s_' % args.tail_steps)
                       if 'tail' in args.optim_method.lower() else '')
                    + (('c_%g_' % (args.c))
                       if args.optim_method.startswith('SLS') else '')
                    + (('Patience_%d_Thres_%g_' % (args.patience, args.threshold))
                       if args.optim_method == 'SGD_ReduceLROnPlateau' else '')
                    + ('Epoch_%d_Batch_%d_' % (args.train_epochs, args.batchsize))
                    + ('%s' % ('Validation' if args.validation else 'Test'))
                    + ('Seed_%s_' % (args.seed if args.reproducible else ''))
                    + '.txt')'''
        mode = 'w' if args.validation else 'a'
            
        with open(args.log_folder + '/' + log_name, mode) as f:
        
            f.write('Training running losses:\n')
            f.write('{0}\n'.format(all_train_losses))
            f.write('Training running accuracies:\n')
            f.write('{0}\n'.format(all_train_accuracies))
            f.write('Final training loss is %g\n' % final_train_loss)
            f.write('Final training accuracy is %g\n' % final_train_accuracy)

            f.write('Test running losses:\n')
            f.write('{0}\n'.format(all_test_losses))
            f.write('Test running accuracies:\n')
            f.write('{0}\n'.format(all_test_accuracies))
            f.write('Final test loss is %g\n' % final_test_loss)
            f.write('Final test accuracy is %g\n' % final_test_accuracy)

        if args.plot_lr:
            # plots learning rate value vs time
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            '''opt_params = f'{args.dataset}_Scheme={args.optim_method}_eta0={args.eta0}_alpha={args.alpha}_milestones={args.milestones}' \
                         f'T_max={args.train_epochs * len(train_loader)}_warmup_steps={args.warmup_steps}' \
                         f'_tail_steps={args.tail_steps}'
            
            folder_path = '/'.join([args.log_folder, "learning_rate_plots"])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            plt.savefig('/'.join([folder_path, f'{opt_params}.jpg']))'''
            
            plt.plot(numpy.arange(0, len(all_learning_rates)), all_learning_rates)
            plt.savefig(os.path.join(args.log_folder, 'scheduler.jpg'))

        print('Finished.')


    main()
