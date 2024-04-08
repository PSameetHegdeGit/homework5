from homework.planner import Planner, save_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from homework.utils import load_data
import os
from torch.optim import lr_scheduler

import warnings


def distance(a, b):
    return np.linalg.norm(a - b)


def train(args, data_folder='drive_data'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress DeprecationWarning

    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Planner().to(device)
    # if args.continue_training:
    #     model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'AimPoint.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

    #Update the transfrm
    # import inspect
    # transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    train_data = load_data(f'{data_folder}/train_data', num_workers=4, batch_size=32)
    valid_data = load_data(f'{data_folder}/valid_data', num_workers=4, batch_size=32)
    loss = torch.nn.MSELoss(reduction='mean')

    global_step = 0
    best_mean_distance = float('inf')
    early_stopping = 10
    for epoch in range(args.num_epoch):

        ## TRAINING

        model.train()
        train_loss, train_distances = [], []
        for img, true_aim_point in train_data:
            img, true_aim_point = img.to(device), true_aim_point.to(device)

            detected_aim_point = model(img).to(device)

            l = loss(detected_aim_point, true_aim_point)
            train_distance = distance(detected_aim_point.cpu().detach().numpy(), true_aim_point.cpu().detach().numpy())
            train_loss.append(l)
            train_distances.append(train_distance)


            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, true_aim_point, detected_aim_point, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', l, global_step)
                train_logger.add_scalar('distance', train_distance, global_step)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            global_step += 1

        if train_logger is not None:
            train_logger.add_scalar('distance', sum(train_distances)/len(train_distances), epoch)
        scheduler.step()
        ## VALIDATION
        model.eval()
        valid_loss, valid_distances = [], []
        with torch.no_grad():
            for img, true_aim_point in valid_data:
                img, true_aim_point = img.to(device), true_aim_point.to(device)

                detected_aim_point = model(img)

                l = loss(detected_aim_point, true_aim_point)
                valid_distance = distance(detected_aim_point.cpu().detach().numpy(), true_aim_point.cpu().detach().numpy())
                valid_loss.append(l)
                valid_distances.append(valid_distance)

                if valid_logger is not None and global_step % 100 == 0:
                    log(valid_logger, img, true_aim_point, detected_aim_point, global_step)

                if valid_logger is not None:
                    valid_logger.add_scalar('loss', l, global_step)
                    train_logger.add_scalar('distance', valid_distance, global_step)
                global_step += 1

        mean_valid_distance = sum(valid_distances)/len(valid_distances)
        if mean_valid_distance < best_mean_distance:
            print("found better model. Saving...")
            save_model(model)
            early_stopping = 10
            best_mean_distance = mean_valid_distance
        else:
            early_stopping -= 1

        if early_stopping == 0:
            print("hit best model, stopping early")
            exit(0)

        print('epoch %-3d \t training loss %0.3f \t validation loss %0.3f \t validation distance %0.3f' % (epoch, sum(train_loss)/len(train_loss), sum(valid_loss)/len(valid_loss), mean_valid_distance))
        save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-g', '--gamma', type=float, default=0.5, help="stepLR gamma")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-d', '--data_folder', default='drive_data')

    args = parser.parse_args()
    train(args, data_folder=args.data_folder)
