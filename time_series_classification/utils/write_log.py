import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



def loss_save(loss_array, mode='train', model=None, model_name=None):
    now = datetime.datetime.now()

    folder = os.path.exists("./logs/"+f"loss{now.year}{now.month}{now.day}{now.hour}{now.minute}")
    if not folder:
        os.makedirs("./logs/" + f"loss{now.year}{now.month}{now.day}{now.hour}{now.minute}")

    save_dir = "./logs/"+f"loss{now.year}{now.month}{now.day}{now.hour}{now.minute}"
    if mode == 'train':
        save_path = os.path.join(save_dir, f"train_loss{now.year}{now.month}{now.day}{now.hour}{now.minute}.txt")
        plot_array = None
        with open(save_path, encoding='utf8', mode='w') as f:
            f.write(''.join(str(i)+"\n" for i in loss_array))
        with open(save_path, encoding='utf8', mode='r') as f:
            plot_array = [float(item.strip()) for item in f.readlines()]
        print(plot_array)
        plt.plot(range(len(plot_array)), plot_array)
        plt.xlabel("epochs")
        plt.ylabel("multi-label focal loss")
        plt.title("Loss in Training")
        plt.savefig(os.path.join(save_dir, f"train_loss{now.year}{now.month}{now.day}{now.hour}{now.minute}.png"))

    if mode == 'valid':
        save_path = os.path.join(save_dir, f"valid_loss{now.year}{now.month}{now.day}{now.hour}{now.minute}.txt")
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))
        plot_array = None
        with open(save_path, encoding='utf8', mode='w') as f:
            f.write(''.join(str(i)+"\n" for i in loss_array))
        with open(save_path, encoding='utf8', mode='r') as f:
            plot_array = [float(item.strip()) for item in f.readlines()]
        plt.plot(range(len(plot_array)), plot_array)
        plt.xlabel("epochs")
        plt.ylabel("multi-label focal loss")
        plt.title("Loss in Validation")
        plt.savefig(os.path.join(save_dir, f"valid_loss{now.year}{now.month}{now.day}{now.hour}{now.minute}.png"))

    plt.close('all')  # 关闭所有 figure windows
