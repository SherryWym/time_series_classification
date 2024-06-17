import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.dataloader import get_dataloader
from model.vst_model import VSTModel
import random
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
from utils.write_log import loss_save
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def sigmoid_function(z):
    return 1/(1 + math.exp(-z))


if __name__ == '__main__':
    # -------------------------------- 超参数 -------------------------------- #
    batch_size = 1
    n_worker = 4
    cuda = True
    optimizer_name = 'adam'
    scheduler_name = 'cosine'
    learning_rate = 0.005
    weight_decay = 1e-5
    epochs = 100
    focal_gamma = 2
    classes_num = 2
    frame_number_per_sample = 84
    video_resolution = 320
    criterion = nn.BCEWithLogitsLoss()
    # ------------------------------------------------------------------------ #

    # ------------------------------- 超参数-训练设备 --------------------------------- #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_ids = [0, 1]
    local_rank = 0
    # -------------------------------------------------------------------------- #

    # --------------------------------- SEED ------------------------------------- #
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(777)
    # ---------------------------------------------------------------------------- #

    # ================================================== 开始训练 ==================================================#
    scaler = GradScaler()
    # ------------------------------- 模型定义 --------------------------------- #
    model = VSTModel(num_classes=2)
    model = nn.DataParallel(model, device_ids=device_ids).cuda(local_rank)
    # -------------------------------------------------------------------------- #

    # ------------------------------- 数据集加载 ------------------------------- #
    trainloader, validloader = get_dataloader(batch_size=batch_size)
    # -------------------------------------------------------------------------- #

    # ------------------------------ Optimizer --------------------------------- #
    if optimizer_name == 'adam':
        optimizer = optim.AdamW(lr=learning_rate, params=model.parameters(), weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(lr=learning_rate, params=model.parameters(), momentum=0.937)
    # -------------------------------------------------------------------------- #

    # ------------------------------ Scheduler --------------------------------- #
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=learning_rate * 0.01,
                                                         T_max=epochs / 10)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.9, step_size=1)
    # -------------------------------------------------------------------------- #

    # ------------------------------ Start Training ---------------------------- #
    print()
    print("================= Training Configuration ===================")
    show_config(annotation_file_path=annotation_file_path, video_path=video_path, batch_size=batch_size,
                cuda=cuda, optimizer=optimizer_name, scheduler=scheduler_name, learning_rate=learning_rate,
                weight_decay=weight_decay, epochs=epochs, focal_gamma=focal_gamma,
                classes_num=classes_num, frame_number_per_sample=frame_number_per_sample,
                video_resolution=video_resolution)
    print("=============================================================")
    loss_min = 1000000
    train_loss_array = []
    valid_loss_array = []
    valid_accuracy = 0
    valid_precision = 0
    valid_recall = 0
    valid_f1 = 0
    # valid_partial_accuracy = 0
    best_model = None
    best_model_name = None
    for epoch in range(epochs):
        train_loss = 0
        train_loop = tqdm(enumerate(trainloader), total=len(trainloader))
        model.train()
        for i, (maps, labels) in train_loop:
            inputs = maps.cuda(local_rank)
            gts = labels.cuda(local_rank)
            predictions = model(inputs).cuda(local_rank)
            with autocast():

                loss = criterion(predictions.squeeze(), gts.squeeze().float())
                train_loss += loss.item()

            # ------------------ 清空梯度,反向传播 ----------------- #
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # ----------------------------------------------------- #
            train_loop.set_description(f'Epoch [{epoch}/{epochs}]')
            train_loop.set_postfix(multi_label_focal_loss=loss.item(), learning_rate=optimizer.param_groups[0]['lr'])
        train_loss_array.append(train_loss)

        # ------------------------------- Validation --------------------------------- #
        validation_loop = tqdm(enumerate(validloader), total=len(validloader))

        print()
        print("########################## start validation #############################")

        model.eval()
        with torch.no_grad():
            validation_loss = 0
            valid_predictions = []
            valid_gts = []
            for i, (maps, labels) in validation_loop:
                inputs = maps.cuda(local_rank)
                gts = labels.cuda(local_rank)
                valid_prediction = model(inputs).cuda(local_rank)
                loss = criterion(valid_prediction.squeeze(), gts.squeeze().float())
                validation_loss += loss.item()
                validation_loss_show = validation_loss / (i+1)

                # if valid_prediction.item() > 0.5:
                #     valid_prediction = 1
                # else:
                #     valid_prediction = 0
                valid_predictions.append(sigmoid_function(valid_prediction.item()))
                valid_gts.append(gts.item())

                validation_loop.set_postfix(multi_label_focal_loss_val=validation_loss_show)
            if validation_loss < loss_min:
                best_model = model
                best_model_name = "val_loss_" + str(validation_loss_show) + '.pth'
                print("best model now:", best_model_name)
                torch.save(best_model.state_dict(), "model_path/" + best_model_name)
                loss_min = validation_loss

            # acc = MulticlassAccuracy(num_classes=classes_num).cuda(local_rank)
            # pre = MulticlassPrecision(num_classes=classes_num).cuda(local_rank)
            # rec = MulticlassRecall(num_classes=classes_num).cuda(local_rank)
            # f1s = MulticlassF1Score(num_classes=classes_num).cuda(local_rank)

            valid_accuracy = accuracy_score(valid_gts, valid_predictions)
            valid_precision = precision_score(valid_gts, valid_predictions, average="macro")
            valid_recall = recall_score(valid_gts, valid_predictions, average="macro")
            valid_f1 = f1_score(valid_gts, valid_predictions, average="macro")

            print("gt:", valid_gts)
            print("pred:", valid_predictions)
            print(valid_accuracy)
            print(valid_precision)
            print(valid_recall)
            print(valid_f1)

        valid_loss_array.append(validation_loss)

        print()
        print("########################## end validation #############################")
        # ---------------------------------------------------------------------------- #

        scheduler.step()

    loss_save(train_loss_array, mode='train')
    loss_save(valid_loss_array, mode='valid', model=best_model, model_name=best_model_name)
    print()
    print("============================== end training =================================")


