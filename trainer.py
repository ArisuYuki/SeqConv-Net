from utils import eval_net_point
from tqdm import tqdm
import torch
from torch import nn
from utils import (
    MaskedSoftmaxCELoss,
    compute_acc,
    statistics,
    eval_fps,
    compute_mean,
)
import numpy as np


def grad_clipping(net, theta):
    """梯度裁剪"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train(
    net,
    train_loader,
    test_loader,
    optimizer,
    epoch,
    num_len,
    classes_weights,
    device,
    area_size,
):
    num_classes = train_loader.dataset.num_classes
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epoch, eta_min=0
    )
    classes_weights = classes_weights.to(device)
    l = MaskedSoftmaxCELoss(classes_weights)
    print("Running on:", device)
    statistics(net, area_size, device=device)
    eval_fps(net, area_size, device=device)
    net = net.to(device)
    best_iou = 0
    for i in range(epoch):
        net.train()
        l_statistic = []
        acc = []
        recall = []
        iou = []
        print(f"-----------------Epoch:{i+1}-----------------")
        with tqdm(
            enumerate(train_loader), desc="Training", leave=False, total=num_len
        ) as pbar_loss:

            for j, (
                valid_input_len,
                input_mat,
                label_mat,
                force_teach_mat,
            ) in pbar_loss:

                input_mat = input_mat.to(device)
                label_mat = label_mat.to(device)
                force_teach_mat = force_teach_mat.to(device)
                valid_input_len = valid_input_len.to(device).to(torch.float32)

                optimizer.zero_grad()
                output = net(input_mat, force_teach_mat)
                loss = l(output, label_mat, valid_input_len)
                l_statistic.append(loss.cpu().item())

                loss.backward()
                grad_clipping(net, 1)
                optimizer.step()

                current_acc, current_recall, current_iou = compute_acc(
                    output.argmax(dim=1), label_mat, num_classes, valid_input_len
                )
                acc.append(current_acc)
                recall.append(current_recall)
                iou.append(current_iou)
                pbar_loss.set_description(
                    f"Loss:{np.mean(l_statistic):.5f} , mIOU:{compute_mean(iou)[-1]:.5f}"
                )

            scheduler.step()

            acc, mACC = compute_mean(acc)
            recall, mRecall = compute_mean(recall)
            iou, mIoU = compute_mean(iou)

            print(f"epoch:{i + 1}, loss:{np.mean(l_statistic):.5f}, mIOU:{mIoU:.5f}")
            print(
                "Class:        ",
                *[f"{i+1:>5d}" for i in range(num_classes)],
                "  AVG",
                sep=" | ",
            )
            print(
                "Train acc:    ",
                *[f"{i:.3f}" for i in acc],
                f"{mACC:.3f}",
                sep=" | ",
            )
            print(
                "Train recall: ",
                *[f"{i:.3f}" for i in recall],
                f"{mRecall:.3f}",
                sep=" | ",
            )
            print(
                "Train iou:    ",
                *[f"{i:.3f}" for i in iou],
                f"{mIoU:.3f}",
                sep=" | ",
            )

            # 每三轮评估一次
            if i % 3 == 0:
                _, _, iou = eval_net_point(net, test_loader, device=device)
                if iou > best_iou:
                    best_iou = iou
                    torch.save(net.state_dict(), "best_model_train.pth")
