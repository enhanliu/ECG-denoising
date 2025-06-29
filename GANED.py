import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
import model
import data as da
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

train_path = 'work/geyiban_generate_noise/train'
test_path = 'work/geyiban_generate_noise/test'
model_path = 'work/results/weights/train_gan_epoch041.pdparams'
start_epoch = 41

paired_dataset_train = da.PairedData(train_path)
paired_dataset_test = da.PairedData(test_path)
# paired_dataset_train = da.PairedData('train')
# paired_dataset_test = da.PairedData('test')
generator = model.UnetGenerator()
if model_path != '':
    generator.set_dict(paddle.load(model_path))
discriminator = model.NLayerDiscriminator()
# 超参数
LR = 1e-4
BATCH_SIZE = 2
EPOCHS = 50

# 优化器
optimizerG = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=generator.parameters(),
    beta1=0.5,
    beta2=0.999)

optimizerD = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=discriminator.parameters(),
    beta1=0.5,
    beta2=0.999)

# 损失函数
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# dataloader
data_loader_train = DataLoader(
    paired_dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
data_loader_test = DataLoader(
    paired_dataset_test,
    batch_size=BATCH_SIZE
)

results_save_path = 'work/results/fig'
os.makedirs(results_save_path, exist_ok=True)  # 保存每个epoch的测试结果

weights_save_path = 'work/results/weights'
os.makedirs(weights_save_path, exist_ok=True)  # 保存模型
best_epoch = 0
best_DSC = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_iou = 0
for epoch in range(start_epoch, EPOCHS):
    for data in tqdm(data_loader_train):
        real_A, real_B = data  # A为有噪声 B为无噪声
        # print(real_A, real_B)
        optimizerD.clear_grad()
        # D(real)
        real_AB = paddle.concat((real_A, real_B), 1)
        d_real_predict = discriminator(real_AB)
        d_real_loss = bce_loss(d_real_predict, paddle.ones_like(d_real_predict))  # 判别器损失，结果为真

        # D(fake)
        fake_B = generator(real_A).detach()
        fake_AB = paddle.concat((real_A, fake_B), 1)
        d_fake_predict = discriminator(fake_AB)
        # print(d_fake_predict.shape)
        d_fake_loss = bce_loss(d_fake_predict, paddle.zeros_like(d_fake_predict))  # 判别器损失，结果为假
        # print(fake_B.shape, d_fake_predict.shape)

        # train D
        d_loss = (d_real_loss + d_fake_loss) / 2.  # 判别器两对样本损失相加
        d_loss.backward()
        optimizerD.step() #更新判别器

        optimizerG.clear_grad()
        # D(fake)
        fake_B = generator(real_A)
        fake_AB = paddle.concat((real_A, fake_B), 1)
        g_fake_predict = discriminator(fake_AB)
        g_bce_loss = bce_loss(g_fake_predict, paddle.ones_like(g_fake_predict))
        g_l1_loss = l1_loss(fake_B, real_B) * 100.
        g_loss = g_bce_loss + g_l1_loss

        # train G
        g_loss.backward()
        optimizerG.step()
        #跳过训练
        break

    print(f'Epoch [{epoch + 1}/{EPOCHS}] Loss D: {d_loss.numpy()}, Loss G: {g_loss.numpy()}')

    # if (epoch + 1) % 10 == 0:
    paddle.save(generator.state_dict(),
                os.path.join(weights_save_path, train_path.split('/')[-2]+'_gan_epoch' + str(epoch + 1).zfill(3) + '.pdparams'))

    # test
    generator.eval()
    y_scores = []
    y_trues = []
    with paddle.no_grad():
        DSC = 0
        precision = 0
        recall = 0
        f1_value = 0
        iou = 0
        for data in data_loader_test:
            real_A, real_B = data  # real_A是带噪声图像 real_B是干净心电图
            fake_B = generator(real_A)  # 使用生成模型对real_A去噪
            # 绘制roc曲线存储
            pred_score = fake_B
            pred_prob = paddle.nn.functional.sigmoid(pred_score)
            probs = pred_prob.numpy().ravel()
            labels = (real_B.numpy() > 0).astype(np.uint8).ravel()
            y_scores.append(probs)
            y_trues.append(labels)
            # 结束
            ones = paddle.ones_like(fake_B)
            zeros = paddle.zeros_like(fake_B)
            real_B = paddle.where(real_B < 0, ones, zeros)
            fake_B = paddle.where(fake_B < 0, ones, zeros)
            fake_jiao_real = paddle.where(paddle.logical_and(real_B == fake_B, real_B == 1), ones, zeros)
            fake_huo_real = paddle.where(paddle.logical_or(real_B == 1, fake_B == 1), ones, zeros)
            # 计算DSC
            temp = 2 * paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (
                    paddle.sum(real_B, axis=(1, 2, 3)) + paddle.sum(fake_B, axis=(1, 2, 3)) + 0.000000001)
            # print(temp.sum())
            DSC += paddle.sum(temp).item()
            # 计算precision
            precision += paddle.sum(
                paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (paddle.sum(fake_B, axis=(1, 2, 3)) + 0.000000001)).item()
            # 计算recall
            recall += paddle.sum(
                paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (paddle.sum(real_B, axis=(1, 2, 3)) + 0.000000001)).item()
            # 计算IOU
            iou += paddle.sum(
                paddle.sum(fake_jiao_real, axis=(1, 2, 3)) / (
                        paddle.sum(fake_huo_real, axis=(1, 2, 3)) + 0.000000001)).item()

        DSC = DSC / len(paired_dataset_test)
        precision = precision / len(paired_dataset_test)
        recall = recall / len(paired_dataset_test)
        iou = iou / len(paired_dataset_test)
        f1_value = 2 * precision * recall / (precision + recall)
        print(f"DSC: {DSC:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"IoU: {iou:.4f}")
        print(f"F1 Score: {f1_value:.4f}")
        fake_B = generator(real_A)
        # 绘制ROC曲线#################################
        # 合并所有 batch 的数据
        y_scores = np.concatenate(y_scores, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        # 计算 ROC 曲线和 AUC
        fpr, tpr, thresholds = roc_curve(y_trues, y_scores)
        auc_score = roc_auc_score(y_trues, y_scores)
   

    if best_f1 < f1_value:
        best_epoch = epoch
        best_DSC = DSC
        best_precision = precision
        best_recall = recall
        best_f1 = f1_value
        best_iou = iou
    result = paddle.concat([real_A[:3], real_B[:3], fake_B[:3]], 3)
    result = result.detach().numpy().transpose(0, 2, 3, 1)
    result = np.vstack(result)
    result = (result * 127.5 + 127.5).astype(np.uint8)
    cv2.imwrite(os.path.join(results_save_path, train_path.split('/')[-2]+'_gan_epoch' + str(epoch + 1).zfill(3) + '.png'), result)
    generator.train()
    print(f"best_epoch: {best_epoch} *************###############################")
    print(f"best_DSC: {best_DSC:.4f}")
    print(f"best_Precision: {best_precision:.4f}")
    print(f"best_Recall: {best_recall:.4f}")
    print(f"best_IoU: {best_iou:.4f}")
    print(f"best_F1 Score: {best_f1:.4f}")
