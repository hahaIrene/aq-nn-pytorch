from src.dataset import AqDataset
from src.model import MLP
from src.graph import generateGraph
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import R2Score
# DataLoader 用於與DATASET連結 從getitem抓東西出來訓練

train_dataset = AqDataset(r"./data/training.csv")
val_dataset = AqDataset(r"./data/validation.csv")


## 設定超參數
# 所有資料進入類神經網路一次，稱為一個epoch
EPOCH = 100
# 每次拿多少筆資料更新類神經網路
BATCH_SIZE = 1024
# 每個EPOCH更新參數的次數
STEP_PER_EPOCH = len(train_dataset) // BATCH_SIZE
# 學習率
LEARNING_RATE = 0.01




## 建立模型
model = MLP().to('cuda')


# 優化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 損失函數(L1用於計算回歸殘差 MAE)
loss_function = nn.L1Loss()


# 將training set送入至data loader
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True
)



epochs = []
train_losses = []
val_losses = []
train_r2_list = []
val_r2_list = []
# 開始訓練
for epoch in range(EPOCH):

    train_loss = 0.0
    train_step = 0
    val_loss = 0.0
    val_step = 0
    train_total_r2score = 0.0
    val_total_r2score = 0.0

    for data, target in train_dataloader:
        r2score = R2Score().to('cuda')
        # 將資料讀入至cpu或gpu
        data, target = data.to('cuda'), target.to('cuda')
        # 清除梯度
        optimizer.zero_grad()
        # 進行預測
        pred = model(data)
        # 計算殘差(loss)
        loss = loss_function(pred, target)
        # 反向傳播(更新模型參數)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        r2s = r2score(pred, target)
        train_total_r2score += r2s.item()
        train_step += 1
        # print(f"epoch : {epoch+1} | step : {train_step} | loss : {loss.item()}")

    for data, target in val_dataloader:
        r2score = R2Score().to('cuda')
        # 將資料讀入至cpu或gpu
        data, target = data.to('cuda'), target.to('cuda')
        pred = model(data)
        loss = loss_function(pred, target.double())
        val_loss += loss.item()
        r2s = r2score(pred, target)
        val_total_r2score += r2s.item()
        val_step += 1

        # print(f"-epoch : {epoch+1} | step : {val_step} | loss : {loss.item()}")
    
    mean_train_loss = train_loss / train_step
    mean_val_loss = val_loss / val_step
    mean_train_r2score = train_total_r2score / train_step
    mean_val_r2score = val_total_r2score / val_step

    print(f"epoch : {epoch+1} | train loss : {mean_train_loss} | train R2: {mean_train_r2score} | val loss : {mean_val_loss} | val R2: {mean_val_r2score}")

    
    epochs.append(epoch)
    train_losses.append(mean_train_loss)
    val_losses.append(mean_val_loss)
    train_r2_list.append(mean_train_r2score)
    val_r2_list.append(mean_val_r2score)

    if (epoch+1) % 10 == 0 and (epoch+1) >= 10:
        generateGraph(epoch+1, epochs, train_losses, val_losses, train_r2_list, val_r2_list)



