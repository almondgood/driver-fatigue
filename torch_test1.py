import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)




# 1-1. 데이터셋 정의
# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        
        self.classes = sorted(os.listdir(data_dir))  # 클래스 목록
        self.classes.remove('test')
        self.classes.remove('weights')
        
       
        self.classes = [item for item in self.classes if 'original' not in item]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # 클래스를 인덱스로 매핑
        self.images = []  # 이미지 파일 경로
        self.targets = []  # 레이블
        self.transform = transform
        self.target_transform = target_transform

         # 각 클래스의 이미지 파일 경로와 레이블 지정
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                img_path = os.path.normpath(img_path)  # 플랫폼에 맞게 경로 정규화
                if 'original' not in img_path:
                    self.images.append(img_path)
                    self.targets.append(self.class_to_idx[cls_name])
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        self.img_path = img_path
        target = int(self.targets[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
                   

        return image, target
    
    
# 합성층에 사용될 변환 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.ToTensor()  # 파이토치 텐서로 변환
])
    
# 커스텀 데이터셋 인스턴스 생성
custom_dataset = CustomDataset(data_dir='./data/', transform=transform)



# 이미지와 레이블을 저장할 리스트 초기화
images = []
targets = []

# 데이터셋 순회
for img, target in custom_dataset:
    images.append(img)
    targets.append(target)

# 리스트를 텐서로 변환
x_train = torch.stack(images)  # 이미지 리스트를 하나의 텐서로
t_train = torch.tensor(targets)  # 레이블 리스트를 텐서로

# 결과 확인
print(x_train.shape, t_train.shape)




# 2. 클래스로 모델 설계
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()    # CNN의 부모 클래스(torch.nn.Module) 호출
        
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(16 * 16 * 64, 2, bias=True)

        # 전결합층 한정으로 가중치 초기화 
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
        ### 세이비어 초기화: 가중치 초기화가 모델에 영향을 미침에 따라 초기화 방법 제안.
        ### 방법은 2가지 균등분포 or 정규분포로 초기화. 이전층의 뉴런 개수 / 다음층의 뉴런개수 사용하여 균등분포 범위 정해서 초기화
        ### He 초기화 방법도 있음. 
        ### sigmoid or tanh  사용할 경우 세이비어 초기화 효율적
        ### ReLU 계열 함수 사용할 때는 He 초기화 방법 효율적
        ### ReLU + He 초기화 방법이 좀 더 보편적임

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out


# 학습에 사용할 하이퍼 파라미터 설정
learning_rate = 0.001
training_epochs = 7
batch_size = 1



def training():
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_size = x_train.size(0)
    batch_size = 1  
    
    train_loss_list = []

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(np.ceil(train_size / batch_size))

        for i in range(total_batch):
            
            batch_indices = np.random.choice(train_size, batch_size, replace=False)
            X_batch = x_train[batch_indices].to(device)
            Y_batch = t_train[batch_indices].to(device)

            optimizer.zero_grad()
            hypothesis = model(X_batch)
            cost = criterion(hypothesis, Y_batch)
            cost.backward()
            optimizer.step()

            avg_cost += cost.item() / total_batch  

        #학습 경과 기록
        train_loss_list.append(avg_cost)  

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    # 학습이 끝난 후 모델 저장
    L=train_loss_list
    plt.plot(L)
    plt.xlabel('Epoch')
    plt.ylabel('Average Cost')
    plt.show()
    torch.save(model.state_dict(),'C:/Users/0/.anaconda/model.pth')    
    return model    
    
     
        


# 7. model test > 정확도 : 
# 학습을 진행하지 않을 것이므로 torch.no_grad()
# def evaluation():
#         # 모델 평가 모드로 전환
#         model = CNN().to(device)
#         model.load_state_dict(torch.load('C:/Users/0/.anaconda/model.pth'))
#         model.eval()
        
#         # 정확도 계산을 위해 레이블 저장할 리스트
#         all_labels = []
#         # 예측 결과 저장할 리스트
#         all_predictions = []
        
#         # DataLoader를 이용하여 테스트 데이터셋의 배치들에 대해 예측 수행
#         for images, labels in data_loader:
#             images = images.to(device)
#             labels = labels.to(device)
            
#             # 모델로부터 로짓(확률값이 아닌 출력) 계산
#             logits = model(images)
            
#             # 소프트맥스 함수를 사용하여 확률로 변환
#             probabilities = F.softmax(logits, dim=1)

#             # 예측값과 정답 레이블 저장
#             all_predictions.extend(torch.argmax(probabilities, dim=1).cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
            
        
#         # 리스트를 NumPy 배열로 변환
#         all_predictions = np.array(all_predictions)
#         all_labels = np.array(all_labels)
        
#         # 정확도 계산
#         accuracy = np.mean(all_predictions == all_labels)
        
#         # 정확도 출력
#         print('Accuracy:', accuracy)

def main():
    training()
    #evaluation()

if __name__ == '__main__':
    main()    