import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 학습에 사용할 하이퍼 파라미터 설정
learning_rate = 0.001
training_epochs = 15
batch_size = 100



# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)  # 데이터 디렉토리 내의 이미지 파일 목록

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image




# 1-1. 데이터셋 정의
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)


mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

# 1-2. 데이터로더로 배치크기 지정
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)



# 2. 클래스로 모델 설계
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()    # CNN의 부모 클래스(torch.nn.Module) 호출
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1) # 원본이미지 크기, 그레이스케일로  
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
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
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

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
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

# 3. CNN 모델 정의
model = CNN().to(device)

# 4. 비용함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. 총 배치의 수 출력
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))    # batch_size =100 즉, 총 데이터 수는 60000개

# 6. 모델 training (시간이 꽤 걸립니다.)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        # print("X=",X, "Y=",Y)
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)    # loss 계산
        cost.backward()                    # 미분
        optimizer.step()                   # 가중치 업데이트

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 7. model test > 정확도 : 0.9869
# 학습을 진행하지 않을 것이므로 torch.no_grad()
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    print(prediction)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())