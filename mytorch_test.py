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

# 합성층에 사용될 변환 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기 조정
    transforms.ToTensor()  # 파이토치 텐서로 변환
])

# 커스텀 데이터셋 인스턴스 생성
custom_dataset = CustomDataset(data_dir='./data', transform=transform)

# 데이터로더 생성
data_loader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)


# 예시로 데이터로더를 통해 데이터 접근
for batch in data_loader:
    # 여기에 원하는 작업 수행
    print(batch.shape)  # 예시로 출력: (batch_size, channels, height, width)