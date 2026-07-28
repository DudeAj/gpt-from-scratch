from torch.utils.data import DataLoader
from dataset.text_dataset import GPTDataset

tokens = [1,2,3,4,5,6,7,8,9]

dataset = GPTDataset(
    tokens=tokens,
    block_size=4
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

for x, y in loader:

    print("Input")
    print(x)

    print("Target")
    print(y)

    print("-"*30)