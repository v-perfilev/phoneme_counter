from torch.utils.data import Dataset


class SpeechPhonemesDataset(Dataset):
    data = []

    def __init__(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
