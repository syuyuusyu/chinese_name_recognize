import torch
from torch.utils.data import Dataset


def pad_sequences(sequences, maxlen, padding='post', value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if padding == 'post':
                padded_seq = seq[:maxlen]
            elif padding == 'pre':
                padded_seq = seq[-maxlen:]
        else:
            if padding == 'post':
                padded_seq = seq + [value] * (maxlen - len(seq))
            elif padding == 'pre':
                padded_seq = [value] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return padded_sequences

class NameData(Dataset):
    def __init__(self,names,non_names) -> None:
        super().__init__()
        self.labels = [1]* len(names) + [0] * len(non_names)
        encode_arr = []
        for string in names+non_names:
            encode = [ ord(char) for char in string]
            encode_arr.append(encode)
        self.data = pad_sequences(encode_arr, maxlen=5, padding='post')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.long), torch.tensor(label, dtype=torch.long)