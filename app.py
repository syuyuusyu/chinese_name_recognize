from flask import Flask, request, jsonify

import torch

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

device = 'cpu'

def name_to_data(name):
    code = [ ord(char) for char in name]
    data = pad_sequences([code], maxlen=5, padding='post')[0]
    return torch.tensor(data, dtype=torch.long).unsqueeze(0).to(device)

model = torch.load('pth/entire_model.pth')
# model.to(device)
# model.eval()

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def predict():
    
    # 返回预测结果
    return jsonify({'prediction': 1})