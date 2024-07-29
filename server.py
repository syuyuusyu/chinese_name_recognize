from flask import Flask, request, jsonify
import torch
from module import Module1

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

model = Module1(vocab_size=65536, embedding_dim=50, hidden_dim=20)
# 加载模型状态字典
model.load_state_dict(torch.load('pth/name_dict.pth', map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)

@app.route('/predict/name', methods=['POST'])
def predict():
    # 获取请求中的 JSON 数据
    data = request.json
    name = data['name']
    
    # 处理输入数据
    input_data = name_to_data(name)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs, 1)
    
    # 返回预测结果
    return jsonify({'prediction': int(predicted.item())==1})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006)