from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import onnxruntime as nxrun


g_model_name = 'gru_toy.onnx'
g_rnn_batch_first = False


class RNNPredictor(nn.Module):
    def __init__(self, embedding_dim, label_size, hidden_dim=64):
        super(RNNPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=g_rnn_batch_first)
        # self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=g_rnn_batch_first)
        self.fc = nn.Linear(hidden_dim, label_size)

    def forward(self, x):
        # print('-------------------------')
        rnn_out, _ = self.rnn(x)
        # print('rnn_out.shape = ', rnn_out.shape)
        # No batch
        rnn_out = rnn_out.view(-1, self.hidden_dim)
        # print('rnn_out.shape = ', rnn_out.shape)
        rnn_out = F.tanh(rnn_out)
        # print('tanh.shape = ', rnn_out.shape)
        y = self.fc(rnn_out)
        # print('y.shape = ', y.shape)
        # print('-------------------------')
        return y


def data_gen(batch_first=g_rnn_batch_first):
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("The man read that book".split(), ["DET", "NN", "V", "DET", "NN"])
    ]

    x_dict = {}
    y_dict = {}
    for x, y in training_data:
        for ch in x:
            if ch not in x_dict:
                x_dict[ch] = len(x_dict)

        for ch in y:
            if ch not in y_dict:
                y_dict[ch] = len(y_dict)

    seq_len = len(training_data[0][0])
    batch = 1
    ret_x = np.zeros((seq_len, batch, len(x_dict)), dtype=np.float32) #ont-hot
    ret_y = np.zeros(seq_len, dtype=np.int)

    for x, y in training_data:
        index = 0
        for ch_x, ch_y in zip(x, y):
            ret_x[index, 0, x_dict[ch_x]] = 1
            ret_y[index] = y_dict[ch_y]
            index += 1

        if batch_first:
            ret_x.transpose(1, 0, 2)

        yield ret_x, ret_y


def to_onnx(model, model_name, dummy_input):
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, model_name, 
        verbose=True, input_names=input_names, output_names=output_names, opset_version=10)


if __name__ == '__main__':
    for x, y in data_gen(g_rnn_batch_first):
        test_x = x
        test_y = y
        break

    vocab_size = test_x.shape[2]
    model = RNNPredictor(vocab_size, 3)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(300):
        for x, y in data_gen(g_rnn_batch_first):
            torch_x = torch.from_numpy(x)
            torch_golden_y = torch.from_numpy(y)

            model.zero_grad()
            torch_pred = model(torch_x)

            loss = loss_function(torch_pred, torch_golden_y)
            loss.backward()
            optimizer.step()


    with torch.no_grad():
        torch_x = torch.from_numpy(test_x)
        torch_pred = model(torch_x)
        print('torch pred scores = ', torch_pred)
        print('torch pred label = ', [torch.argmax(y) for y in torch_pred])

    print(model)
    to_onnx(model, g_model_name, torch_x)
    onnx_sess = nxrun.InferenceSession(g_model_name)
    input_name = onnx_sess.get_inputs()[0].name
    onnx_pred = onnx_sess.run(None, {input_name: test_x})[0]
    print('onnx pred scores = ', onnx_pred)
    print('onnx pred label = ', [np.argmax(y) for y in onnx_pred])
    print('golden = ', y)
