import torch.nn as nn
from modules import ENetConv, DownSample, UpSample
import torch


class Net(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        features=[32, 64, 128, 256, 512],
    ):
        super(Net, self).__init__()
        self.down1 = DownSample(features[0])
        self.down2 = DownSample(features[1])
        self.down3 = DownSample(features[2])
        self.down4 = DownSample(features[3])

        self.up1 = UpSample(features[4])
        self.up2 = UpSample(features[3])
        self.up3 = UpSample(features[2])
        self.up4 = UpSample(features[1])

        self.conv_down1 = ENetConv(in_channel, features[0])
        self.conv_down2 = ENetConv(features[0], features[1])
        self.conv_down3 = ENetConv(features[1], features[2])
        self.conv_down4 = ENetConv(features[2], features[3])
        self.conv_down5 = ENetConv(features[3], features[4])

        self.conv_up1 = ENetConv(features[3] + features[4], features[3])
        self.conv_up2 = ENetConv(features[2] + features[3], features[2])
        self.conv_up3 = ENetConv(features[1] + features[2], features[1])
        self.conv_up4 = ENetConv(features[0] + features[1], features[0])
        # self.final = nn.Sequential(
        #     nn.Conv2d(features[1], out_channel, kernel_size=3, stride=1, padding=1),
        #     nn.LayerNorm(out_channel),
        # )
        self.final = ENetConv(features[0], out_channel)

    def forward(self, x):
        x_down1 = self.conv_down1(x)
        x_down2 = self.conv_down2(self.down1(x_down1))
        x_down3 = self.conv_down3(self.down2(x_down2))
        x_down4 = self.conv_down4(self.down3(x_down3))
        x_down5 = self.conv_down5(self.down4(x_down4))

        x_up1 = self.conv_up1(torch.cat((x_down4, self.up1(x_down5)), dim=1))
        x_up2 = self.conv_up2(torch.cat((x_down3, self.up2(x_up1)), dim=1))
        x_up3 = self.conv_up3(torch.cat((x_down2, self.up3(x_up2)), dim=1))
        x_up4 = self.conv_up4(torch.cat((x_down1, self.up4(x_up3)), dim=1))

        return self.final(x_up4)


def init_seq2seq(module):
    """Initialize weights for sequence-to-sequence learning.

    Defined in :numref:`sec_seq2seq`"""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqDecoder(nn.Module):
    """循环神经网络解码器"""

    def __init__(
        self,
        num_classes,
        embed_size,
        num_hiddens,
        num_layers,
        dropout=0.2,
        num_seq=None,
        num_heads=2,
    ):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(num_classes + 2, num_heads * embed_size)
        # self.embedding = PositionalEncoding(
        #     embed_size,
        #     num_classes + 2,
        # )
        self.state_linear = nn.Linear(
            num_hiddens,
            num_heads * embed_size,
        )
        self.attention = nn.MultiheadAttention(
            num_heads * embed_size, num_heads=8, dropout=dropout, batch_first=True
        )

        self.x_linear = nn.Linear(num_heads * embed_size, embed_size)
        self.num_heads = num_heads
        self.num_seq = num_seq
        self.num_classes = num_classes
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.dense = nn.Linear(num_hiddens, num_classes + 1)
        self.apply(init_seq2seq)

    def forward(self, X, state, init_state=None):
        # X:(batch_size,num_steps,num_heads*emb_size)
        # state:(num_layers,batch_size,num_hiddens)
        X = self.embedding(X)

        # context:(batch_size,1,num_heads*emb_size)
        if init_state is not None:
            context = self.state_linear(init_state[-1]).unsqueeze(dim=1)
        else:
            context = self.state_linear(state[-1]).unsqueeze(dim=1)
        # X_and_context:(batch_size,num_steps,num_heads*emb_size)
        X_and_context, _ = self.attention(X, context, context)
        # X_and_context:(num_steps,batch_size,emb_size)
        X_and_context = self.x_linear(X_and_context).permute(1,0,2)

        if self.training and self.num_seq is not None:
            states = []
            outputs = []
            index = X_and_context.shape[1] / self.num_seq
            assert index == int(index)
            area_range = torch.arange(0, index, dtype=torch.int64) * self.num_seq

            for i in area_range:
                output, area_state = self.rnn(
                    X_and_context[:, i : (i + self.num_seq)],
                    state[:, i : (i + self.num_seq)].contiguous(),
                )
                states.append(area_state)
                outputs.append(output)

            output = torch.cat(outputs, dim=1)
            state = torch.cat(states, dim=1)
        else:
            output, state = self.rnn(X_and_context, state.contiguous())
        output = self.dense(output).permute(1, 0, 2)
        return output, state


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的P
        self.P = torch.zeros((max_len, embed_size))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, embed_size, 2, dtype=torch.float32) / embed_size
        )
        self.P[:, 0::2] = torch.sin(X)
        self.P[:, 1::2] = torch.cos(X)
        # self.P[0, :] = 0
        self.P = nn.Parameter(self.P
                              )

    def forward(self, x):
        return self.P[x]


class Seq2SeqEncoder(nn.Module):
    """循环神经网络编码器"""

    def __init__(
        self,
        elevation_resolution,
        embed_size,
        num_hiddens,
        num_layers,
        dropout=0.2,
        num_seq=None,
    ):
        super().__init__()
        # self.embedding = nn.Embedding(elevation_resolution + 2, embed_size)
        self.embedding = PositionalEncoding(
            embed_size,
            elevation_resolution + 2,
        )
        self.num_seq = num_seq
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.apply(init_seq2seq)

    def forward(self, X):
        embs = self.embedding(X.to(torch.int64)).permute(1, 0, 2)
        if self.training and self.num_seq is not None:
            states = []
            outputs = []

            index = embs.shape[1] / self.num_seq
            assert index == int(index)

            area_range = torch.arange(0, index + 1, dtype=torch.int64) * self.num_seq
            for i in area_range:
                output, state = self.rnn(embs[:, i : (i + self.num_seq)])
                states.append(state)
                outputs.append(output)

            outputs = torch.cat(outputs, dim=1)
            state = torch.cat(states, dim=1)
        else:
            outputs, state = self.rnn(embs)
        return outputs, state


class RadarNet(nn.Module):
    """编码器+卷积+解码器"""

    def __init__(
        self,
        num_classes,
        elevation_resolution=100,
        embed_size=16,
        num_hiddens=3,
        num_layers=2,
        features=[32, 64, 128, 256, 512],
        dropout=0,
        num_seq=128,
    ):
        super().__init__()
        self.encoder = Seq2SeqEncoder(
            elevation_resolution,
            embed_size,
            num_hiddens,
            num_layers,
            dropout=dropout,
            num_seq=num_seq,
        )
        self.decoder = Seq2SeqDecoder(
            num_classes=num_classes,
            embed_size=embed_size,
            num_hiddens=num_hiddens,
            num_layers=num_layers,
            dropout=dropout,
            num_seq=num_seq,
        )

        self.convNet = Net(num_hiddens, num_hiddens, features)
        # self.convNet = DeepLabV3Plus(backbone="resnet50", num_classes=num_hiddens)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.elevation_resolution = elevation_resolution

    def MatToSeq(self, x):
        """
        矩阵转换到序列形式
        param:
            x shape: (batch_size,num_steps,num_rows,num_cols)
        return
            shape: (batch_size*num_rows*num_cols,num_steps)
        """
        num_steps = x.shape[1]
        return x.permute(0, 2, 3, 1).reshape(-1, num_steps)

    def forward(self, input, label_input):
        """
        param:
            input shape: (batch_size,num_steps,num_rows,num_cols)
        """
        batch_size, num_steps, num_rows, num_cols = input.shape
        # input_seq = self.MatToSeq(input)
        # _, state = self.encoder(input_seq)

        # state = state.reshape(
        #     self.num_layers, num_rows, num_cols, self.num_hiddens
        # ).permute(0, 3, 1, 2)

        # state = (
        #     self.convNet(state)
        #     .permute(0, 2, 3, 1)
        #     .reshape(self.num_layers, -1, self.num_hiddens)
        # )
        state = self.encode(input)

        label_input = self.MatToSeq(label_input)
        output, _ = self.decoder(label_input, state)
        return output.reshape(
            batch_size, num_rows, num_cols, num_steps, self.num_classes + 1
        ).permute(0, 4, 3, 1, 2)

    def encode(self, input):
        """
        param:
            input shape: (batch_size,num_steps,num_rows,num_cols)
        return:
            state shape:  (num_layers,batch_size*num_rows*num_cols,num_hiddens)
        """

        _, _, num_rows, num_cols = input.shape
        input = self.MatToSeq(input)
        _, state = self.encoder(input)

        state_mat = state.reshape(
            self.num_layers, num_rows, num_cols, self.num_hiddens
        ).permute(0, 3, 1, 2)

        state_mat = (
            self.convNet(state_mat)
            .permute(0, 2, 3, 1)
            .reshape(self.num_layers, -1, self.num_hiddens)
        )
        return state_mat + state
