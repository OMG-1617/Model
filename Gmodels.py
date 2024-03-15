
import torch
import torch.nn as nn
import time
import math
import torch.nn.functional as F

class FaceFeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FaceFeatureExtractorCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'D:/Net/FaceFeatureExtractorCNN_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        # self.load_state_dict(torch.load(path))
        self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


class FaceFeatureExtractor(nn.Module):
    def __init__(self, feature_size=16, pretrain=True):
        super(FaceFeatureExtractor, self).__init__()
        cnn = FaceFeatureExtractorCNN()
        if pretrain:
            cnn.load('D:/Net(SA-convLSTM+cross)/pretrained_cnn.pth')
        self.cnn = cnn.net
        self.rnn = SAConvLSTM(128, 128, 64, (3, 3), 1, True, True, True)
        self.fc = nn.Linear(128*6*6, feature_size)

    def forward(self, x):
        # input should be 5 dimension: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)
        cnn_output = self.cnn(x)
        rnn_input = cnn_output.view(b, t, 128, 6, 6)
        rnn_output = self.rnn(rnn_input)
        # print(rnn_output.shape)
        # rnn_output = torch.stack(rnn_output)
        rnn_output = torch.flatten(rnn_output, 1)
        output = self.fc(rnn_output)
        return output


class BioFeatureExtractor(nn.Module):
    def __init__(self, input_size=32, feature_size=40):
        super(BioFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=24, kernel_size=5),
            nn.BatchNorm1d(num_features=24),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(8*120, feature_size)

    def forward(self,x):
        x = self.cnn(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


class NetVision(nn.Module):
    def __init__(self,feature_size=16,pretrain=True):
        super(NetVision,self).__init__()
        self.features = FaceFeatureExtractor(feature_size=feature_size,pretrain=pretrain)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'D:/Net/checkpoints/' + 'face_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))


class NetBio(nn.Module):
    def __init__(self, input_size=32, feature_size=64):
        super(NetBio, self).__init__()
        self.features = BioFeatureExtractor(input_size=input_size, feature_size=feature_size)
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'D:/Net/checkpoints/' + 'physiological_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

class BertAttention(nn.Module):
    def __init__(self, num_heads=4, ctx_dim=16):
        super().__init__()
        self.num_attention_heads = 4
        self.hidden_size = 16
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads))
        self.num_attention_heads = self.num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = self.hidden_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout()

    def transpose_for_scores(self, x):
        bsz, hsz = x.shape
        x = x.view(bsz,  self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

BertLayerNorm = torch.nn.LayerNorm

class BertAttOutput(nn.Module):
    def __init__(self):
        super(BertAttOutput, self).__init__()
        self.hidden_size = 16
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = BertLayerNorm(self.hidden_size)
        self.dropout = nn.Dropout()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertCrossattLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = BertAttention()
        self.output = BertAttOutput()

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        # input_tensor = input_tensor.permute(1, 0)
        # ctx_tensor = ctx_tensor.permute(1, 0)
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

class BertSelfattLayer(nn.Module):
    def __init__(self):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention()
        self.output = BertAttOutput()

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class MultTransfromer(nn.Module):
    def __init__(self,bio_input_size=32, face_feature_size=16, bio_feature_size=64,pretrain=True):
        super(MultTransfromer, self).__init__()
        super().__init__()
        self.face_feature_extractor = FaceFeatureExtractor(feature_size=face_feature_size, pretrain=pretrain)

        self.bio_feature_extractor = BioFeatureExtractor(input_size=bio_input_size, feature_size=bio_feature_size)

        self.fusion_feature = BertCrossattLayer()

        self.face_self_att = BertSelfattLayer()
        self.eeg_self_att = BertSelfattLayer()

        # self.try_conv_1x1 = nn.Conv1d(1024, 512,
        #                               kernel_size=1, stride=1)

        self.to_trans_face = nn.Sequential(
            nn.LayerNorm(face_feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            GeLU(),

        )

        self.to_trans_eeg = nn.Sequential(
            nn.LayerNorm(bio_feature_size),
            nn.Linear(bio_feature_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 16),
            GeLU(),
        )

        # self.after_trans_eeg = nn.Sequential(#可能会根据449修改，因为涉及到两个叠加
        #     nn.Linear(16, 8),
        #
        # )

        # self.after_trans_face = nn.Sequential(
        #     nn.Linear(16, 8),
        #     GeLU(),
        # )

        self.classifier = nn.Sequential(
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 1),
            nn.Sigmoid()

        )

        self.LayerNorm_face = torch.nn.LayerNorm(16)
        self.LayerNorm_eeg = torch.nn.LayerNorm(16)


        # self.pool_vid = nn.AdaptiveAvgPool1d(1)
        # self.pool_aud = nn.AdaptiveAvgPool1d(1)

        # self.metrics = MetricCollection([Accuracy(), Precision(num_classes=2, average='macro'),Recall(num_classes=2, average='macro'), F1Score(num_classes=2, average='macro')])



    def self_att(self, eeg_input, face_input, eeg_attention_mask=None, face_attention_mask=None):
        # Self Attention
        eeg_att_output = self.eeg_self_att(eeg_input, eeg_attention_mask)
        face_att_output = self.face_self_att(face_input, face_attention_mask)
        return eeg_att_output, face_att_output

    def cross_att(self, eeg_input, face_input, eeg_attention_mask=None, face_attention_mask=None):
        # Cross Attention
        eeg_att_output = self.fusion_feature(eeg_input, face_input, ctx_att_mask=face_attention_mask)
        face_att_output = self.fusion_feature(face_input, eeg_input, ctx_att_mask=eeg_attention_mask)
        return eeg_att_output, face_att_output

    def forward(self, x):
        img_features = self.face_feature_extractor(x[0])
        bio_features = self.bio_feature_extractor(x[1])


        bio_features = self.to_trans_eeg(bio_features)

        eeg_att_output, face_att_output = self.self_att(bio_features, img_features)  # N.B. first self attention and then cross

        eeg_att_output, face_att_output = self.cross_att(eeg_att_output, face_att_output)

        eeg_att_output, face_att_output = self.self_att(eeg_att_output, face_att_output)  # N.B. first self attention and then cross
        # eeg_att_output = bio_features + eeg_att_output
        # face_att_output = img_features + face_att_output
        eeg_att_output = self.LayerNorm_eeg(bio_features + eeg_att_output)
        face_att_output = self.LayerNorm_face(img_features + face_att_output)

        # lang_att_output = self.pool_aud(lang_att_output.permute(1, 0, 2)).squeeze()
        # visn_att_output = self.pool_vid(visn_att_output.permute(1, 0, 2)).squeeze()

        # eeg_att_output = self.after_trans_eeg(eeg_att_output)
        # face_att_output = self.after_trans_face(face_att_output)

        concated_features = torch.cat((face_att_output, eeg_att_output), dim=1)
        logits = self.classifier(concated_features)
        logits = logits.squeeze(-1)
        return logits
    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'D:/DeepVANet-main/modal0/checkpoints/' + 'fusion_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

class Net(nn.Module):
    def __init__(self, bio_input_size=32, face_feature_size=16, bio_feature_size=64,pretrain=True):
        super(Net,self).__init__()
        self.face_feature_extractor = FaceFeatureExtractor(feature_size=face_feature_size,pretrain=pretrain)

        self.bio_feature_extractor = BioFeatureExtractor(input_size=bio_input_size, feature_size=bio_feature_size)

        self.classifier = nn.Sequential(
            nn.Linear(face_feature_size + bio_feature_size, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        img_features = self.face_feature_extractor(x[0])
        bio_features = self.bio_feature_extractor(x[1])
        features = torch.cat([img_features,bio_features.float()],dim=1)
        output = self.classifier(features)
        output = output.squeeze(-1)
        return output

    def save(self, name=None):
        """
        save the model
        """
        if name is None:
            prefix = 'D:/Net/checkpoints/' + 'fusion_classifier_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        self.load_state_dict(torch.load(path))
        # self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

class SA_Attn_Mem(nn.Module):
    # SAM 
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, (1, 1))
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, (1, 1))
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, (1, 1))

        self.layer_v = nn.Conv2d(input_dim, input_dim, (1, 1))
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, (1, 1))

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, (1, 1))
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, (1, 1))

    def forward(self, h, m):
        batch_size, channels, H, W = h.shape
        # **********************  feature aggregation ******************** #

        # Use 1x1 convolution for Q,K,V Generation
        K_h = self.layer_k(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)

        Q_h = self.layer_q(h)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)

        K_m = self.layer_k2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)

        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)

        # **********************  hidden h attention ******************** #
        # [batch_size,H*W,H*W]
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)

        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        # **********************  memory m attention ******************** #
        # [batch_size,H*W,H*W]
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)

        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)   # [batch_size,in_channels*2,H,W]

        # Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        #
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m

class SAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, att_hidden_dim, kernel_size, bias):
        """
           Initialize SA ConvLSTM cell.
           Parameters
           ---------
           input_dim: int
               Number of channels of input tensor.
           hidden_dim: int
               Number of channels of hidden state.
           kernel_size: (int, int)
               Size of the convolutional kernel.
           bias: bool
               Whether to add the bias.
           att_hidden_dim: int
               Number of channels of attention hidden state
        """
        super(SAConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.attention_layer = SA_Attn_Mem(hidden_dim, att_hidden_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=kernel_size,
                      padding=self.padding),
            nn.GroupNorm(4 * hidden_dim, 4 * hidden_dim)
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur, m_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        h_next, m_next = self.attention_layer(h_next, m_cur)
        return h_next, c_next, m_next

    # initialize h, c, m
    def init_hidden(self, batch_size, image_size,device):
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        m = torch.zeros(batch_size, self.hidden_dim, height, width,device=device)
        return h, c, m


class SAConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_hidden_dim, kernel_size , num_layers, batch_first=False, bias=True,
                 return_all_layers=False):
        super().__init__()
        self._check_kernel_size_consistency(kernel_size)
        # make sure that both "kernel_size" and 'hidden_dim' are lists having len=num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        attn_hidden_dim = self._extend_for_multilayer(attn_hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == len(attn_hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.attn_hidden_dim = attn_hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=self.hidden_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            att_hidden_dim=self.attn_hidden_dim[i],
                                            bias=self.bias,
                                            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t,b,c,h,w)->(b,t,c,h,w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w),device=input_tensor.device)
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c, m = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c, m = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c, m])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c, m])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return h
        'layer_output_list, last_state_list'

    def _init_hidden(self, batch_size, image_size,device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size,device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
                isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
