import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn.utils import rnn

from models.bert import Attention
from models.common_modules import PredictionBiLinear


class CommonNetBiLSTM(nn.Module):
    def __init__(self, config, hidden_size, output_modules = 1):
        super().__init__()
        self.linear_re = nn.Linear(hidden_size * 2, hidden_size)
        self.use_distance = True
        self.use_cross_attention = config.cross_encoder

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            dis_size = config.dis_size
        else:
            dis_size = 0

        if self.use_cross_attention:
            self.attention = Attention(hidden_size, hidden_size, hidden_size)

        self.num_output_modules = output_modules
        if output_modules == 1:
            self.bili = PredictionBiLinear(hidden_size + dis_size, hidden_size + dis_size,
                                          config.relation_num)
        else:
            bili_list = [PredictionBiLinear(hidden_size + dis_size, hidden_size + dis_size, config.relation_num) for _ in range(output_modules)]
            self.bili_list = nn.ModuleList(bili_list)

    def fix_prediction_bias(self, bias):
        assert self.num_output_modules == 1
        self.bili.fix_bias(bias)

    def add_prediction_bias(self, bias):
        assert self.num_output_modules == 1
        self.bili.add_bias(bias)


    def forward(self, sent, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h):
        context_output = torch.relu(self.linear_re(sent))
        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)

        if self.use_cross_attention:
            end_re_output, _ = self.attention(start_re_output, context_output, t_mapping)
            start_re_output, _ = self.attention(end_re_output, context_output, h_mapping)

        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
        else:
            s_rep = start_re_output
            t_rep = end_re_output

        if self.num_output_modules == 1:
            predict_re = self.bili(s_rep, t_rep)
            return predict_re

        output = [self.bili_list[i](s_rep, t_rep) for i in range(self.num_output_modules)]
        return output

class BiLSTM(nn.Module):
    name = "BiLSTM"

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.light = True
        self.config = config
        self.num_output_module = self.config.num_output_module
        if self.num_output_module>1:
            self.twin_init = config.twin_init

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))

        self.word_emb.weight.requires_grad = False
        self.use_entity_type = True
        self.use_coreference = True
        self.use_distance = True
        self.use_cross_attention = config.cross_encoder

        # performance is similar with char_embed
        # self.char_emb = nn.Embedding(config.data_char_vec.shape[0], config.data_char_vec.shape[1])
        # self.char_emb.weight.data.copy_(torch.from_numpy(config.data_char_vec))

        # char_dim = config.data_char_vec.shape[1]
        # char_hidden = 100
        # self.char_cnn = nn.Conv1d(char_dim,  char_hidden, 5)

        hidden_size = config.hidden_size
        input_size = config.data_word_vec.shape[1]
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            input_size += config.coref_size
            # self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        # input_size += char_hidden

        self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)


        #self.pred_net = CommonNetBiLSTM(config, hidden_size)


        if self.light:
            self.output_net = CommonNetBiLSTM(config, hidden_size, output_modules= self.num_output_module)
        else:
            output_net_list = [CommonNetBiLSTM(config, hidden_size) for _ in range(self.num_output_module)]
            self.output_net_list = nn.ModuleList(output_net_list)
            #for i, out_net in enumerate(self.output_net_list):
            #    self.add_module(f"out_{i}", out_net)

            if self.num_output_module > 1 and self.twin_init:
                self.output_net_list[0].load_state_dict(self.output_net_list[1].state_dict())


    def fix_prediction_bias(self, bias):
        assert self.num_output_module == 1
        self.output_net.fix_prediction_bias(bias)

    def add_prediction_bias(self, bias):
        assert self.num_output_module == 1
        self.output_net.add_prediction_bias(bias)

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h):
        # para_size, char_size, bsz = context_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)
        # context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        # context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)

        sent = self.word_emb(context_idxs)
        if self.use_coreference:
            sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)

        if self.use_entity_type:
            sent = torch.cat([sent, self.ner_emb(context_ner)], dim=-1)

        # sent = torch.cat([sent, context_ch], dim=-1)
        context_output = self.rnn(sent, context_lens)

        if self.light:
            output = self.output_net(context_output, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h)
            return output

        #predict_re = self.pred_net(context_output, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h)
        output = [self.output_net_list[i](context_output, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h) for i in range(self.num_output_module)]
        if self.num_output_module == 1:
            return output[0]

        return output








#TRAINING_INSTANCE_OPT_HA_ONLY = 1
#TRAINING_INSTANCE_OPT_DS_ONLY = 2
#TRAINING_INSTANCE_OPT_DS_HA = 3





class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, hidden)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        #lengths = torch.tensor(input_lengths)
        lengths = input_lengths

        for i in range(self.nlayers):
            self.rnns[i].flatten_parameters()

        lens, indices = torch.sort(lengths, 0, True)
        input = input[indices]
        _, _indices = torch.sort(indices, 0)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        # if input_lengths is not None:
        #    lens = input_lengths.data.cpu().numpy()
        lens[lens == 0] = 1

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, (hidden, c) = self.rnns[i](output, (hidden, c))

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        for i, output in enumerate(outputs):
            outputs[i] = output[_indices]
        if self.concat:
            return torch.cat(outputs, dim=-1)
        return outputs[-1]



class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1)
