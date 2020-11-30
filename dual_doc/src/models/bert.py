import torch
import torch.nn as nn
from pytorch_transformers import *
from torch.nn.utils.rnn import pad_sequence

from models.common_modules import PredictionBiLinear, PredictionBiLinearMulti
from settings import conditional_profiler

PREDICT_OPT_BILINEAR = 1
PREDICT_OPT_CONCAT_MLP = 2


class PredictionNet(nn.Module):
    def __init__(self, config, bert_hidden_size, hidden_size, opt = PREDICT_OPT_BILINEAR, sibling_model = None):
        super().__init__()
        self.linear_re = nn.Linear(bert_hidden_size, hidden_size)
        self.use_distance = True

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            self.bili = torch.nn.Bilinear(hidden_size + config.dis_size, hidden_size + config.dis_size,
                                          config.relation_num)
            # self.linear_re = nn.Linear((hidden_size+config.dis_size)*2, config.relation_num)
        else:
            # self.linear_re = nn.Linear(hidden_size*2, config.relation_num)
            self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)

        if sibling_model is not None:
            self.load_state_dict(sibling_model.state_dict())


    def forward(self, context, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h):
        #'''
        context = self.linear_re(context)

        start_re_output = torch.matmul(h_mapping, context)
        end_re_output = torch.matmul(t_mapping, context)
        '''
        start_re_output = torch.matmul(h_mapping, context)
        end_re_output = torch.matmul(t_mapping, context)

        start_re_output = self.linear_re(start_re_output)
        end_re_output = self.linear_re(end_re_output)
        #'''
        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
            # rep = torch.cat([s_rep, t_rep], dim=-1)
            # rep = s_rep - t_rep
            # predict_re = self.linear_re(rep)
            predict_re = self.bili(s_rep, t_rep)
        else:
            # rep = s_rep - t_rep
            # rep = torch.cat([s_rep, t_rep], dim=-1)
            # predict_re = self.linear_re(rep)
            predict_re = self.bili(start_re_output, end_re_output)

        # print(predict_re[0])

        return predict_re

class ModelParameter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        dat = torch.zeros((dim,))
        self.param = torch.nn.Parameter(dat,requires_grad= True)


    def forward(self, context_output):
        repeat_shape = context_output.shape[:-1]
        repeat_size = 1
        for r in repeat_shape:
            repeat_size *= r
        out = self.param.repeat(1,repeat_size).view(repeat_shape + (self.dim,))
        return out




class Attention(nn.Module):
    def __init__(self, context_size, query_size, hidden_size):
        super().__init__()
        # Additive attention
        #hidden_size -=1
        self.linear_c = torch.nn.Linear(context_size, hidden_size)
        self.linear_q = torch.nn.Linear(query_size, hidden_size,bias=False)
        self.linear_o = torch.nn.Linear(hidden_size, 1)

        self.tanh = nn.Tanh()
        #Multihead
        # num_head = 4
        # self.attention = nn.MultiheadAttention(vect_size, num_head)


    def forward(self, query, context, mapping = None, mask = None):
        #print(query.shape)
        #print(context.shape)
        #print(mapping.shape)

        n_batches, n_cands, h_input = query.shape
        if mask is None:
            mapping_mask = (mapping > 0.0).float()
        else:
            if len(mask.shape) == 2:
                mapping_mask = mask.unsqueeze(1)
            else:
                mapping_mask = mask
        res_q = self.linear_q(query)  #B*P*H
        res_c = self.linear_c(context) #B*L*H


        h_tmp = res_q.shape[2]

        weights_list = []
        epsilon = 0.00001

        #B*L*P*H
        tmp = res_c.unsqueeze(1) + res_q.unsqueeze(2)
        tmp = self.tanh(tmp)
        weights_logit = self.linear_o(tmp)
        weights_exp = mapping_mask *torch.exp(weights_logit.squeeze(-1))
        weights = weights_exp/ (weights_exp.sum(dim=2,keepdim=True)+epsilon)

        '''
        for i in range(n_cands):
            tmp.txt = res_c + res_q[:,i:i+1,:]
            tmp.txt = self.tanh(tmp.txt)
            weights_logit = self.linear_o(tmp.txt)


            weights_exp = mapping_mask[:, i] * torch.exp(weights_logit).squeeze(-1)
            weights_i = weights_exp / (weights_exp.sum(dim=1)+epsilon).unsqueeze(-1)
            weights_list.append(weights_i)

        weights = torch.stack(weights_list,1)
        #'''
        
        output = torch.bmm(weights,context)

        return output, weights


class AttentionModel(nn.Module):
    OPT_ADDITIVE = 1
    OPT_MULTIPLICATIVE = 2

    def __init__(self, context_size, state_size, hidden_size, att_opt=OPT_ADDITIVE):
        super(AttentionModel, self).__init__()

        self.net_W = nn.Linear(state_size, hidden_size)
        self.net_U = nn.Linear(context_size, hidden_size)
        self.net_v = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

        self.attn_softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

    def forward(self, context, state):
        '''
        :param context: (batchsize, state_length, state_size)
        :param state:   (batchsize, context_length, context_size)
        :return: output_context, att_weight
                output_context (batchsize, state_length, context_size)
                att_weight (batchsize, state_length, context_length)
        '''

        batchsize, state_length, state_size = state.shape
        _batchsize, context_length, context_size = context.shape
        hidden_size = self.hidden_size
        assert batchsize == _batchsize

        v_state = self.net_W(state)  # (batchsize, state_length, hidden_size)
        v_context = self.net_U(context)  # (batchsize, context_length, hidden_size)

        att_weight = []
        output_context = []
        for s in range(state_length):
            vs = v_state[:, s]  # (batchsize, hidden)
            att_weight_s = vs.view(batchsize, 1, hidden_size) + v_context  # (batchsize, context_length, hidden_size)
            att_weight_s = self.tanh(att_weight_s)
            att_weight_s = self.net_v(att_weight_s)  # (batchsize, context_length, 1)
            att_weight_s = self.attn_softmax(att_weight_s)

            output_s = torch.bmm(att_weight_s.transpose(1, 2), context)

            att_weight.append(att_weight_s.squeeze(-1))
            output_context.append(output_s.squeeze(1))

        att_weight = torch.stack(att_weight, dim=1)
        output_context = torch.stack(output_context, dim=1)

        return output_context, att_weight





class BERT_ext(nn.Module):
    name = "DualBERTaLA"
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_output_module = self.config.num_output_module
        self.use_entity_type = config.use_ner_emb
        self.mu_activation = config.mu_activation_option
        bert_hidden_size = 768
        hidden_size = config.hidden_size
        if self.num_output_module>1:
            self.twin_init = config.twin_init
        self.use_cross_attention = config.cross_encoder
        entity_vector_size = config.entity_type_size if self.use_entity_type else 0

        if self.use_entity_type:
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not self.config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False


        self.linear = nn.Linear(bert_hidden_size, hidden_size)
        context_hidden_size = hidden_size + entity_vector_size

        self.use_distance = True

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            vect_size = context_hidden_size + config.dis_size
        else:
            vect_size = context_hidden_size

        #num_head = 4
        #self.attention = nn.MultiheadAttention(vect_size, num_head)
        if self.use_cross_attention:
            self.attention = Attention(context_hidden_size, context_hidden_size, context_hidden_size)


        if self.num_output_module == 1:
            self.bili = PredictionBiLinear(vect_size, vect_size, config.relation_num)
        else:
            #self.bili_multi = PredictionBiLinearMulti(self.num_output_module, vect_size, vect_size, config.relation_num)
            bili_list = [PredictionBiLinear(vect_size, vect_size, config.relation_num) for _ in range(self.num_output_module)]
            if self.twin_init:
                bili_list[1].load_state_dict(bili_list[0].state_dict())
            self.bili_list = nn.ModuleList(bili_list)


    def fix_prediction_bias(self, bias):
        assert self.num_output_module == 1
        self.bili.fix_bias(bias)

    def add_prediction_bias(self, bias):
        assert self.num_output_module == 1
        self.bili.add_bias(bias)

    @conditional_profiler
    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs, sent_lengths, reverse_sent_idxs, context_masks,
                context_starts, ):

        # sent = torch.cat([sent, context_ch], dim=-1)
        # print(context_idxs.size())
        bert_out = self.bert(context_idxs, attention_mask=context_masks)[0]
        # print('output_1',context_output[0])

        '''
        padded_starts = torch.zeros(bert_out.shape[:-1],dtype = torch.long).cuda().contiguous()
        for i, context_start in enumerate(context_starts):  # repeat Batch times
            temp_cs = context_start.nonzero().squeeze(-1)
            length_temp_cs = temp_cs.shape[0]
            padded_starts[i, :length_temp_cs] = temp_cs  # [L]

        context_output2 = bert_out[padded_starts.unsqueeze(-1)]  # [B,L,1]
        '''

        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(bert_out, context_starts)]
        # print('output_2',context_output[0])
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
        

        # print('output_3',context_output[0])
        # print(context_output.size())
        context_output = torch.nn.functional.pad(context_output,
                                                 (0, 0, 0, context_idxs.size(-1) - context_output.size(-2)))

        #context_output = bert_out

        context = self.linear(context_output)
        if self.use_entity_type:
            context = torch.cat([context, self.ner_emb(context_ner)], dim=-1)

        start_re_output = torch.matmul(h_mapping, context)
        end_re_output = torch.matmul(t_mapping, context)

        if self.use_cross_attention:
            tail, _ = self.attention(start_re_output, context, t_mapping)
            head, _ = self.attention(end_re_output, context, h_mapping)
        else:
            head = start_re_output
            tail = end_re_output

        if self.use_distance:
            head_arr = [head]
            tail_arr = [tail]
            if self.use_distance:
                head_arr.append(self.dis_embed(dis_h_2_t))
                tail_arr.append(self.dis_embed(dis_t_2_h))

            s_rep = torch.cat(head_arr, dim=-1)
            t_rep = torch.cat(tail_arr, dim=-1)
        else:
            s_rep = head
            t_rep = tail


        #'''
        if self.num_output_module == 1:
            predict_re_ha_logit = self.bili(s_rep, t_rep)
            return predict_re_ha_logit
        #output_list = self.bili_multi(s_rep, t_rep)
        output_list = [bili(s_rep, t_rep) for bili in self.bili_list]
        '''
        predict_re_ha_logit = self.bili_ha(s_rep, t_rep)

        predict_re_ds_logit = self.bili_ds(s_rep, t_rep)

        if self.mu_activation == "param":
            diff_mean_logit = self.bili_mu(h_mapping)
        else:
            diff_mean_logit = self.bili_mu(s_rep, t_rep)
        diff_std_logit = self.bili_sigma(s_rep, t_rep)

        return predict_re_ha_logit, predict_re_ds_logit, diff_mean_logit, diff_std_logit
        #'''
        return output_list


class BERT_RE(nn.Module):
    name = "BERT"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_output_module = self.config.num_output_module
        self.use_entity_type = config.use_ner_emb
        self.mu_activation = config.mu_activation_option
        bert_hidden_size = 768
        hidden_size = config.hidden_size
        if self.num_output_module > 1:
            self.twin_init = config.twin_init
        self.use_cross_attention = config.cross_encoder
        entity_vector_size = config.entity_type_size if self.use_entity_type else 0

        if self.use_entity_type:
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not self.config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(bert_hidden_size, hidden_size)
        context_hidden_size = hidden_size + entity_vector_size

        self.use_distance = True

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            vect_size = context_hidden_size + config.dis_size
        else:
            vect_size = context_hidden_size

        # num_head = 4
        # self.attention = nn.MultiheadAttention(vect_size, num_head)
        if self.use_cross_attention:
            self.attention = Attention(context_hidden_size, context_hidden_size, context_hidden_size)

        # '''
        if self.num_output_module == 1:
            self.bili = PredictionBiLinear(vect_size, vect_size, config.relation_num)
        else:
            # self.bili_multi = PredictionBiLinearMulti(self.num_output_module, vect_size, vect_size, config.relation_num)
            bili_list = [PredictionBiLinear(vect_size, vect_size, config.relation_num) for _ in
                         range(self.num_output_module)]
            if self.twin_init:
                bili_list[1].load_state_dict(bili_list[0].state_dict())
            self.bili_list = nn.ModuleList(bili_list)


    def fix_prediction_bias(self, bias):
        assert self.num_output_module == 1
        self.bili.fix_bias(bias)

    def add_prediction_bias(self, bias):
        assert self.num_output_module == 1
        self.bili.add_bias(bias)

    @conditional_profiler
    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs, sent_lengths, reverse_sent_idxs, context_masks,
                context_starts, ):

        # sent = torch.cat([sent, context_ch], dim=-1)
        # print(context_idxs.size())
        bert_out = self.bert(context_idxs, attention_mask=context_masks)[0]
        # print('output_1',context_output[0])

        '''
        padded_starts = torch.zeros(bert_out.shape[:-1],dtype = torch.long).cuda().contiguous()
        for i, context_start in enumerate(context_starts):  # repeat Batch times
            temp_cs = context_start.nonzero().squeeze(-1)
            length_temp_cs = temp_cs.shape[0]
            padded_starts[i, :length_temp_cs] = temp_cs  # [L]

        context_output2 = bert_out[padded_starts.unsqueeze(-1)]  # [B,L,1]
        '''

        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(bert_out, context_starts)]
        # print('output_2',context_output[0])
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)

        # print('output_3',context_output[0])
        # print(context_output.size())
        context_output = torch.nn.functional.pad(context_output,
                                                 (0, 0, 0, context_idxs.size(-1) - context_output.size(-2)))

        # context_output = bert_out

        context = self.linear(context_output)
        if self.use_entity_type:
            context = torch.cat([context, self.ner_emb(context_ner)], dim=-1)

        start_re_output = torch.matmul(h_mapping, context)
        end_re_output = torch.matmul(t_mapping, context)

        if self.use_cross_attention:
            tail, _ = self.attention(start_re_output, context, t_mapping)
            head, _ = self.attention(end_re_output, context, h_mapping)
        else:
            head = start_re_output
            tail = end_re_output

        if self.use_distance:
            head_arr = [head]
            tail_arr = [tail]
            if self.use_distance:
                head_arr.append(self.dis_embed(dis_h_2_t))
                tail_arr.append(self.dis_embed(dis_t_2_h))

            s_rep = torch.cat(head_arr, dim=-1)
            t_rep = torch.cat(tail_arr, dim=-1)
        else:
            s_rep = head
            t_rep = tail

        # '''
        if self.num_output_module == 1:
            predict_re_ha_logit = self.bili(s_rep, t_rep)
            return predict_re_ha_logit
        # output_list = self.bili_multi(s_rep, t_rep)
        output_list = [bili(s_rep, t_rep) for bili in self.bili_list]

        return output_list


class DualSplit(nn.Module):
    name = "DualSplit"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_output_module = self.config.num_output_module
        self.use_entity_type = config.use_ner_emb
        self.mu_activation = config.mu_activation_option
        self.twin_init = config.twin_init
        bert_hidden_size = 768
        hidden_size = config.hidden_size
        if self.num_output_module > 1:
            self.twin_init = config.twin_init
        self.use_cross_attention = config.cross_encoder
        entity_vector_size = config.entity_type_size if self.use_entity_type else 0

        if self.use_entity_type:
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not self.config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        linear_list = []
        for i in range(self.num_output_module):
            linear_list.append(nn.Linear(bert_hidden_size, hidden_size))
        self.linear_list = nn.ModuleList(linear_list)


        context_hidden_size = hidden_size + entity_vector_size
        self.use_distance = True

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            vect_size = context_hidden_size + config.dis_size
        else:
            vect_size = context_hidden_size

        # num_head = 4
        # self.attention = nn.MultiheadAttention(vect_size, num_head)
        if self.use_cross_attention:
            attention_list = []
            for i in range(self.num_output_module):
                attention_list.append(Attention(context_hidden_size, context_hidden_size, context_hidden_size))
            self.attention_list = nn.ModuleList(attention_list)
            if self.twin_init:
                attention_list[1].load_state_dict(attention_list[0].state_dict())

        bili_list = [PredictionBiLinear(vect_size, vect_size, config.relation_num) for _ in
                     range(self.num_output_module)]
        if self.twin_init:
            bili_list[1].load_state_dict(bili_list[0].state_dict())

        self.bili_list = nn.ModuleList(bili_list)



    @conditional_profiler
    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs, sent_lengths, reverse_sent_idxs, context_masks,
                context_starts, ):

        # sent = torch.cat([sent, context_ch], dim=-1)
        # print(context_idxs.size())
        bert_out = self.bert(context_idxs, attention_mask=context_masks)[0]
        # print('output_1',context_output[0])

        '''
        padded_starts = torch.zeros(bert_out.shape[:-1],dtype = torch.long).cuda().contiguous()
        for i, context_start in enumerate(context_starts):  # repeat Batch times
            temp_cs = context_start.nonzero().squeeze(-1)
            length_temp_cs = temp_cs.shape[0]
            padded_starts[i, :length_temp_cs] = temp_cs  # [L]

        context_output2 = bert_out[padded_starts.unsqueeze(-1)]  # [B,L,1]
        '''

        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(bert_out, context_starts)]
        # print('output_2',context_output[0])
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)

        # print('output_3',context_output[0])
        # print(context_output.size())
        context_output = torch.nn.functional.pad(context_output,
                                                 (0, 0, 0, context_idxs.size(-1) - context_output.size(-2)))

        # context_output = bert_out

        output_list = []
        for o in range(self.num_output_module):
            context = self.linear_list[o](context_output)
            if self.use_entity_type:
                context = torch.cat([context, self.ner_emb(context_ner)], dim=-1)

            start_re_output = torch.matmul(h_mapping, context)
            end_re_output = torch.matmul(t_mapping, context)

            if self.use_cross_attention:
                tail, _ = self.attention_list[o](start_re_output, context, t_mapping)
                head, _ = self.attention_list[o](end_re_output, context, h_mapping)
            else:
                head = start_re_output
                tail = end_re_output

            if self.use_distance:
                head_arr = [head]
                tail_arr = [tail]
                if self.use_distance:
                    head_arr.append(self.dis_embed(dis_h_2_t))
                    tail_arr.append(self.dis_embed(dis_t_2_h))

                s_rep = torch.cat(head_arr, dim=-1)
                t_rep = torch.cat(tail_arr, dim=-1)
            else:
                s_rep = head
                t_rep = tail


            predict_re_logit = self.bili_list[o](s_rep, t_rep)
            output_list.append(predict_re_logit)
        if self.num_output_module == 1:
            return output_list[0]

        return output_list




if __name__ =="__main__":
    pass
    '''
    paramnet = ModelParameter(4)
    input = torch.zeros((3,2,7))
    input += 1
    out = paramnet(input)

    print(input.shape)
    print(out.shape)
    print(out)
    '''
    pass
