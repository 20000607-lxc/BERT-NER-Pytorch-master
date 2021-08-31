import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from .transformers_master.models.gpt2.modeling_gpt2 import GPT2Model as New_GPT2
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel
from models.p_tuning.label_embedder import LabelEmbeder

class GPT2SoftmaxForNer_LE(torch.nn.Module):
    """
    one step 输出input 对应的 hidden state
    """
    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        self.num_labels = config.num_labels
        if model_name == None:# 用于load gpt2-large
            model_name = 'gpt2'
        self.gpt2 = New_GPT2.from_pretrained(model_name)# 可以接受inputs_embeds和input_ids
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        self.embeddings = GPT2LMHeadModel.from_pretrained(model_name).base_model.get_input_embeddings()#embedding是GPT2LMHeadModel的embedding

        # self.embeddings.weight.requires_grad = False
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False
        # perform fine_tuning

        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.loss_type = 'ce'
        self.device = device

        self.pseudo_token_id = 50257# prompt word 的id
        self.hidden_size = self.embeddings.embedding_dim
        self.template = template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)

        self.num_entities = 17# todo for conll2003 区分bio or bieso

        self.label_embedding = LabelEmbeder([self.num_entities], self.hidden_size, device)
        self.label_embedding = self.label_embedding.to(self.device)
        self.attn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1, bias=False)

        print("***************** init GPT2SoftmaxForNer with label embedding *********************")

    def get_query(self, input_id, prompt_tokens):
        input = []
        prompt1 = []
        prompt2 = []
        count = 0
        for i in range(self.template[0]):
            prompt1.append(prompt_tokens[0])
        for i in range(self.template[1]):
            prompt2.append(prompt_tokens[0])

        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())
        if self.template[0] == self.template[1]:
            query = prompt1 + input + prompt2 + input
        else:
            query = prompt1 + input + prompt2

        return query, count

    def embed_input(self, queries, counts):
        """
        turn the queries(word index) :[batch_size,query_length]
        into embeddings: [batch_size,query_length,768]
        """
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1

        replace_embeds = self.prompt_encoder()
        raw_embeds = self.embeddings(queries_for_embedding)

        for bidx in range(bz):
            for i in range(self.template[0]):
                raw_embeds[bidx, i, :] = replace_embeds[i, :]
            for i in range(self.template[1]):
                raw_embeds[bidx, i+counts[bidx]+self.template[0], :] = replace_embeds[i+self.template[0], :]
        return raw_embeds

    def attention(self, input_state, label_embedding, bz):
        """
        Args:
            input_state: [batch_size, hidden state]
            label_embedding: [batch_size, num_label_type, hidden state]

        Returns:
            output_state:[batch_size, hidden state]
        """
        input_state_attn = self.attn_linear(input_state)
        input_state_expanded = input_state_attn.unsqueeze(1).expand(bz, self.num_entities, self.hidden_size).contiguous()  # B x 5 x hidden_dim
        input_state_expanded = input_state_expanded.view(-1, self.hidden_size)     # B*5 x hidden_dim

        label_embedding_fea = label_embedding.view(-1, self.hidden_size)
        att_features = label_embedding_fea + input_state_expanded    # B*self.num_entities x hidden_dim
        e = torch.tanh(att_features)
        scores = self.fc(e)                                      # B*self.num_entities x 1
        scores = scores.view(-1, self.num_entities)                              # B x self.num_entities
        attn_dist_ = F.softmax(scores, dim=1)                    # B x self.num_entities
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        attn_dist = attn_dist.unsqueeze(1)                        # B x 1 x self.num_entities
        output_state = torch.bmm(attn_dist, label_embedding)      # B x 1 x self.num_entities * self.num_entities x hidden_dim

        output_state = output_state.squeeze(1)
        output_state += input_state
        return output_state

    def add_label_embedding(self, sequence_output, label_init, counts=None):
        """
        Args:
            sequence_output: the input embeds or the gpt2 output logits
        Returns:
            output: add label embedding to input embeds

        """
        bz = sequence_output.shape[0]
        new_sequence_output = torch.zeros_like(sequence_output).to(self.device)
        label_embedding = torch.empty(bz, self.num_entities, self.hidden_size).to(self.device)

        for k in range(bz):
            label_embedding[k, :, :] = label_init

        for i in range(sequence_output.shape[1]):
            new_sequence_output[:, i, :] = self.attention(sequence_output[:, i, :], label_embedding, bz)
            # donot use a = ...a , which will trigger error during loss.backward() cause this assigns value to one variable repeatedly

        for bidx in range(bz):
            # input ids 对应的embedding不变
            new_sequence_output[bidx, self.template[0]:counts[bidx]+self.template[0], :] =\
                sequence_output[bidx, self.template[0]:counts[bidx]+self.template[0], :]

        return new_sequence_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        """

        Args:
            input_ids: padded seuqence:[batch_size, max_length]
            if Chinese: input_ids = [101,...,102, 0,...,0]
            attention_mask: [batch_size, max_length]
            token_type_ids: [batch_size, max_length]
            position_ids: [batch_size, max_length]
            head_mask: [batch_size, max_length]
            labels: [batch_size, max_length]

        Returns:
            outputs

        """
        label_init = self.label_embedding()
        bz = len(input_ids)#batch_size
        bx = len(input_ids[0])
        prompt_tokens = [self.pseudo_token_id]
        counts = []
        queries = []
        for i in range(bz):
            query, count = self.get_query(input_ids[i], prompt_tokens)
            counts.append(count)
            queries.append(torch.LongTensor(query).squeeze(0))

        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask1 = queries != self.pad_token_id

        inputs_embeds = self.embed_input(queries, counts)
        #  直接在query上加LE (X)

        # todo 3 只在prompt上加LE
        inputs_embeds = self.add_label_embedding(inputs_embeds, label_init, counts)
        inputs = inputs_embeds.to(self.device)
        outputs = self.gpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
        # decode the output ids to see if there is some patterns
        outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
        example = torch.argsort(outputs2[0], dim=2, descending=True)[0, sum(self.template)+counts[0]+1:, 0]

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)

        for bdix in range(bz):
            if self.template[0] == self.template[1]:
                place = sum(self.template)+counts[bdix]# 45 = 6+6+32+1
            else:
                place = self.template[0] + counts[bdix]
            sequence[bdix, :counts[bdix], :] = sequence_output[bdix, place:place+counts[bdix], :]
            # todo 只截取没有pad的id对应的input


        # 1 在generate之后加LE (X)
        #sequence = self.add_label_embedding(sequence, label_init)

        logits = self.classifier(sequence)
        outputs = (example,)+outputs[2:]

        outputs = (logits,) + outputs # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.contiguous().view(-1, self.num_labels), labels.contiguous().view(-1))
            outputs = (loss,) + outputs

        return outputs



class GPT2generateForNer_LE(torch.nn.Module):
    """
    循环输出hidden state, 在每一步的output中加入label embedding
    """
    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        self.num_labels = config.num_labels
        if model_name == None:# 用于load gpt2-large
            model_name = 'gpt2'
        self.gpt2 = New_GPT2.from_pretrained(model_name)# 可以接受inputs_embeds和input_ids
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.embeddings = GPT2LMHeadModel.from_pretrained(model_name).base_model.get_input_embeddings()#embedding是GPT2LMHeadModel的embedding

        # self.embeddings.weight.requires_grad = False
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.loss_type = 'ce'
        self.device = device

        self.pseudo_token_id = 50257# prompt word 的id
        self.hidden_size = self.embeddings.embedding_dim
        self.template = template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)

        self.num_entities = 9# todo for conll2003 区分bio or bieso

        self.label_embedding = LabelEmbeder([self.num_entities], self.hidden_size, device)
        self.label_embedding = self.label_embedding.to(self.device)
        self.attn_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.linear_out = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.cat = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size*2, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.hidden_size))

        print("***************** init GPT2SoftmaxForNer with label embedding *********************")
        print("**************** generate hidden state in a loop ****************")
        print("***************** "+str(model_name) + "  *********************")
        print("************** num_labels *** "+str(self.num_labels) + "  *********************")

    def get_query(self, input_id, prompt_tokens):
        input = []
        prompt1 = []
        prompt2 = []
        count = 0
        for i in range(self.template[0]):
            prompt1.append(prompt_tokens[0])
        for i in range(self.template[1]):
            prompt2.append(prompt_tokens[0])

        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())
        query = prompt1 + input + prompt2
        return query, count

    def embed_input(self, queries, counts):
        """
        turn the queries(word index) :[batch_size,query_length]
        into embeddings: [batch_size,query_length,768]
        """
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1

        replace_embeds = self.prompt_encoder()
        raw_embeds = self.embeddings(queries_for_embedding)

        for bidx in range(bz):
            for i in range(self.template[0]):
                raw_embeds[bidx, i, :] = replace_embeds[i, :]
            for i in range(self.template[1]):
                raw_embeds[bidx, i+counts[bidx]+self.template[0], :] = replace_embeds[i+self.template[0], :]
        return raw_embeds


    def attention(self, input_state, label_init, bz):
        """
        Args:
            input_state: [batch_size, hidden state]
            label_embedding: [batch_size, num_label_type, hidden state]

        Returns:
            output_state:[batch_size, hidden state]
        """
        label_embedding = torch.empty(bz, self.num_entities, self.hidden_size).to(self.device)# [bz, 5, hs]
        for k in range(bz):
            label_embedding[k, :, :] = label_init

        input_state_attn = self.attn_linear(input_state).unsqueeze(2)
        input_state_attn = self.tanh(input_state_attn)# [bz, hs, 1]
        weights = torch.bmm(label_embedding, input_state_attn).squeeze(2)# [bz, 5, hs] * [bz, hs, 1] = [bz, 5] 表示五类label的分数
        weights = self.softmax(weights)

        c_t = torch.bmm(weights.unsqueeze(1), label_embedding).squeeze(1)# [bz, 1, 5] * [bz, 5, hs] = [bz, hs]  =  sigma(c_i*label_i)
        output = self.tanh(self.linear_out(torch.cat([c_t, input_state], 1)))

        return output
    # def old_attention(self, input_state, label_init, bz):
    #     """
    #     Args:
    #         input_state: [batch_size, hidden state]
    #         label_embedding: [batch_size, num_label_type, hidden state]
    #
    #     Returns:
    #         output_state:[batch_size, hidden state]
    #     """
    #     label_embedding = torch.empty(bz, self.num_entities, self.hidden_size).to(self.device)
    #     for k in range(bz):
    #         label_embedding[k, :, :] = label_init
    #
    #     input_state_attn = self.attn_linear(input_state)
    #     input_state_expanded = input_state_attn.unsqueeze(1).expand(bz, self.num_entities, self.hidden_size).contiguous()  # B x 5 x hidden_dim
    #     input_state_expanded = input_state_expanded.view(-1, self.hidden_size)     # B*5 x hidden_dim
    #
    #     label_embedding_fea = label_embedding.view(-1, self.hidden_size)
    #
    #     #
    #     att_features = label_embedding_fea + input_state_expanded    # B*self.num_entities x hidden_dim
    #     e = torch.tanh(att_features)
    #     scores = self.fc(e)                                      # B*self.num_entities x 1
    #     scores = scores.view(-1, self.num_entities)                              # B x self.num_entities
    #     attn_dist_ = F.softmax(scores, dim=1)                    # B x self.num_entities
    #     normalization_factor = attn_dist_.sum(1, keepdim=True)
    #     attn_dist = attn_dist_ / normalization_factor
    #     attn_dist = attn_dist.unsqueeze(1)                        # B x 1 x self.num_entities
    #     output_state = torch.bmm(attn_dist, label_embedding)      # B x 1 x self.num_entities   *   B x self.num_entities x hidden_dim
    #
    #     output_state = output_state.squeeze(1)
    #     output_state += input_state
    #     return output_state

    def add_label_embedding(self, sequence_output, label_init):
        """
        Args:
            sequence_output: the output hidden state from gpt2 model  [batch_size, 1, 768]
        Returns:
            output: add label embedding to sequence_output  [batch_size, 1, 768]
        """
        bz = sequence_output.shape[0]
        new_sequence_output = self.attention(sequence_output, label_init, bz)
        return new_sequence_output


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        """
        Args:
            input_ids: padded seuqence:[batch_size, max_length]
            if Chinese: input_ids = [101,...,102, 0,...,0]
            attention_mask: [batch_size, max_length]
            token_type_ids: [batch_size, max_length]
            position_ids: [batch_size, max_length]
            head_mask: [batch_size, max_length]
            labels: [batch_size, max_length]
        Returns:
            outputs
        """
        label_init = self.label_embedding()
        bz = len(input_ids)#batch_size
        prompt_tokens = [self.pseudo_token_id]
        counts = []
        queries = []
        for i in range(bz):
            query, count = self.get_query(input_ids[i], prompt_tokens)
            counts.append(count)
            queries.append(torch.LongTensor(query).squeeze(0))

        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask1 = queries != self.pad_token_id
        inputs_embeds = self.embed_input(queries, counts)
        inputs = inputs_embeds.to(self.device)
        outputs = self.gpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())

        outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
        example = torch.argsort(outputs2[0], dim=2, descending=True)[0, sum(self.template)+counts[0]+1:, 0]

        sequence_output = outputs[0][..., -1, :]# [batch_size, 768]
        past_key_values = outputs.past_key_values

        assert outputs[0][0][0][0] == outputs.last_hidden_state[0][0][0]
        sequence = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden_size).to(self.device)

        # 第一个token
        new_sequence_output = self.add_label_embedding(sequence_output, label_init)
        sequence[:, 0, :] = new_sequence_output
        # loop
        for round in range(1, max(counts)):
            input_this_step =  inputs[:, self.template[0]+round-1:self.template[0]+round, :]
            outputs = self.gpt2(inputs_embeds=input_this_step,
                                past_key_values=past_key_values, return_dict=None)
            sequence_output = outputs[0][..., -1, :]
            past_key_values = outputs.past_key_values

            new_sequence_output = self.add_label_embedding(sequence_output, label_init)
            sequence[:, round, :] = new_sequence_output

        sequence = self.dropout(sequence)
        logits = self.classifier(sequence)

        outputs = (example,)+outputs[2:]
        outputs = (logits,) + outputs# add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.contiguous().view(-1, self.num_labels), labels.contiguous().view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

