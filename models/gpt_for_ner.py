import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from .transformers_master.models.gpt2.modeling_gpt2 import GPT2Model as New_GPT2
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel

class GPT2SoftmaxForNer_fix(torch.nn.Module):
    """
    输出input对应的hidden state
    tokenizer: bert-base-chinese or gpt2 tokenizer
    """
    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        self.num_labels = config.num_labels
        if model_name == None:
            model_name = 'gpt2'
        self.device = device
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

        self.gpt2 = self.LMgpt2.base_model# New_GPT2.from_pretrained(model_name).to(self.device)# 可以接受inputs_embeds和input_ids
        self.embeddings = self.gpt2.get_input_embeddings().to(device)#embedding是GPT2LMHeadModel的embedding


        # self.embeddings.weight.requires_grad = False
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False
        # perform fine_tuning

        self.dropout = nn.Dropout(config.resid_pdrop).to(self.device)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(self.device)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size).to(self.device)
        self.loss_type = 'ce'

        self.pseudo_token_id = 50257# prompt word 的id
        self.hidden_size = self.embeddings.embedding_dim
        self.template = template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)
        print("***************** init GPT2SoftmaxForNer *********************")
        print("***************** "+str(model_name) + " *********************")
        print("************** num_labels *** " + str(self.num_labels) + " *********************")

    def get_query(self, input_id, prompt_tokens):
        input = []
        prompt1 = []
        prompt2 = []
        #prompt3 = []
        count = 0
        for i in range(self.template[0]):
            prompt1.append(prompt_tokens[0])
        for i in range(self.template[1]):
            prompt2.append(prompt_tokens[0])

        #prompt3.append(prompt_tokens[0])
        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())
        query = prompt1 + input + prompt2 + input
        # if self.template[0] == self.template[1]:
        #     query = prompt1 + input + prompt2 + input
        # else:
        #     query = prompt1 + input + prompt2

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
            # 加入最后一位
            # raw_embeds[bidx, i+1+counts[bidx]+self.template[0], :] = replace_embeds[i+1+self.template[0], :]
        return raw_embeds

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
        inputs = inputs_embeds.to(self.device)
        outputs = self.gpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())

        # decode the output ids to see if there is some strange patterns
        outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())

        example = torch.argsort(outputs2[0], dim=2, descending=True)[:, sum(self.template)+max(counts)-1:, 0].to(self.device)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)

        for bdix in range(bz):
            # example:
            # inputs = [p,p,2,2,2,p,2,2,2]
            # outputs= 后一截[2,2,2]对应的hidden state

            # todo shift left 1
            place = sum(self.template)+counts[bdix]-1
            sequence[bdix, :counts[bdix], :] = sequence_output[bdix, place:place+counts[bdix], :]
            # todo 只截取没有pad的id对应的input

        logits = self.classifier(sequence)#logits：每个词的labels分数
        outputs = (example,)+outputs[2:]

        outputs = (logits,) + outputs # add hidden states and attention if they are here
        if labels is not None:
            # 所有loss的默认ignore index 都为-100
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
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




class GPT2GenerateForNer(torch.nn.Module):
    """
    循环生成下一位的hidden state
    """
    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        if model_name == None:
            model_name = 'gpt2'
        self.device = device
        self.num_labels = config.num_labels
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

        self.gpt2 = self.LMgpt2.base_model# New_GPT2.from_pretrained(model_name).to(self.device)# 可以接受inputs_embeds和input_ids
        self.embeddings = self.gpt2.get_input_embeddings().to(device)#embedding是GPT2LMHeadModel的embedding


        # self.embeddings.weight.requires_grad = False
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False
        # perform fine_tuning

        self.dropout = nn.Dropout(config.resid_pdrop).to(self.device)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(self.device)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size).to(self.device)
        self.loss_type = 'ce'
        self.pseudo_token_id = 50257# prompt word 的id

        self.hidden_size = self.embeddings.embedding_dim
        self.template = template
        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)
        self.cat = nn.Linear(config.hidden_size*2, config.hidden_size).to(self.device)
        # todo 加一个激活层看会不会好
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size*2, self.hidden_size),
                                                 nn.ReLU(),
                                                 nn.Linear(self.hidden_size, self.hidden_size)).to(self.device)

        print("****************init  GPT2GenerateForNer  ***********************")
        print("****************generate hidden state in a loop****************")
        print("***************** "+str(model_name) + " *********************")
        print("************** num_labels *** "+str(self.num_labels) + " *********************")

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
        example = torch.argsort(outputs2[0], dim=2, descending=True)[:, sum(self.template)+max(counts):, 0].to(self.device)

        sequence_output = outputs[0][..., -1, :]# [batch_size, 768]
        past_key_values = outputs.past_key_values

        assert outputs[0][0][0][0] == outputs.last_hidden_state[0][0][0]
        sequence = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden_size).to(self.device)

        # 第一个token
        sequence[:, 0, :] = sequence_output

        for round in range(1, max(counts)):
            # choice1: inputs[:, self.template[0]+round-1:self.template[0]+round, :]
            # choice2: inputs[:, self.template[0]+round:self.template[0]+round+1, :]

            # choice3: (X) inputs[:, self.template[0]+round-1:self.template[0]+round, :] + freeze past_key_values
            # choice3: (X) sequence_output.unsqueeze(1)
            # choice4: (X) self.cat(torch.cat((inputs[:, self.template[0]+round:self.template[0]+round+1, :], inputs[:, self.template[0]+round-1:self.template[0]+round, :]),dim=2))
            # choice5: (X)  self.mlp(torch.cat((inputs[:, self.template[0]+round:self.template[0]+round+1, :], inputs[:, self.template[0]+round-1:self.template[0]+round, :]),dim=2))

            input_this_step = inputs[:, self.template[0]+round-1:self.template[0]+round, :]
            outputs = self.gpt2(inputs_embeds=input_this_step,
                                past_key_values=past_key_values, return_dict=None)
            sequence_output = outputs[0][..., -1, :]

            past_key_values = outputs.past_key_values

            sequence[:, round, :] = sequence_output

        sequence = self.dropout(sequence)
        logits = self.classifier(sequence)

        outputs = (example,)+outputs[2:]
        outputs = (logits,) + outputs# add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
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


class BareGPT2(torch.nn.Module):
    """
    输出input对应的hidden state
    tokenizer: bert-base-chinese
    """
    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        self.num_labels = config.num_labels
        self.device = device
        if model_name == None:
            model_name = 'gpt2'
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

        self.gpt2 = self.LMgpt2.base_model# New_GPT2.from_pretrained(model_name).to(self.device)# 可以接受inputs_embeds和input_ids
        self.embeddings = self.gpt2.get_input_embeddings().to(device)#embedding是GPT2LMHeadModel的embedding

        #self.embeddings.weight.requires_grad = False
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False
        # perform fine_tuning
        self.dropout = nn.Dropout(config.resid_pdrop).to(self.device)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(self.device)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size).to(self.device)
        self.loss_type = 'ce'
        self.pseudo_token_id = 50257# prompt word 的id
        self.hidden_size = self.embeddings.embedding_dim
        self.pad_token_id = 0
        print("************************init BareGPT2 **************************")

    def get_query(self, input_id, prompt_tokens):
        input = []
        count = 0
        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())
        query = input
        return query, count

    def embed_input(self, queries, counts):
        """
        turn the queries(word index) :[batch_size,query_length]
        into embeddings: [batch_size,query_length,768]
        """
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1
        raw_embeds = self.embeddings(queries_for_embedding)
        return raw_embeds

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
        inputs = inputs_embeds.to(self.device)
        outputs = self.gpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
        # decode the output ids to see if there is some patterns
        outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
        example = torch.argsort(outputs2[0], dim=2, descending=True)[:, sum(self.template)+max(counts):, 0].to(self.device)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)#logits：每个词的labels分数
        outputs = (example,)+outputs[2:]

        outputs = (logits,) + outputs # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
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


# class GPT2SoftmaxForNer(GPT2PreTrainedModel):
#     """
#     输出input[1:] + prompt3 对应的hidden state
#     tokenizer: bert-base-chinese
#     """
#     def __init__(self, config, device, template, model_name=None):
#         super(GPT2SoftmaxForNer, self).__init__(config)
#         self.num_labels = config.num_labels
#         if model_name == None:
#             model_name = 'gpt2'
#         self.gpt2 = New_GPT2.from_pretrained(model_name)# 可以接受inputs_embeds和input_ids
#         self.LMgpt2 = GPT2LMHeadModel.from_pretrained(model_name)
#
#         self.embeddings = GPT2LMHeadModel.from_pretrained(model_name).base_model.get_input_embeddings()#embedding是GPT2LMHeadModel的embedding
#         #self.embeddings.weight.requires_grad = False
#         # for param in self.gpt2.parameters():
#         #     param.requires_grad = False
#         # perform fine_tuning
#
#         self.dropout = nn.Dropout(config.resid_pdrop)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
#         self.loss_type = 'ce'
#         self.init_weights()
#
#         self.pseudo_token_id = 50257# prompt word 的id
#         self.hidden_size = self.embeddings.embedding_dim
#         self.template = template
#
#         self.pad_token_id = 0
#         self.spell_length = sum(self.template)
#         self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
#         self.prompt_encoder = self.prompt_encoder.to(device)
#         print("init  GPT2SoftmaxForNer")
#
#     def get_query(self, input_id, prompt_tokens):
#         input = []
#         prompt1 = []
#         prompt2 = []
#         prompt3 = []
#         count = 0
#         for i in range(self.template[0]):
#             prompt1.append(prompt_tokens[0])
#         for i in range(self.template[1]):
#             prompt2.append(prompt_tokens[0])
#
#         prompt3.append(prompt_tokens[0])
#         for i in range(len(input_id)):
#             if input_id[i] != 0:
#                 count += 1
#                 input.append(input_id[i].item())
#         query = prompt1 + input + prompt2 + input + prompt3# prompt3 一位
#         #query = prompt1 + input + prompt2 + prompt3
#
#         return query, count
#
#     def embed_input(self, queries, counts):
#         """
#         turn the queries(word index) :[batch_size,query_length]
#         into embeddings: [batch_size,query_length,768]
#         """
#         bz = queries.shape[0]
#         queries_for_embedding = queries.clone()
#         queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1
#
#         replace_embeds = self.prompt_encoder()
#         raw_embeds = self.embeddings(queries_for_embedding)
#
#         for bidx in range(bz):
#             for i in range(self.template[0]):
#                 raw_embeds[bidx, i, :] = replace_embeds[i, :]
#             for i in range(self.template[1]):
#                 raw_embeds[bidx, i+counts[bidx]+self.template[0], :] = replace_embeds[i+self.template[0], :]
#
#             # 加入最后一位
#             raw_embeds[bidx, i+1+counts[bidx]+self.template[0], :] = replace_embeds[i+1+self.template[0], :]
#         return raw_embeds
#
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
#         """
#
#         Args:
#             input_ids: padded seuqence:[batch_size, max_length]
#             if Chinese: input_ids = [101,...,102, 0,...,0]
#             attention_mask: [batch_size, max_length]
#             token_type_ids: [batch_size, max_length]
#             position_ids: [batch_size, max_length]
#             head_mask: [batch_size, max_length]
#             labels: [batch_size, max_length]
#
#         Returns:
#             outputs
#
#         """
#         bz = len(input_ids)#batch_size
#         bx = len(input_ids[0])
#         prompt_tokens = [self.pseudo_token_id]
#         counts = []
#         queries = []
#         for i in range(bz):
#             query, count = self.get_query(input_ids[i], prompt_tokens)
#             counts.append(count)
#             queries.append(torch.LongTensor(query).squeeze(0))
#
#         queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
#         attention_mask1 = queries != self.pad_token_id
#
#         inputs_embeds = self.embed_input(queries, counts)
#         inputs = inputs_embeds.to(self.device)
#         outputs = self.gpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
#         # decode the output ids to see if there is some patterns
#         outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
#         example = torch.argsort(outputs2[0], dim=2, descending=True)[0, sum(self.template)+counts[0]+1:, 0]
#
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
#         sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)
#
#         for bdix in range(bz):
#             place = sum(self.template)+counts[bdix]+1# 45 = 6+6+32+1
#
#             #place2 = self.template[0] + counts[bdix] + 1
#             sequence[bdix, :counts[bdix], :] = sequence_output[bdix, place:place+counts[bdix], :]
#
#
#         logits = self.classifier(sequence)#logits：每个词的labels分数
#         outputs = (example,)+outputs[2:]
#
#         outputs = (logits,) + outputs # add hidden states and attention if they are here
#         if labels is not None:
#             assert self.loss_type in ['lsr', 'focal', 'ce']
#             if self.loss_type == 'lsr':
#                 loss_fct = LabelSmoothingCrossEntropy()
#             elif self.loss_type == 'focal':
#                 loss_fct = FocalLoss()
#             else:
#                 loss_fct = CrossEntropyLoss()
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.contiguous().view(-1) == 1
#                 active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
#                 active_labels = labels.contiguous().view(-1)[active_loss]
#                 loss = loss_fct(active_logits, active_labels)
#             else:
#                 loss = loss_fct(logits.contiguous().view(-1, self.num_labels), labels.contiguous().view(-1))
#             outputs = (loss,) + outputs
#
#         return outputs  # (loss), scores, (hidden_states), (attentions)