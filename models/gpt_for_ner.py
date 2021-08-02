import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from .transformers_master.models.gpt2.modeling_gpt2 import GPT2Model as New_GPT2
from .transformers_master.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from processors.utils_ner import CNerTokenizer
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .layers.model.lstmcrf import NNCRF
import copy

class GPT2SoftmaxForNer(GPT2PreTrainedModel):
    """
    输出input[1:] + prompt3 对应的hidden state
    tokenizer: bert-base-chinese
    """
    def __init__(self, config, device, template, model_name=None):
        super(GPT2SoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        if model_name == None:
            model_name = 'gpt2'
        self.gpt2 = New_GPT2.from_pretrained(model_name)# 可以接受inputs_embeds和input_ids
        self.embeddings = GPT2LMHeadModel.from_pretrained(model_name).base_model.get_input_embeddings()#embedding是GPT2LMHeadModel的embedding
        self.embeddings.weight.requires_grad = False
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False
        # perform fine_tuning

        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.loss_type = 'ce'
        self.init_weights()

        self.pseudo_token_id = 50257# prompt word 的id

        self.hidden_size = self.embeddings.embedding_dim
        self.template = template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)
        print("init  GPT2SoftmaxForNer")

    def get_query(self, input_id, prompt_tokens):
        input = []
        prompt1 = []
        prompt2 = []
        prompt3 = []
        count = 0
        for i in range(self.template[0]):
            prompt1.append(prompt_tokens[0])
        for i in range(self.template[1]):
            prompt2.append(prompt_tokens[0])

        prompt3.append(prompt_tokens[0])

        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())
        query = prompt1 + input + prompt2 + input + prompt3# prompt3 一位

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
            raw_embeds[bidx, i+1+counts[bidx]+self.template[0], :] = replace_embeds[i+1+self.template[0], :]
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

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)

        for bdix in range(bz):
            place = sum(self.template)+counts[bdix]+1# 45 = 6+6+32+1
            sequence[bdix, :counts[bdix], :] = sequence_output[bdix, place:place+counts[bdix], :]
            # todo 只截取没有pad的id对应的input

        logits = self.classifier(sequence)#logits：每个词的labels分数

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
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



# class GPT2GenerateForNer(GPT2PreTrainedModel):
#     """
#     循环生成下一位的hidden state
#     """
#
#     def __init__(self, config, device, template):
#         super(GPT2GenerateForNer, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.gpt2 = New_GPT2.from_pretrained('gpt2')# 可以接受inputs_embeds和input_ids
#         self.embeddings = GPT2LMHeadModel.from_pretrained('gpt2').base_model.get_input_embeddings()#embedding是GPT2LMHeadModel的embedding
#         self.embeddings.weight.requires_grad = False
#         # for param in self.gpt2.parameters():
#         #     param.requires_grad = False
#         # perform fine_tuning
#         self.dropout = nn.Dropout(config.resid_pdrop)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
#         self.loss_type = 'ce'
#         self.init_weights()
#
#         self.pseudo_token_id = 50257# prompt word 的id
#
#         self.hidden_size = self.embeddings.embedding_dim
#         self.template = template
#
#         self.pad_token_id = 0
#         self.spell_length = sum(self.template)
#         self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
#         self.prompt_encoder = self.prompt_encoder.to(device)
#         print("init  GPT2GenerateForNer")
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
#         for i in range(len(input_id)):
#             if input_id[i] != 0:
#                 count += 1
#                 input.append(input_id[i].item())
#         query = prompt1 + input + prompt2 # prompt3 一位
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
#         return raw_embeds
#
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
#         """
#         Args:
#             input_ids: padded seuqence:[batch_size, max_length]
#             if Chinese: input_ids = [101,...,102, 0,...,0]
#             attention_mask: [batch_size, max_length]
#             token_type_ids: [batch_size, max_length]
#             position_ids: [batch_size, max_length]
#             head_mask: [batch_size, max_length]
#             labels: [batch_size, max_length]
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
#
#         sequence_output = outputs[0][..., -1, :]# [batch_size, 768]
#         sequence_output = self.dropout(sequence_output)
#         past_key_values = outputs.past_key_values
#
#         sequence = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden_size).to(self.device)
#
#         for bdix in range(bz):
#             sequence[bdix, 0, :] = sequence_output[bdix, :]
#
#         for round in range(1, max(counts)):# 1 ...  19
#             # todo 没有label的(pad token) 应该不会影响计算loss和准确性吧？需要改成32吗？
#             # 另外就是可可能有的batch已经到头了，有的还没有，不同的batch之间应该不会相互影响吧？？？？
#             sequence_output = sequence_output.unsqueeze(1)
#             outputs = self.gpt2(inputs_embeds=sequence_output, past_key_values=past_key_values, return_dict=None)
#             sequence_output = outputs.last_hidden_state[..., -1, :]
#             # todo 采用last_hidden_state对吗？
#             past_key_values = outputs.past_key_values
#             sequence[:, round, :] = sequence_output[:, :]
#
#         logits = self.classifier(sequence)#logits：每个词的labels分数
#
#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#         if labels is not None:
#             assert self.loss_type in ['lsr', 'focal', 'ce']
#             if self.loss_type == 'lsr':
#                 loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
#             elif self.loss_type == 'focal':
#                 loss_fct = FocalLoss(ignore_index=0)
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


class GPT2CrfForNer(GPT2PreTrainedModel):
    def __init__(self, config, device, template):
        super(GPT2CrfForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.gpt2 = New_GPT2.from_pretrained('gpt2')
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = 'ce'
        self.embeddings = GPT2LMHeadModel.from_pretrained('gpt2').base_model.get_input_embeddings()
        self.embeddings.weight.requires_grad = False
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.lstmcrf = NNCRF(config=config, device=device, num_tags=config.num_labels, batch_first=True)
        self.init_weights()

        self.pseudo_token_id = 50257# prompt word的id

        self.hidden_size = self.embeddings.embedding_dim
        self.template = template
        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)
        self.lstm = torch.nn.LSTM(input_size=self.hidden_size,
                                  hidden_size=self.hidden_size // 2,
                                  num_layers=2,
                                  dropout=0.0,
                                  bidirectional=True,
                                  batch_first=True)

        print("init GPT2CrfForNer ")

    def get_query(self, input_id, prompt_tokens, Bi_lstm=False, lstm=False):

        # reversed = False
        # if reversed:
        #     reversed_sequence = [input_id[len(input_id)-1-i] for i in range(len(input_id))]
        #     return [prompt_tokens * self.template[0] +
        #             reversed_sequence +
        #             prompt_tokens * self.template[1] +
        #             [input_id[i] for i in range(len(input_id))]
        #             ]
        # elif Bi_lstm:
        #     return [ prompt_tokens * self.template[0]
        #             + prompt_tokens * len(input_id)
        #             + prompt_tokens * self.template[1]
        #             + [input_id[i] for i in range(len(input_id))]
        #             ]
        # elif lstm:
        #     return [ prompt_tokens * self.template[0]
        #              + [input_id[i] for i in range(len(input_id))]
        #              + prompt_tokens * self.template[1]
        #              + [input_id[i] for i in range(len(input_id))]
        #              ]
        # else:
        input = []
        prompt1 = []
        prompt2 =[]
        count = 0
        for i in range(self.template[0]):
            prompt1.append(prompt_tokens[0])
        for i in range(self.template[1]):
            prompt2.append(prompt_tokens[0])

        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())

        query = prompt1 + input + prompt2 + input
        return query, count

    def embed_input(self, queries, input_embeds, Bi_lstm=False, lstm=False, counts=None):
        """
        turn the queries(word index) :[batch_size,query_length]
        into embeddings: [batch_size,query_length,768]
        """
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1
        # if Bi_lstm:
        #     sequence_length = input_embeds.shape[1]
        #     raw_embeds = torch.zeros(queries.shape[0],queries.shape[1], 768)
        #     prompt_length =raw_embeds.shape[1]-sequence_length
        #     for j in range(bz):
        #         replace_embeds = self.prompt_encoder(raw_embeds[j, prompt_length:, :])
        #         raw_embeds[j] = self.embeddings(queries_for_embedding[j])
        #         block_length = prompt_length
        #         for i in range(block_length):
        #             raw_embeds[j, i:i+1, :] = replace_embeds[i, :]
        #     return raw_embeds
        #
        # elif lstm:
        #     replace_embeds = self.prompt_encoder()
        #     raw_embeds = self.embeddings(queries_for_embedding)
        #     input = copy.deepcopy(raw_embeds)
        #     l = len(queries[0])-(len(queries[0])-12)//2
        #     block_length1 = self.template[0]
        #     block_length2 = self.template[1]
        #     for bidx in range(bz):
        #         for i in range(block_length1):
        #             raw_embeds[bidx, i:i+1, :] = replace_embeds[i, :]
        #         for j in range(block_length2):
        #             raw_embeds[bidx, l-block_length2+j:l-block_length2+j+1, :] = replace_embeds[block_length1+j, :]
        #         output, _ = self.lstm(input[bidx, block_length1:l-block_length1, :].unsqueeze(0))#lstm的输入dimension = 3
        #         raw_embeds[bidx, block_length1:l-block_length1, :] = output.squeeze(0)
        #         # 记得要采用deepcopy 否则会有以下错误（在loss.backward()时）
        #         # RuntimeError: one of the variables needed for gradient computation has been modified
        #         # by an inplace operation: [torch.cuda.FloatTensor [1, 27, 768]], which is output 0 of
        #         # UnsqueezeBackward0, is at version 104; expected version 103 instead. Hint: enable
        #         # anomaly detection to find the operation that failed to compute its gradient, with
        #         # torch.autograd.set_detect_anomaly(True).
        #
        #     return raw_embeds
        #
        # else:
        replace_embeds = self.prompt_encoder()
        raw_embeds = self.embeddings(queries_for_embedding)
        for bidx in range(bz):
            for i in range(self.template[0]):
                raw_embeds[bidx, i, :] = replace_embeds[i, :]
            for i in range(self.template[1]):
                raw_embeds[bidx, i+counts[bidx]+self.template[0], :] = replace_embeds[i+self.template[0], :]

        return raw_embeds

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, input_lens=None):

        bz = len(input_ids)#batch_size
        bx = len(input_ids[0])
        prompt_tokens = [self.pseudo_token_id]

        Bi_lstm = False
        lstm = False


        counts = []
        queries = []
        for i in range(bz):
            query, count = self.get_query(input_ids[i], prompt_tokens, Bi_lstm, lstm)
            counts.append(count)
            queries.append(torch.LongTensor(query).squeeze(0))

        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask1 = queries != self.pad_token_id

        inputs_embeds = self.embed_input(queries, input_ids, Bi_lstm, lstm, counts)
        inputs = inputs_embeds.to(self.device)
        outputs = self.gpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence = torch.empty(input_ids.shape[0], input_ids.shape[1], self.hidden_size).to(self.device)
        for bdix in range(bz):
            sequence[bdix] = sequence_output[bdix, sum(self.template)+counts[bdix]:sum(self.template)+counts[bdix]+bx, :]
        logits = self.classifier(sequence)#logits：每个词的labels分数

        if labels is not None:
            word_seq_length = torch.LongTensor([sum(attention_mask[i]).item() for i in range(len(attention_mask))])
            word_seq_length = word_seq_length.unsqueeze(dim=0)
            word_seq_length = word_seq_length.transpose(0, 1)
            word_seq_length = word_seq_length.squeeze(dim=1)#shape = batch_size, 没有1

            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)#crf的作用即为计算loss
            # loss = self.lstmcrf(word_embeds=sequence_output, word_seq_length=word_seq_length,
            #                     #emissions=logits,
            #                     tags=labels, mask=attention_mask)

            # def forward(self, emissions: torch.Tensor, tags: torch.LongTensor,
            # mask:Optional[torch.ByteTensor] = None, reduction: str = 'mean') #mask是optional的,
            # -> torch.Tensor:loss
            # mask的作用：在CRF中做了分母

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            outputs = (-1*loss,)+outputs
        return outputs, word_seq_length, sequence_output # (loss), scores
