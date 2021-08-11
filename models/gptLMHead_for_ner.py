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
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .layers.model.lstmcrf import NNCRF
import copy

class GPT2LMSoftmaxForNer(torch.nn.Module):

    """
    采用中文预训练gpt2 model的embedding weights, 由于许多参数与gpt2model不一致，因此单独写了该class
    """

    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        self.num_labels = config.num_labels
        self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").base_model# 21128 768 donnt use it anymore!!!
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

        #self.gpt2 = New_GPT2.from_pretrained('gpt2') #50257 768  this model is much better !!!

        self.embeddings = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").base_model.get_input_embeddings()# 21128 768
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False

        self.pseudo_token_id = 21128
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.loss_type = 'ce'
        self.device = device

        # embedding是GPT2LMHeadModel的embedding
        # self.embeddings.weight.requires_grad = False
        self.hidden_size =  self.embeddings.embedding_dim
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)
        self.template = template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)

        print('******************init chinese_pretrained_gpt2 form "uer/gpt2-chinese-cluecorpussmall"*********************')

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

        # sequence_output = outputs[0]## gpt2model
        sequence_output = outputs.last_hidden_state# gpt2MLMHeadbasemodel # gpt2model
        # the example of the output words of batch[0]
        outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())

        # todo example is computed by GPT2LMhead model
        example = torch.argsort(outputs2[0], dim=2, descending=True)[0, sum(self.template)+counts[0]+1:, 0]
        sequence_output = self.dropout(sequence_output)
        sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)

        for bdix in range(bz):
            place = sum(self.template)+counts[bdix]+1# 45 = 6+6+32+1
            sequence[bdix, :counts[bdix], :] = sequence_output[bdix, place:place+counts[bdix], :]
            # todo 只截取没有pad的id对应的input

        logits = self.classifier(sequence)#logits：每个词的labels分数
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

        return outputs  # (loss), scores, (hidden_states), (attentions)





class BareChineseGPT2(torch.nn.Module):

    """
    采用中文预训练gpt2 model的embedding weights, 由于许多参数与gpt2model不一致，因此单独写了该class
    """

    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        self.num_labels = config.num_labels
        self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").base_model# 21128 768 donnt use it anymore!!!
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

        #self.gpt2 = New_GPT2.from_pretrained('gpt2') #50257 768  this model is much better !!!

        self.embeddings = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").base_model.get_input_embeddings()# 21128 768
        # for param in self.gpt2.parameters():
        #     param.requires_grad = False

        self.pseudo_token_id = 21128
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.loss_type = 'ce'
        self.device = device

        # embedding是GPT2LMHeadModel的embedding
        # self.embeddings.weight.requires_grad = False
        self.hidden_size =  self.embeddings.embedding_dim
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)
        self.template = template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)

        print('****************** init BARE chinese_pretrained_gpt2 form "uer/gpt2-chinese-cluecorpussmall"*********************')

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

        # sequence_output = outputs[0]## gpt2model
        sequence_output = outputs.last_hidden_state# gpt2MLMHeadbasemodel # gpt2model
        # the example of the output words of batch[0]
        outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())

        # todo example is computed by GPT2LMhead model
        example = torch.argsort(outputs2[0], dim=2, descending=True)[0, sum(self.template)+counts[0]+1:, 0]
        sequence_output = self.dropout(sequence_output)
        sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)

        for bdix in range(bz):
            place = counts[bdix]+1# 45 = 6+6+32+1
            sequence[bdix, :counts[bdix], :] = sequence_output[bdix, place:place+counts[bdix], :]
            # todo 只截取没有pad的id对应的input

        logits = self.classifier(sequence)#logits：每个词的labels分数
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

        return outputs  # (loss), scores, (hidden_states), (attentions)