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

class GPT2LMSoftmaxForNer(GPT2PreTrainedModel):

    """
    采用中文预训练gpt2 model的embedding weights, 由于许多参数与gpt2model不一致，因此单独写了该class
    """

    def __init__(self, config, device, template):
        super(GPT2LMSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.gpt2 = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").base_model
        # self.gpt2 = New_GPT2.from_pretrained('gpt2')

        # for param in self.gpt2.parameters():
        #     param.requires_grad = False
        self.pseudo_token_id = 21128
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.loss_type = 'ce'
        self.embeddings = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").base_model.get_input_embeddings()
        # embedding是GPT2LMHeadModel的embedding
        self.embeddings.weight.requires_grad = False
        self.hidden_size =  self.embeddings.embedding_dim
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)
        self.template = template
        self.init_weights()

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)
        print('init chinese_pretrained_gpt2 form "uer/gpt2-chinese-cluecorpussmall"')

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
        query = prompt1 + input + prompt2 + input
        return query, count


    def embed_input(self, queries, counts):
        """
        turn the queries(word index) :[batch_size,query_length]
        into embeddings: [batch_size,query_length,768]
        """
        bz = queries.shape[0]

        replace_embeds = self.prompt_encoder()
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1

        raw_embeds = self.embeddings(queries_for_embedding)

        for bidx in range(bz):
            for i in range(self.template[0]):
                raw_embeds[bidx, i, :] = replace_embeds[i, :]
            for i in range(self.template[1]):
                raw_embeds[bidx, i+counts[bidx]+self.template[0], :] = replace_embeds[i+self.template[0], :]

        return raw_embeds

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):

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

        #sequence_output = outputs.logits # gpt2model
        sequence_output = outputs.last_hidden_state# gpt2MLMHeadbasemodel


        sequence_output = self.dropout(sequence_output)
        sequence = torch.empty(input_ids.shape[0], input_ids.shape[1], self.hidden_size).to(self.device)

        for bdix in range(bz):
            sequence[bdix] = sequence_output[bdix, sum(self.template)+counts[bdix]:sum(self.template)+counts[bdix]+bx, :]

        logits = self.classifier(sequence)#logits：每个词的labels分数

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
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
