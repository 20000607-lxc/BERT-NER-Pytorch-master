import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from .transformers_master.models.bart.modeling_bart import BartModel as bart
from .transformers_master.models.bart.modeling_bart import BartPretrainedModel,  BartForCausalLM, BartForConditionalGeneration
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence

class BartSoftmaxForNer(BartPretrainedModel):

    def __init__(self, config, device, template):
        super(BartSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bart = bart.from_pretrained('facebook/bart-large')# 可以接受inputs_embeds和input_ids
        self.embeddings = BartForConditionalGeneration.from_pretrained('facebook/bart-large').get_input_embeddings()
        self.embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)# config.hidden_size=1024
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.loss_type = 'ce' #采用的新的 config 没有loss_type这一属性
        self.init_weights()

        self.pseudo_token_id = 50265# prompt word id

        self.hidden_size = config.hidden_size
        self.template = template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)


    def get_query(self, input_id, prompt_tokens):

        return [prompt_tokens * self.template[0] +
                [input_id[i] for i in range(len(input_id))] +  # head entity
                prompt_tokens * self.template[1]
                + [input_id[i] for i in range(len(input_id))]]


    def embed_input(self, queries):
        """
        turn the queries(word index) :[batch_size,query_length]
        into embeddings: [batch_size,query_length,768]
        """
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.pseudo_token_id-1

        replace_embeds = self.prompt_encoder()
        raw_embeds = self.embeddings(queries_for_embedding)

        l = len(queries[0])-(len(queries[0])-12)//2
        block_length1 = self.template[0]
        block_length2 = self.template[1]
        for bidx in range(bz):
            for i in range(block_length1):
                raw_embeds[bidx, i:i+1, :] = replace_embeds[i, :]
            for j in range(block_length2):
                raw_embeds[bidx, l-block_length2+j:l-block_length2+j+1, :] = replace_embeds[block_length1+j, :]
        return raw_embeds

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        bz = len(input_ids)#batch_size
        prompt_tokens = [self.pseudo_token_id]

        queries = [torch.LongTensor(self.get_query(input_ids[i], prompt_tokens)).squeeze(0) for i in range(bz)]

        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        attention_mask1 = queries != self.pad_token_id

        inputs_embeds = self.embed_input(queries)
        inputs = inputs_embeds.to(self.device)
        outputs = self.bart(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        l = len(sequence_output[0])
        block_length2 = self.template[1]
        sequence_output = sequence_output[:, l//2+block_length2-1:l-1, :]

        logits = self.classifier(sequence_output)#logits：每个词的labels分数

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
