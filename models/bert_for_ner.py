import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .transformers_master.models.bert.modeling_bert import BertPreTrainedModel, BertLMHeadModel
from .transformers_master.models.bert.modeling_bert import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence

class BertSoftmaxForNer(BertPreTrainedModel):

    def __init__(self, config, device, template, no_fine_tune):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        if no_fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("do not fine tune! ")

        self.LMBert = BertLMHeadModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = 'ce'# config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask.to(self.device).half())# token_type_ids=token_type_ids 全都是0 不要他了
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs2 = self.LMBert(input_ids=input_ids, attention_mask=attention_mask.to(self.device).half())
        example = torch.argsort(outputs2[0], dim=2, descending=True)[:, :, 0]
        outputs = (example,)+outputs[2:]

        outputs = (logits,) + outputs  # add hidden states and attention if they are here


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
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)



class BertPromptForNer(BertPreTrainedModel):
    """
    验证prompt 是否对bert有效果
    """
    def __init__(self, config, device, template):
        super( BertPromptForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config).to(device)
        self.embeddings = self.bert.get_input_embeddings().to(device)
        # self.embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob).to(device)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(device)
        self.loss_type = 'ce'#config.loss_type
        self.init_weights()

        self.pseudo_token_id = 2

        self.hidden_size =  self.embeddings.embedding_dim
        self.template = template#自定义一种template

        self.pad_token_id = 0
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, device)
        self.prompt_encoder = self.prompt_encoder.to(device)



    def get_query(self, input_id, prompt_tokens):
        input = []
        prompt = []
        count = 0
        for i in range(self.template[0]):
            prompt.append(prompt_tokens[0])
        for i in range(len(input_id)):
            if input_id[i] != 0:
                count += 1
                input.append(input_id[i].item())

        query = prompt + input + prompt + input
        return query, count

    def embed_input(self, queries, counts):
        """
        args:
        queries:[batch_size, sequence_length]
        counts:lengths of input_id for each seuqnece in this batch

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

        attention_mask1_queries = queries != self.pad_token_id

        attention_mask1_queries = attention_mask1_queries.to(self.device)

        inputs_embeds = self.embed_input(queries, counts)
        inputs = inputs_embeds.to(self.device)

        outputs = self.bert(inputs_embeds=inputs, attention_mask=attention_mask1_queries.to(self.device).half())# token_type_ids=token_type_ids 全都是0 不要他了
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        sequence = torch.empty(input_ids.shape[0], input_ids.shape[1], self.hidden_size).to(self.device)

        for bdix in range(bz):
            sequence[bdix] = sequence_output[bdix, sum(self.template)+counts[bdix]:sum(self.template)+counts[bdix]+bx, :]

        logits = self.classifier(sequence) # logits：每个词的label分数
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
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
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, device = None, template = None):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)#crf的作用即为计算loss
            # def forward(self, emissions: torch.Tensor, tags: torch.LongTensor,
            # mask:Optional[torch.ByteTensor] = None, reduction: str = 'mean') #mask是optional的,
            # -> torch.Tensor:loss
            # mask的作用：在CRF中做了分母
            # 从logits中可以取h(y[i];X)
            # GPT->classifier之后得到的logits可以表示h(y[i];x[1],...,x[i])？
            outputs = (-1*loss,)+outputs
        return outputs# (loss), scores

class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config,device=None, template=None):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)#在output的sequence里寻找start position
        #hidden state --> classes 应该是标记出一个词属于每一个class的分数，不过如何选择每个class的起点呢？
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)#直接采用nn.softmax
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)#输入start position之后找position
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type =='lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)

            active_loss = attention_mask.view(-1) == 1

            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs

