import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel

class GPT2SoftmaxForNer_filling_entity(torch.nn.Module):
    """
    输出input对应的hidden state
    tokenizer: bert-base-chinese or gpt2 tokenizer
    """
    def __init__(self, config, device, template, model_name=None):
        super().__init__()


        self.only_filling_entity = True
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
        print("***************** init GPT2SoftmaxForNer train also with the input ids as label *********************")
        print("***************** "+str(model_name) + " *********************")
        print("************** num_labels *** " + str(self.num_labels) + " *********************")

    def get_query(self, input_id, prompt_tokens, removed_input_ids):
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
        query = prompt1 + input + prompt2 + removed_input_ids

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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, removed_input_ids=None):
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
            # todo choice:
            #  (1) eval 的时候也是用removed input ids , test 的时候给input ids
            #  (2) 一直用removed

            # if labels == None:
            #     query, count = self.get_query(input_ids[i], prompt_tokens, input_ids[i])
            # else:


            query, count = self.get_query(input_ids[i], prompt_tokens, removed_input_ids[i])
            counts.append(count)
            queries.append(torch.LongTensor(query).squeeze(0))

        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        attention_mask1 = queries != self.pad_token_id

        inputs_embeds = self.embed_input(queries, counts)
        inputs = inputs_embeds.to(self.device)
        outputs = self.gpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())

        # decode the output ids to see if there is some strange patterns
        outputs2 = self.LMgpt2.lm_head(outputs[0])

        example = torch.argsort(outputs2, dim=2, descending=True)[:, sum(self.template)+max(counts)-1:, 0].to(self.device)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)
        word_logits = torch.zeros(bz, bx, 50257).to(self.device)

        for bdix in range(bz):
            # example:
            # inputs = [p,p,2,2,2,p,2,2,2]
            # outputs= 后一截[2,2,2]对应的hidden state

            # todo shift left 1 and 只截取没有pad的id对应的input
            place = sum(self.template)+counts[bdix]-1
            sequence[bdix, :counts[bdix], :]    = sequence_output[bdix, place:place+counts[bdix], :]
            word_logits[bdix, :counts[bdix], :] = outputs2[bdix, place:place+counts[bdix], :]


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
                loss1 = loss_fct(active_logits, active_labels)

                if self.only_filling_entity:
                    attention_mask_entity = input_ids != removed_input_ids
                    # attention_mask_entity = attention_mask_entity.contiguous().view(-1) == 1
                    active_logits2 = word_logits.contiguous().view(-1, 50257)[attention_mask_entity]
                    active_inputs = input_ids.contiguous().view(-1)[attention_mask_entity]
                    loss2 = loss_fct(active_logits2, active_inputs)

                else:
                    active_logits2 = word_logits.contiguous().view(-1, 50257)[active_loss]
                    active_inputs = input_ids.contiguous().view(-1)[active_loss]
                    loss2 = loss_fct(active_logits2, active_inputs)

                loss = loss1 + loss2
            else:
                loss = loss_fct(logits.contiguous().view(-1, self.num_labels), labels.contiguous().view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)