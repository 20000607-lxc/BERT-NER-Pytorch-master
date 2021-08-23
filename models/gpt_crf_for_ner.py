import torch
import torch.nn as nn
from .layers.crf import CRF
from .transformers_master.models.gpt2.modeling_gpt2 import GPT2Model as New_GPT2
from models.p_tuning.prompt_encoder import PromptEncoder
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel
from .layers.model.lstmcrf import NNCRF

class GPT2CrfForNer(torch.nn.Module):
    """
    输出input[1:] + prompt3 对应的hidden state
    tokenizer: bert-base-chinese or gpt2 tokenizer
    """
    def __init__(self, config, device, template, model_name=None):
        super().__init__()
        self.num_labels = config.num_labels
        if model_name == None:
            model_name = 'gpt2'
        self.gpt2 = New_GPT2.from_pretrained(model_name)# 可以接受inputs_embeds和input_ids
        self.LMgpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = 'ce'
        self.device = device
        self.embeddings = GPT2LMHeadModel.from_pretrained('gpt2').base_model.get_input_embeddings()

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.lstmcrf = NNCRF(config=config, device=device, num_tags=config.num_labels, batch_first=True)

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
        if self.template[0] == self.template[1]:
            query = prompt1 + input + prompt2 + input# + prompt3# prompt3 一位
        else:
            query = prompt1 + input + prompt2 # + prompt3
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
            # 加入最后一位
            raw_embeds[bidx, i+1+counts[bidx]+self.template[0], :] = replace_embeds[i+1+self.template[0], :]
        return raw_embeds

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, input_lens=None):
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
        bz = len(input_ids)
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

        # decode the output ids to see if there is some patterns
        outputs2 = self.LMgpt2(inputs_embeds=inputs, attention_mask=attention_mask1.to(self.device).half())
        example = torch.argsort(outputs2[0], dim=2, descending=True)[0, sum(self.template)+counts[0]+1:, 0]

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence = torch.zeros(bz, bx, self.hidden_size).to(self.device)

        for bdix in range(bz):
            if self.template[0] == self.template[1]:
                place = sum(self.template)+counts[bdix]
            else:
                place = self.template[0] + counts[bdix]
            sequence[bdix, :counts[bdix], :] = sequence_output[bdix, place:place+counts[bdix], :]
            # todo 只截取没有pad的id对应的input

        logits = self.classifier(sequence)
        outputs = (example,)+outputs[2:]
        outputs = (logits,) + outputs # add hidden states and attention if they are here
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)#crf的作用即为计算loss
            # loss = self.lstmcrf(word_embeds=sequence_output, word_seq_length=word_seq_length,
            #                     #emissions=logits,
            #                     tags=labels, mask=attention_mask)

            # def forward(self, emissions: torch.Tensor, tags: torch.LongTensor,
            # mask:Optional[torch.ByteTensor] = None, reduction: str = 'mean') #mask是optional的,
            # -> torch.Tensor:loss
            # mask的作用：在CRF中做了分母
            outputs = (-1*loss,)+outputs

        return outputs # (loss), scores