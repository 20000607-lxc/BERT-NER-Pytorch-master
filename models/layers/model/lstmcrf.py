# 
# @author: Allan
#
import torch
import torch.nn as nn
from typing import List, Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NNCRF(nn.Module):

    def __init__(self, config, device, num_tags, batch_first=True):
        #super(NNCRF, self).__init__()
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super(NNCRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        #Whether the first dimension corresponds to the size of a minibatch.
        #标记emissions的shape是（batch_size, sequence_length, num_tags)[true]还是（sequence_length, batch_size, num_tags)[false]
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.label_size = config.num_labels
        self.device = device

        self.label2idx = config.label2id
        self.labels = config.id2label

        self.input_size = 768 #config.hidden_size#768
        self.hidden_state = self.input_size//2

        """
            Input size to LSTM description
        """

        num_layers = 2
        # if config.num_lstm_layer > 1 and self.dep_model != DepModelType.dglstm:
        #     num_layers = config.num_lstm_layer
        # if config.num_lstm_layer > 0:
        self.lstm = nn.LSTM(self.input_size, self.hidden_state, num_layers=num_layers, batch_first=True, bidirectional=True).to(self.device)
        self.num_lstm_layer = 2#config.num_lstm_layer
        self.lstm_hidden_dim = self.hidden_state#config.hidden_dim
        #self.drop_lstm = nn.Dropout(0.0).to(self.device)
        #final_hidden_dim = config.hidden_dim if self.num_lstm_layer >0 else self.input_size
        #final_hidden_dim = self.hidden_state if self.num_lstm_layer > 0 else self.input_size
        """
        Model description
        """
        self.hidden2tag = nn.Linear(self.input_size, self.label_size).to(self.device)
        # init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        # init_transition[:, self.start_idx] = -10000.0
        # init_transition[self.end_idx, :] = -10000.0
        # init_transition[:, self.pad_idx] = -10000.0
        # init_transition[self.pad_idx, :] = -10000.0
        # self.transition = nn.Parameter(init_transition)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def neural_scoring(self, word_emb, word_seq_lens, batch_context_emb = None):# char_inputs, char_seq_lens, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, dep_head_tensor, dep_label_tensor, trees=None):
        """
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :param chars: (batch_size * sent_len * word_length)
        :param char_seq_lens: numpy (batch_size * sent_len , 1)
        :param dep_label_tensor: (batch_size, max_sent_len)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        # batch_size = word_seq_tensor.size(0)
        # sent_len = word_seq_tensor.size(1)

        #word_emb = self.word_embedding(word_seq_tensor)
        # if self.use_char:
        #     if self.dep_model == DepModelType.dglstm:
        #         char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lens)
        #         word_emb = torch.cat((word_emb, char_features), 2)
        # if self.dep_model == DepModelType.dglstm:
        #     size = self.embedding_dim if not self.use_char else (self.embedding_dim + self.charlstm_dim)
        #     dep_head_emb = torch.gather(word_emb, 1, dep_head_tensor.view(batch_size, sent_len, 1).expand(batch_size, sent_len, size))
        # if self.context_emb != ContextEmb.none:
        #     word_emb = torch.cat((word_emb, batch_context_emb.to(self.device)), 2)
        # if self.use_char:
        #     if self.dep_model != DepModelType.dglstm:
        #         char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lens)
        #         word_emb = torch.cat((word_emb, char_features), 2)
        # if self.dep_model == DepModelType.dglstm:
        #     dep_emb = self.dep_label_embedding(dep_label_tensor)
        #     word_emb = torch.cat((word_emb, dep_head_emb, dep_emb), 2)
        word_rep = word_emb #self.word_drop(word_emb)
        #这里sort的原因是pack_padded_sequence需要一个batch中input的长度从大到小排列
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        if self.num_lstm_layer > 0:
            #sorted_seq_tensor 类型为int64， torch.LongTensor
            packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
            lstm_out, _ = self.lstm(packed_words, None)
            feature_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
            #feature_out = self.drop_lstm(lstm_out)
        else:
            feature_out = sorted_seq_tensor

        """
        Higher order interactions
        """
        # if self.num_lstm_layer > 1 and (self.dep_model == DepModelType.dglstm):
        #     for l in range(self.num_lstm_layer-1):
        #         dep_head_emb = torch.gather(feature_out, 1, dep_head_tensor[permIdx].view(batch_size, sent_len, 1).expand(batch_size, sent_len, self.lstm_hidden_dim))
        #         if self.interaction_func == InteractionFunction.concatenation:
        #             feature_out = torch.cat((feature_out, dep_head_emb), 2)
        #         elif self.interaction_func == InteractionFunction.addition:
        #             feature_out = feature_out + dep_head_emb
        #         elif self.interaction_func == InteractionFunction.mlp:
        #             feature_out = F.relu(self.mlp_layers[l](feature_out) + self.mlp_head_linears[l](dep_head_emb))
        #
        #         packed_words = pack_padded_sequence(feature_out, sorted_seq_len, True)
        #         lstm_out, _ = self.add_lstms[l](packed_words, None)
        #         lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        #         feature_out = self.drop_lstm(lstm_out)
        """
        Model forward if we have GCN
        """
        # if self.dep_model == DepModelType.dggcn:
        #     feature_out = self.gcn(feature_out, sorted_seq_len, adj_matrixs[permIdx], dep_label_adj[permIdx])

        outputs = self.hidden2tag(feature_out)#feature_out = [216 x 768]
        return outputs[recover_idx]
        #return: emission scores (batch_size, sent_len, hidden_dim)#hidden_dim should = label num

    def forward(self, word_embeds: torch.Tensor, word_seq_length: torch.Tensor,
                #emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        #self._validate(emissions, tags=tags, mask=mask)

        #TODO 实际上唯一的改进就是这里把sequence先过了一遍lstm再过classifier

        emissions = self.neural_scoring(word_emb=word_embeds, word_seq_lens=word_seq_length)
        for i in range(len(emissions)):
            emissions[i] = emissions[i]*10
            # TODO neural_scoring 之后的数量级是0.01, 但是本来的crf要求的emissions的数量级是0.1, 所以尝试一下乘以10看效果是否有改变，

        self._validate(emissions, tags=tags, mask=mask)
        if self.batch_first:# 即shape: (batch_size,)
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)#这个score是分子，也就是ppt中的(X,y)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)#ppt中的第3部分
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, word_embeds: torch.Tensor, word_seq_length: torch.LongTensor,
               #emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        # if mask is None:
        #     mask = torch.ones(emissions.shape[:2], dtype=torch.uint8,
        #                       device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()

        #self._validate(emissions, mask=mask)

        emissions = self.neural_scoring(word_emb=word_embeds, word_seq_lens=word_seq_length)
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor,
                  tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        """
        emissions: (seq_length, batch_size, num_tags)
        tags: (seq_length, batch_size)
        mask: (seq_length, batch_size)
        """

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]#sart_transitions的shape:[num_tags,num_tags]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]# 即ppt中的h(y[i];X)

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]
        #写错了？score的shape只有[num_tags,num_tags]呀？

        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.contiguous().view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.contiguous().view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.contiguous().view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor,
                              mask: torch.ByteTensor,
                              nbest: int,
                              pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        # + score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # + history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.contiguous().view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.contiguous().view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.contiguous().view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.contiguous().view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
            .contiguous().view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].contiguous().view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.contiguous().view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)

    # def calculate_all_scores(self, features):
    #     batch_size = features.size(0)
    #     seq_len = features.size(1)
    #     scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
    #              features.view(batch_size, seq_len, 1, self.label_size).expand(batch_size,seq_len,self.label_size, self.label_size)
    #     return scores
    #
    # def forward_unlabeled(self, all_scores, word_seq_lens, masks):
    #     batch_size = all_scores.size(0)
    #     seq_len = all_scores.size(1)
    #     alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)
    #
    #     alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.
    #
    #     for word_idx in range(1, seq_len):
    #         ## batch_size, self.label_size, self.label_size
    #         before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
    #         alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)
    #
    #     ### batch_size x label_size
    #     last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
    #     last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
    #     last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)
    #
    #     return torch.sum(last_alpha)
    #
    # def forward_labeled(self, all_scores, word_seq_lens, tags, masks):
    #     '''
    #     :param all_scores: (batch, seq_len, label_size, label_size)
    #     :param word_seq_lens: (batch, seq_len)
    #     :param tags: (batch, seq_len)
    #     :param masks: batch, seq_len
    #     :return: sum of score for the gold sequences
    #     '''
    #     batchSize = all_scores.shape[0]
    #     sentLength = all_scores.shape[1]
    #
    #     ## all the scores to current labels: batch, seq_len, all_from_label?
    #     currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
    #     if sentLength != 1:
    #         tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
    #     tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
    #     endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
    #     tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
    #     score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
    #     if sentLength != 1:
    #         score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
    #     return score
    #
    # def neg_log_obj(self, emissions, tags, mask): # chars, char_seq_lens, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, tags, batch_dep_label, trees=None):
    #     # features = self.neural_scoring(words, word_seq_lens, batch_context_emb) #chars, char_seq_lens, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, batch_dep_label, trees
    #     # features = logits
    #     features = emissions
    #     all_scores = self.calculate_all_scores(features)
    #
    #     batch_size = features.size(0)
    #     sent_len = features.size(1)
    #     word_seq_lens = torch.zeros(batch_size)
    #     for i in range(batch_size):
    #         word_seq_lens[i] = sent_len
    #
    #     # maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
    #     # mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
    #
    #     unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens, mask)
    #     labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
    #     return unlabed_score - labeled_score

    # def viterbiDecode(self, all_scores, word_seq_lens):
    #     batchSize = all_scores.shape[0]
    #     sentLength = all_scores.shape[1]
    #     # sent_len =
    #     scoresRecord = torch.zeros([batchSize, sentLength, self.label_size]).to(self.device)
    #     idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64).to(self.device)
    #     mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
    #     startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64).to(self.device)
    #     decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)
    #
    #     scores = all_scores
    #     # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
    #     scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
    #     idxRecord[:,  0, :] = startIds
    #     for wordIdx in range(1, sentLength):
    #         ### scoresIdx: batch x from_label x to_label at current index.
    #         scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
    #                                                                                                self.label_size) + scores[:, wordIdx, :, :]
    #         idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
    #         scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
    #
    #     lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
    #     lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
    #     decodeIdx[:, 0] = torch.argmax(lastScores, 1)
    #     bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))
    #
    #     for distance2Last in range(sentLength - 1):
    #         lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
    #         decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)
    #
    #     return bestScores, decodeIdx

    # def decode(self, batchInput):
    #     wordSeqTensor, wordSeqLengths, batch_context_emb, charSeqTensor, charSeqLengths, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, trees, tagSeqTensor, batch_dep_label = batchInput
    #     features = self.neural_scoring(wordSeqTensor, wordSeqLengths, batch_context_emb,charSeqTensor,charSeqLengths, adj_matrixs, adjs_in, adjs_out, graphs, dep_label_adj, batch_dep_heads, batch_dep_label, trees)
    #     all_scores = self.calculate_all_scores(features)
    #     bestScores, decodeIdx = self.viterbiDecode(all_scores, wordSeqLengths)
    #     return bestScores, decodeIdx
