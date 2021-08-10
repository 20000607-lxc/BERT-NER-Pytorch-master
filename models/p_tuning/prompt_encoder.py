import torch
import torch.nn as nn

class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, device):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = (template[0], template[1]+1, 0) # todo prompt多加一位，用于将output向后偏移一位

        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=0.0,# self.args.lstm_dropout = 0.0
                                       bidirectional=True,
                                       batch_first=True)

        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self, input_sequence=None):
        if input_sequence == None:
            input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()

        else:
            input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
            input_sequence = input_sequence.unsqueeze(0)
            input_sequence = input_sequence.to(self.device)
            #output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
            input_embeds_cat = torch.cat((input_embeds[:, 0:self.cloze_length[0]-1, :],
                                      input_sequence, input_embeds[:, self.cloze_length[0]:self.cloze_length[0]*2-1, :]), dim=1)

            output_embeds = self.mlp_head(self.lstm_head(input_embeds_cat)[0]).squeeze()

        return output_embeds

