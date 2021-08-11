import torch

class LabelEmbeder(torch.nn.Module):
    def __init__(self, template, hidden_size, device):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = template[0]

        self.cloze_mask = [[1] * self.cloze_length]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)

        print("init label embeder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        return input_embeds

