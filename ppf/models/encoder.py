import torch
import torch.nn as nn
from transformers import AutoModel


class Encoder(nn.Module):
    """
    DistilBERT backbone — returns raw hidden states for both task and privacy paths.

    Returns a tuple:
        cls_embedding:    (batch, hidden_size) — [CLS] token for sentence-level task
        token_embeddings: (batch, seq_len, hidden_size) — all tokens for token-level probing

    No projection head — the task head and probe operate on the full 768-dim
    hidden states so the probe has maximum information to work with.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager",
        )
        self.hidden_size = self.backbone.config.hidden_size  # 768

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls:    (batch, hidden_size)           — for task head
            tokens: (batch, seq_len, hidden_size)  — for privacy probe
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state             # (batch, seq_len, 768)
        cls = tokens[:, 0, :]                      # (batch, 768)
        return cls, tokens

    def freeze_lower_layers(self, num_frozen: int = 4) -> None:
        """Freeze embeddings + first num_frozen transformer layers.

        DistilBERT has 6 layers (0-5). Freezing 0-3 leaves only layers 4-5
        trainable, reducing capacity from ~66M to ~14M params. This balances
        the adversarial game — the encoder can't trivially overwhelm the probe.
        """
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        for i in range(num_frozen):
            for param in self.backbone.transformer.layer[i].parameters():
                param.requires_grad = False

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    encoder = Encoder()
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    cls, tokens = encoder(input_ids, attention_mask)
    assert cls.shape == (2, 768), f"CLS shape: {cls.shape}"
    assert tokens.shape == (2, 32, 768), f"Tokens shape: {tokens.shape}"
    print(f"CLS: {cls.shape}  Tokens: {tokens.shape}")
