from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device, load_seq2seq_model_and_tokenizer


class AttackDecoderModel(nn.Module):
    """ALGEN-style embedding-to-text generator with an mT5/T5 backbone."""

    def __init__(
        self,
        model_name: str,
        input_dim: Optional[int] = None,
        prompt_length: int = 32,
        max_length: int = 32,
        normalize_input: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or get_device()
        self.encoder_decoder, self.tokenizer = load_seq2seq_model_and_tokenizer(
            model_name, self.device
        )
        self.model_name = model_name
        self.decoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.input_dim = input_dim or self.decoder_hidden_dim
        self.prompt_length = prompt_length
        self.max_length = max_length
        self.normalize_input = normalize_input
        self.encoder_decoder.config.max_length = max_length

        bottleneck_dim = max(self.input_dim, self.decoder_hidden_dim)
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.input_dim, bottleneck_dim),
            nn.Dropout(getattr(self.encoder_decoder.config, "dropout_rate", 0.0)),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.decoder_hidden_dim * prompt_length),
        ).to(self.device)

    def _decoder_start_token_id(self) -> int:
        config = self.encoder_decoder.config
        if config.decoder_start_token_id is not None:
            return config.decoder_start_token_id
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        return self.tokenizer.eos_token_id

    def get_encoder_inputs(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.to(self.device)
        if self.normalize_input:
            hidden_states = F.normalize(hidden_states, p=2, dim=1)
        prompt = self.embedding_transform(hidden_states)
        prompt = prompt.reshape(
            hidden_states.shape[0], self.prompt_length, self.decoder_hidden_dim
        )
        attention_mask = torch.ones(
            prompt.shape[:2], dtype=torch.long, device=prompt.device
        )
        return prompt, attention_mask

    def forward(self, inputs: Dict[str, torch.Tensor]):
        if self.training:
            self.encoder_decoder.train()
        inputs_embeds, attention_mask = self.get_encoder_inputs(inputs["hidden_states"])
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=inputs["labels"].to(self.device),
        )

    @torch.no_grad()
    def generate(self, inputs: Dict[str, torch.Tensor], **generation_kwargs):
        inputs_embeds, attention_mask = self.get_encoder_inputs(inputs["hidden_states"])
        defaults = {
            "max_length": self.max_length,
            "num_beams": 3,
            "repetition_penalty": 2.0,
            "length_penalty": 2.0,
            "early_stopping": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        defaults.update(generation_kwargs)
        decoder_input_ids = torch.full(
            (inputs_embeds.shape[0], 1),
            self._decoder_start_token_id(),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **defaults,
        )
