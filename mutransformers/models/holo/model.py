from typing import Optional

import mup
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from . import hrr
from .config import HFHoloConfig


class HRRSelfAttention(nn.Module):
    def __init__(self, model_dims):
        super().__init__()
        self.model_dims = model_dims
        self.queries = nn.Linear(model_dims, model_dims, bias=False)
        self.keys = nn.Linear(model_dims, model_dims, bias=False)
        self.values = nn.Linear(model_dims, model_dims, bias=False)
        self.output = nn.Linear(model_dims, model_dims, bias=False)

    def forward(self, x, causal=True, mask=None):
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        values_hat = hrr.key_value_query(k, v, q, causal=causal)
        # values_hat = self.post_kvq(values_hat)
        values_hat = self.output(values_hat)
        return values_hat


class HoloLayer(nn.Module):
    def __init__(self, model_dims, gain_init=1., attention_class=HRRSelfAttention):
        super().__init__()
        self.self_attention = attention_class(model_dims)

    def forward(self, x, mask=None, labels=None):
        values_hat = self.self_attention(x, causal=True)
        x = x + values_hat
        return x


class HoloDecoder(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        layer_class = HoloLayer
        attention_class = dict(
            hrr=HRRSelfAttention,
        )[config.attention_class]

        self.layers = nn.ModuleList([
            layer_class(config.model_dims, attention_class=attention_class)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class HFHolo(PreTrainedModel):
    config_class = HFHoloConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder = HoloDecoder(config)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.model_dims)
        self.input_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        self.predict_token = mup.MuReadout(config.model_dims, config.vocab_size, bias=False)

        freeze_list = []
        if not config.learn_input_embs:
            freeze_list += [self.input_embedding, self.position_embedding]
        if not config.learn_output_embs:
            freeze_list += [self.predict_token]
        if freeze_list:
            list(map(lambda x: x.requires_grad_(False), freeze_list))

        self.post_init()

    def _init_weights(self, module, readout_zero_init=False, query_zero_init=False):
        if isinstance(module, mup.MuReadout) and readout_zero_init:
            module.weight.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617

            if hasattr(module.weight, 'infshape'):
                mup.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            else:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        depth_std = self.config.initializer_range / np.sqrt(2 * self.config.num_hidden_layers)
        for name, module in module.named_modules():
            for target_name in ('queries',):
                if target_name in name and query_zero_init:
                    module.weight.data.zero_()
                    if module.bias is not None:
                        module.bias.data.zero_()

            if "output" in name:
                if hasattr(module.weight, 'infshape'):
                    mup.init.normal_(module.weight, mean=0.0, std=depth_std)
                else:
                    module.weight.data.normal_(mean=0.0, std=depth_std)

    def forward(
        self,
        input_ids: Optional = None,
        attention_mask: Optional = None,
        token_type_ids: Optional = None,
        position_ids: Optional = None,
        head_mask: Optional = None,
        inputs_embeds: Optional = None,
        encoder_hidden_states: Optional = None,
        encoder_attention_mask: Optional = None,
        labels: Optional = None,
        past_key_values: Optional = None,
        use_cache: Optional = None,
        output_attentions: Optional = None,
        output_hidden_states: Optional = None,
        return_dict: Optional = None,
    ):
        tokens = self.input_embedding(input_ids)

        position_ids = torch.arange(tokens.shape[1]).long().to(tokens.device)
        position_ids = position_ids[None, :].repeat(tokens.shape[0], 1)
        positions = self.position_embedding(position_ids)

        if self.config.attention_class == 'hrr':
            inputs = hrr.bind(tokens, positions)
        else:
            inputs = tokens + positions

        feats = self.decoder(inputs)
        logits = self.predict_token(feats)

        loss = 0.
        if labels is not None:
            preds = logits[:, :-1].contiguous().view(-1, logits.shape[-1])
            targets = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(preds, targets)

        if return_dict is not None and not return_dict:
            output = (logits, feats)
            output = (loss,) + output if loss else output
            return output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,  # transformer_outputs.past_key_values,
            hidden_states=feats,
            attentions=None,  # transformer_outputs.attentions,
            cross_attentions=None,  # transformer_outputs.cross_attentions,
        )

    def get_input_embeddings(self):
        return self.input_embedding

    def set_input_embeddings(self, new_embs):
        self.input_embedding = new_embs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        """
        https://github.com/huggingface/transformers/blob/08a194fcd615dcf9406a7e319d637cc303097f46/src/transformers/models/gpt2/modeling_gpt2.py#L1227C5-L1272C28
        """
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    @classmethod
    def mup_base_shapes(cls, filename=None, base_kwargs=None, delta_kwargs=None):
        if not hasattr(cls, '_mup_base_shapes'):
            base_kwargs = base_kwargs or {}
            delta_kwargs = delta_kwargs or {}
            base_config = HFHoloConfig(
                model_dims=128,
                **base_kwargs,
            )
            delta_config = HFHoloConfig(
                model_dims=256,
                **delta_kwargs
            )
            base_model = HFHolo(config=base_config)
            delta_model = HFHolo(config=delta_config)
            base_shapes = mup.make_base_shapes(base_model, delta_model, savefile=filename)
            cls._mup_base_shapes = base_shapes
            del base_model
            del delta_model
            base_model = delta_model = None
        return cls._mup_base_shapes


AutoModel.register(HFHoloConfig, HoloDecoder)
AutoModelForCausalLM.register(HFHoloConfig, HFHolo)
