from simplejson import OrderedDict
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartModel, BartForConditionalGeneration, BartEncoderLayer, BartDecoderLayer
from transformers.models.bart.modeling_bart import shift_tokens_right, CrossEntropyLoss, Seq2SeqLMOutput, ACT2FN
from transformers.models.bart.configuration_bart import BartConfig 
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight, gain=0.0000001)

    if bias:
        nn.init.constant_(m.bias, 0.0)
    
    return m

class BartWithAdapterConfig(BartConfig):
    def __init__(
        self,
        vocab_size=50265,
        d_model=1024,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=4096,
        decoder_ffn_dim=4096,
        activation_function='gelu',
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        classifier_dropout=0.0,
        max_position_embeddings=1024,
        init_std=0.02,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        scale_embedding=False,
        is_encoder_decoder=True,
        num_labels=3,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        adapter_dim=64,
        **common_kwargs
    ):
        if 'hidden_size' in common_kwargs:
            raise ValueError('hidden size is called d_model')
        
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_layerdrop = decoder_layerdrop
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std
        self.activation_function = activation_function
        self.scale_embedding = scale_embedding
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout
        self.classifier_dropout = classifier_dropout

        self.adapter_dim = adapter_dim

class ModelWithAdapter(nn.Module):
    def init_adapter(self, input_dim, mid_dim, output_dim, config):
        self.config = config
        self.adapter_name_to_weight = OrderedDict()
        self.adapter_down_weight, self.adapter_up_weight, self.adapter_down_bias, self.adapter_up_bias = None, None, None, None
        self.adapter_id = 0

        self.all_adapters = nn.ModuleList()

        task_num = self.config.task_num

        for i in range(task_num):
            adapter = nn.ModuleList([
                nn.Linear(input_dim, mid_dim),
                nn.Linear(mid_dim, output_dim)]).cuda()
            self.all_adapters.append(adapter)
        
        self.adapter_down_weight = torch.zeros(input_dim, mid_dim).cuda()
        self.adapter_down_bias = torch.zeros(mid_dim).cuda()
        self.adapter_up_weight = torch.zeros(mid_dim, output_dim).cuda()
        self.adapter_up_bias = torch.zeros(output_dim).cuda()
    
    def set_adapter_down_weight(self, tensor):
        self.adapter_down_weight = tensor
        self.all_adapters[self.adapter_id][0].weight.copy_(tensor)
    
    def set_adapter_down_bias(self, tensor):
        self.adapter_down_bias = tensor
        self.all_adapters[self.adapter_id][0].bias.copy_(tensor)

    def set_adapter_up_weight(self, tensor):
        self.adapter_up_weight = tensor
        self.all_adapters[self.adapter_id][0].weight.copy_(tensor)

    def set_adapter_up_bias(self, tensor):
        self.adapter_up_bias = tensor
        self.all_adapters[self.adapter_id][0].bias.copy_(tensor)
    
    def set_adapter_id(self, adapter_id):
        self.adapter_id = adapter_id
    
    def register_adapter_name_to_weight(self, names, weights):
        for name, weight in zip(names, weights):
            self.adapter_name_to_weight[name] = weight
    
    def get_my_module_weight_dims(self):
        return [
            self.adapter_down_weight.size(),
            self.adapter_down_bias.size(),
            self.adapter_up_weight.size(),
            self.adapter_up_bias.size()
        ]

    def adapter_down(self, x):
        return self.all_adapters[self.adapter_id][0](x)
        
    def adapter_up(self, x):
        return self.all_adapters[self.adapter_id][1](x)
        
    def set_adapter_weights(self, weight_vector):
        return self.set_my_adapter_weights(weight_vector)
    
    def set_my_adapter_weights(self, weight_vector):
        sizes = self.get_my_module_weight_dims()
        prev_start = 0
        for size, (name, value) in zip(sizes, self.adapter_name_to_weight.items()):
            flat_size = np.product(size)
            weight_data = weight_vector[prev_start:prev_start + flat_size].cuda()
            
            weight = weight_data.view(*value.size())
            if name == 'adapter_down_weight':
                self.all_adapters[self.adapter_id][0].weight.data.copy_(weight_data.view(*value.size()).t())
            elif name == 'adapter_down_bias':
                self.all_adapters[self.adapter_id][0].bias.data.copy_(weight_data.view(*value.size()))
            elif name == 'adapter_up_weight':
                self.all_adapters[self.adapter_id][1].weight.data.copy_(weight_data.view(*value.size()).t())
            elif name == 'adapter_up_bias':
                self.all_adapters[self.adapter_id][1].bias.data.copy_(weight_data.view(*value.size()))
            else:
                raise ValueError(name)

            prev_start += flat_size

class EncoderLayerWithAdapter(BartEncoderLayer, ModelWithAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter_dim = config.adapter_dim
        self.init_adapter(self.embed_dim, self.adapter_dim, self.embed_dim, config)
        self.register_adapter_name_to_weight(['adapter_down_weight', 'adapter_down_bias','adapter_up_weight',
                                              'adapter_up_bias'],[self.adapter_down_weight, self.adapter_down_bias,
                                             self.adapter_up_weight, self.adapter_up_bias])

    def forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        
        residual_adapter = hidden_states
        hidden_states = self.adapter_down(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.adapter_up(hidden_states)
        hidden_states = residual_adapter + hidden_states

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        
        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class DecoderLayerWithAdapter(BartDecoderLayer, ModelWithAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter_dim = config.adapter_dim
        self.init_adapter(self.embed_dim, self.adapter_dim, self.embed_dim, config)
        self.register_adapter_name_to_weight(
            ['adapter_down_weight', 'adapter_down_bias','adapter_up_weight', 'adapter_up_bias'],
            [self.adapter_down_weight, self.adapter_down_bias, self.adapter_up_weight, self.adapter_up_bias]
        )

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, layer_head_mask=None, encoder_layer_head_mask=None, past_key_value=None, output_attentions=False, use_cache=True):
        residual = hidden_states

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        residual_adapter = hidden_states
        hidden_states = self.adapter_down(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.adapter_up(hidden_states)
        hidden_states = residual_adapter + hidden_states

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            present_key_value = present_key_value + cross_attn_present_key_value

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BartEncodeWithAdapter(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens):
        super(BartEncodeWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [EncoderLayerWithAdapter(config) for _ in range(config.encoder_layers)]
        )

class BartDecoderWithAdapter(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super(BartDecoderWithAdapter, self).__init__(config, embed_tokens)
        self.layers = nn.ModuleList(
            [DecoderLayerWithAdapter(config) for _ in range(config.decoder_layers)]
        )

class BartModelWithAdapter(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModelWithAdapter, self).__init__(config)
        self.encoder = BartEncodeWithAdapter(config, self.shared)
        self.decoder = BartDecoderWithAdapter(config, self.shared)
