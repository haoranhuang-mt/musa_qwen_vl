import sys
sys.path.append("..")
import warnings
from functools import partial
from typing import Callable, Dict, List, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from transformers.modeling_outputs import BaseModelOutputWithPast

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    VocabParallelEmbedding1D,
)
from colossalai.shardformer.layer._operation import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)


from colossalai.shardformer.modeling.llama import (
    # LlamaPipelineForwards,
    get_llama_flash_attention_forward,
)
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)
from colossalai.shardformer.layer.utils import SeqParallelUtils

from colossalai.lazy import LazyInitContext
from utils.global_vars import get_fuse_flag
from .llama_forward_fn import LlamaPipelineForwards


__all__ = [
    "LlamaPolicy",
    "LlamaForCausalLMPolicy",
    "LlamaForSequenceClassificationPolicy",
]


class RMSNorm(object):
    r"""
    This is a wrapper around the RMSNorm. It is meant to be used only with the from_native_module interface.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "FusedLayerNorm is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to convert a native RMSNorm module to colossalai layer norm module."
        )

    @staticmethod
    def from_native_module(
        module: nn.Module, sp_partial_derived: bool = False, *args, **kwargs
    ) -> nn.Module:
        """
        Convert a native RMSNorm module to colossalai layer norm module,
        and optionally mark parameters for gradient aggregation.

        Args:
            module (nn.Module): The native RMSNorm module to be converted.
            sp_partial_derived (bool): Whether this module's gradients are partially derived in sequence parallelism.

        Returns:
            nn.Module: The RMSNorm module.
        """

        LazyInitContext.materialize(module)

        if sp_partial_derived:
            # Since gradients are computed using only a subset of the data,
            # aggregation of these gradients is necessary during backpropagation.
            # Therefore, we annotate these parameters in advance to indicate the need for gradient aggregation.
            SeqParallelUtils.marked_as_sp_partial_derived_param(module.weight)

        return module


def llama1_sequence_parallel_forward_fn(shard_config):
    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            #if use_cache:
            #    logger.warning_once(
            #        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            #    )
            use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        hidden_states = split_forward_gather_backward(
            hidden_states,
            dim=1,
            process_group=shard_config.tensor_parallel_process_group,
        )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        hidden_states = gather_forward_split_backward(
            hidden_states,
            dim=1,
            process_group=shard_config.tensor_parallel_process_group,
        )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return forward


class LlamaPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        if self.shard_config.enable_tensor_parallelism:
            # Resize embedding
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size

            if vocab_size % world_size != 0:
                new_vocab_size = vocab_size + world_size - vocab_size % world_size
                self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from .modeling_llama import (
            LlamaAttention,
            LlamaDecoderLayer,
            LlamaModel,
        )

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            # self.shard_config.enable_sequence_parallelism = False
            warnings.warn("Llama2 dosen't support sequence parallelism now")
        use_sequence_parallel = self.shard_config.enable_sequence_parallelism
        overlap = False
        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = (
                    self.model.config.num_key_value_heads
                    // self.shard_config.tensor_parallel_size
                )
            assert get_fuse_flag()  # make sure fused

            # assert self.timers is not None
            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.qkv_proj",
                        target_module=Linear1D_Col,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                            # "timers": self.timers
                        },
                    ),
                    # SubModuleReplacementDescription(
                    #     suffix="self_attn.k_proj",
                    #     target_module=Linear1D_Col,
                    # ),
                    # SubModuleReplacementDescription(
                    #     suffix="self_attn.v_proj",
                    #     target_module=Linear1D_Col,
                    # ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_and_up_proj",
                        target_module=Linear1D_Col,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                        },
                    ),
                    # SubModuleReplacementDescription(
                    #     suffix="mlp.up_proj",
                    #     target_module=Linear1D_Col,
                    # ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                        },
                    ),
                ],
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=VocabParallelEmbedding1D,
                ),
                policy=policy,
                target_key=LlamaModel,
            )
        if use_sequence_parallel:
            self.append_or_create_method_replacement(
                description={
                    "forward": llama1_sequence_parallel_forward_fn(self.shard_config)
                },
                policy=policy,
                target_key=LlamaModel,
            )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="input_layernorm",
                    target_module=RMSNorm,
                    kwargs={"sp_partial_derived": use_sequence_parallel},
                ),
                SubModuleReplacementDescription(
                    suffix="post_attention_layernorm",
                    target_module=RMSNorm,
                    kwargs={"sp_partial_derived": use_sequence_parallel},
                ),
            ],
            policy=policy,
            target_key=LlamaDecoderLayer,
        )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=RMSNorm,
                )
            ],
            policy=policy,
            target_key=LlamaModel,
        )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_llama_flash_attention_forward(),
                },
                policy=policy,
                target_key=LlamaAttention,
            )

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: Dict
    ) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "LlamaModel":
                module = self.model
            else:
                module = self.model.model

            layers_per_stage = Policy.distribute_layers(
                len(module.layers), stage_manager.num_stages
            )
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {
                "forward": partial(
                    new_forward, stage_manager=stage_manager, stage_index=stage_index, shard_config=self.shard_config
                )
            }
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )

        return

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "LlamaModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(
            len(module.layers), stage_manager.num_stages
        )
        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)

        # print(f"#### On rank {self.pipeline_stage_manager.get_rank()}, we have:")
        # for i, layer in enumerate(held_layers):
        # print(list(layer.named_parameters()))
        # print(f"layer {i}: {type(layer)}, device = {list(layer.named_parameters())[0][1].device}")
        # if param:
        #     print(f"layer {i}: {name}, device = {param.device}")
        # else:
        #     print(f"layer {i}: {name}, param is None!")

        return held_layers


class LlamaModelPolicy(LlamaPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        from .modeling_llama import LlamaModel

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=LlamaModel,
                new_forward=LlamaPipelineForwards.llama_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in llama model"""
        return []


class LlamaForCausalLMPolicy(LlamaPolicy):
    def module_policy(self):
        from .modeling_llama import LlamaForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                LlamaForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=Linear1D_Col,
                            kwargs=dict(gather_output=True),
                        )
                    ]
                )
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=LlamaForCausalLM,
                new_forward=LlamaPipelineForwards.llama_for_causal_lm_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        llama_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if (
                id(llama_model.embed_tokens.weight) == id(self.model.lm_head.weight)
                and self.pipeline_stage_manager.num_stages > 1
            ):
                # tie weights
                return [
                    {
                        0: llama_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: self.model.lm_head.weight,
                    }
                ]
        return []


class LlamaForSequenceClassificationPolicy(LlamaPolicy):
    def module_policy(self):
        from .modeling_llama import LlamaForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for sequence classification
            new_item = {
                LlamaForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="score",
                            target_module=Linear1D_Col,
                            kwargs=dict(gather_output=True),
                        )
                    ]
                )
            }
            policy.update(new_item)
        # to be confirmed
        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=LlamaForSequenceClassification,
                new_forward=LlamaPipelineForwards.llama_for_sequence_classification_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in llama for sequence classification model"""
        return []

