
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_and_cache_direction_ablation_input_pre_hook(direction: Tensor, cache: Float[Tensor, "batch layer d_model"],
                                                    layer:int,positions: List[int],batch_id:int,batch_size:int,
                                                    target_layer,
                                                    len_prompt=1):
    def hook_fn(module, input):
        nonlocal direction, cache, layer, positions, batch_id, batch_size,target_layer,len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # only apply the ablation to the target layers
        if layer in target_layer:
                direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                direction = direction.to(activation)
                activation -= (activation @ direction).unsqueeze(-1) * direction
                # only cache the last token of the prompt not the generated answer
                if activation.shape[1]==len_prompt:
                     cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)

        # if not target layer, cache the original activation value
        else:
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                    cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_and_cache_activation_addition_input_pre_hook(direction: Tensor, cache: Float[Tensor, "batch layer d_model"],
                                                    layer:int,positions: List[int],batch_id:int,batch_size:int,
                                                    target_layer,
                                                    len_prompt=1,coeff=1):
    def hook_fn(module, input):
        nonlocal direction, cache, layer, positions, batch_id, batch_size,target_layer,len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # only apply the ablation to the target layers
        if layer in target_layer:
            # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation += direction*coeff

            # only cache the last token of the prompt not the generated answer
            if activation.shape[1]==len_prompt:
                 cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)

        # if not target layer, cache the original activation value
        else:
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1]==len_prompt:
                 cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_and_cache_direction_ablation_output_hook(direction: Tensor,
                                                 layer:int,positions: List[int],batch_id:int,batch_size:int,
                                                 target_layer,
                                                 ):
    def hook_fn(module, input,output):
        nonlocal direction, layer, positions, batch_id, batch_size,target_layer

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only apply the ablation to the target layers
        if layer in target_layer:
                direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                direction = direction.to(activation)
                activation -= (activation @ direction).unsqueeze(-1) * direction
                # cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)
        # if not target layer, cache the original activation value
        # else:
            # cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn

def get_and_cache_activation_addition_output_hook(direction: Tensor,
                                                    layer:int,positions: List[int],batch_id:int,batch_size:int,
                                                    target_layer,coeff=1):
    def hook_fn(module, input,output):
        nonlocal direction, layer, positions, batch_id, batch_size,target_layer

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only apply the ablation to the target layers
        if layer in target_layer:
            # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation += direction*coeff
            # cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)
        # if not target layer, cache the original activation value
        # else:
            # cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn
def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn