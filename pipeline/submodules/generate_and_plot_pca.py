import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_activations_pre_hook(layer, cache: Float[Tensor, "batch layer d_model"], n_samples, positions: List[int],batch_id,batch_size):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[batch_id:batch_id+batch_size, layer] =  torch.squeeze(activation[:, positions, :],1)
    return hook_fn

def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return mean_activations

def get_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)


    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        fwd_pre_hooks = [(block_modules[layer],
                          get_activations_pre_hook(layer=layer,
                                                   cache=activations,
                                                   n_samples=n_samples,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size)) for layer in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations

def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    mean_activations_harmful = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    mean_activations_harmless = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    return mean_diff

def get_activations_all(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module],batch_size=32, positions=[-1]):
    activations_harmful = get_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules,
                                          batch_size=batch_size, positions=positions)
    activations_harmless = get_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules,
                                           batch_size=batch_size, positions=positions)
    n_samples = 2*len(activations_harmful)

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_harmful,activations_harmless),dim=0)


    return activations_all

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions, model_base.tokenize_instructions_fn, model_base.model_block_modules, positions=list(range(-len(model_base.eoi_toks), 0)))

    assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs

def generate_and_plot_pca(cfg,model_base: ModelBase, harmful_instructions, harmless_instructions,batch_size=1):
    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    activations_all = get_activations_all(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions,
                             model_base.tokenize_instructions_fn, model_base.model_block_modules,
                             positions=[-1],
                             batch_size=batch_size,
                                          )
    # todo: do pca here
    pca = PCA(n_components=3)

    label_text = []
    label = []
    for ii in range(len(harmless_instructions)):
        label_text = np.append(label_text, f'harmless:{ii}')
        label.append(0)
    for ii in range(len(harmful_instructions)):
        label_text = np.append(label_text, f'harmful:{ii}')
        label.append(1)
    n_layers = model_base.model.config.num_hidden_layers
    cols = 4
    rows = int(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in range(n_layers)])


    for row in range(rows):
        # print(f'row:{row}')
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer <= n_layers:
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                df = {}
                df['label'] = label
                df['pca0'] = activations_pca[:, 0]
                df['pca1'] = activations_pca[:, 1]
                df['label_text'] = label_text
                color_map = {'harmless': 'red', 'virginica': 'blue'}
                iris_color = df['label']
                fig.add_trace(
                    go.Scatter(x=df['pca0'],
                               y=df['pca1'],
                               mode='markers',
                               marker_color=df['label'],
                               text=df['label_text']),
                    row=row + 1, col=ll + 1,
                )

    fig.update_layout(height=1600, width=1000)
    fig.show()
    model_name =cfg.model_alias
    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'refusal_activation_pca.html')
    # fig.write_image(artifact_dir + os.sep + model_name + '_' + 'refusal_activation_pca_'  + '_layer_'  + '_' + + '.png')


