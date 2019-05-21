import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO: Create two maps as described in the docstring above.
    # It's best if you also sort the chars before assigning indices, so that
    # they're in lexical order.
    unique = np.unique(list(text)) #unique is already sorted
    char_to_idx = {}
    idx_to_char = {}

    for index, char in enumerate(unique):
        char_to_idx[char] = index
        idx_to_char[index] = char

    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    table = str.maketrans({char: None for char in chars_to_remove})
    text_clean = text.translate(table)
    n_removed = len(text) - len(text_clean)
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    N = len(text)
    D = len(char_to_idx)
    text = list(text)
    result = torch.zeros((N, D), dtype=torch.int8)
    indices = [char_to_idx[char] for char in text]
    result[range(N), indices] = 1
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """

    # TODO: Implement the reverse-embedding.
    indices = torch.argmax(embedded_text, dim=1)
    result_l = [idx_to_char[int(index)] for index in indices]
    result = ''.join(result_l)
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO: Implement the labelled samples creation.
    # 1. Embed the given text.
    # 2. Create the samples tensor by splitting to groups of seq_len.
    #    Notice that the last char has no label, so don't use it.
    # 3. Create the labels tensor in a similar way and convert to indices.
    # Note that no explicit loops are required to implement this function.

    V = len(char_to_idx)
    S = seq_len
    L = len(text) - 1
    N = int(L / S)

    # TODO should we do padding for last sample?
    embedded_text = chars_to_onehot(text, char_to_idx)[:-1]
    embedded_text = embedded_text[:N * S]
    samples = embedded_text.reshape((N, S, V))
    # TODO make sure reshape is good

    text_list = list(text)
    text_list_cut = text_list[1:N * S + 1]

    labels = torch.Tensor([char_to_idx[char] for char in text_list_cut]).reshape((N, S))
    #TODO avoid loop in tensor initialization

    samples = samples.to(device)
    labels = labels.to(device)

    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    result = torch.exp(y/temperature)
    denom = torch.sum(result, dim=dim)
    result *= 1/denom
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO: Implement char-by-char text generation.
    # 1. Feed the start_sequence into the model.
    # 2. Sample a new char from the output distribution of the last output
    #    char. Convert output to probabilities first.
    #    See torch.multinomial() for the sampling part.
    # 3. Feed the new char into the model.
    # 4. Rinse and Repeat.
    #
    # Note that tracking tensor operations for gradient calculation is not
    # necessary for this. Best to disable tracking for speed.
    # See torch.no_grad().

    x = start_sequence
    h = None
    with torch.no_grad():
        for _ in range(n_chars - len(start_sequence)):
            samples = chars_to_onehot(x, char_to_idx).to(device)
            samples = samples.reshape((1, samples.shape[0], samples.shape[1]))
            y, h = model(samples.to(dtype=torch.float), h)
            char_probabilities = hot_softmax(y, temperature=T).reshape((y.shape[1], 81))
            new_char_index = torch.multinomial(char_probabilities, 1)[-1].item()
            new_char = idx_to_char[new_char_index]
            x += new_char

    return x


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []
        self.dropout = nn.Dropout(dropout)


        # TODO: Create the parameters of the model.
        # To implement the affine transforms you can use either nn.Linear
        # modules (recommended) or create W and b tensor pairs directly.
        # Create these modules or tensors and save them per-layer in
        # the layer_params list.
        # Important note: You must register the created parameters so
        # they are returned from our module's parameters() function.
        # Usually this happens automatically when we assign a
        # module/tensor as an attribute in our module, but now we need
        # to do it manually since we're not assigning attributes. So:
        #   - If you use nn.Linear modules, call self.add_module() on them
        #     to register each of their parameters as part of your model.
        #   - If you use tensors directly, wrap them in nn.Parameter() and
        #     then call self.register_parameter() on them. Also make
        #     sure to initialize them. See functions in torch.nn.init.

        def inner_layer_params(j):

            # z:
            W_x_z = nn.Linear(h_dim, h_dim, bias=False)
            self.add_module('W_x_z_' + j, W_x_z)
            W_h_z = nn.Linear(h_dim, h_dim, bias=True)
            self.add_module('W_h_z_' + j, W_h_z)
            # r:
            W_x_r = nn.Linear(h_dim, h_dim, bias=False)
            self.add_module('W_x_r_' + j, W_x_r)
            W_h_r = nn.Linear(h_dim, h_dim, bias=True)
            self.add_module('W_h_r_' + j, W_h_r)
            # g:
            W_x_g = nn.Linear(h_dim, h_dim, bias=False)
            self.add_module('W_x_g_' + j, W_x_g)
            W_h_g = nn.Linear(h_dim, h_dim, bias=True)
            self.add_module('W_h_g_' + j, W_h_g)
            return W_x_z, W_h_z, W_x_r, W_h_r, W_x_z, W_h_z

        #first layer -
        W_x_z_first = nn.Linear(in_dim, h_dim, bias=False)
        self.add_module('W_x_z_0', W_x_z_first)
        W_h_z_first = nn.Linear(h_dim, h_dim, bias=True)
        self.add_module('W_h_z_0', W_h_z_first)
        # r:
        W_x_r_first  = nn.Linear(in_dim, h_dim, bias=False)
        self.add_module('W_x_r_0', W_x_r_first)
        W_h_r_first  = nn.Linear(h_dim, h_dim, bias=True)
        self.add_module('W_h_r_0', W_h_r_first)
        # g:
        W_x_g_first  = nn.Linear(in_dim, h_dim, bias=False)
        self.add_module('W_x_g_0', W_x_g_first)
        W_h_g_first  = nn.Linear(h_dim, h_dim, bias=True)
        self.add_module('W_h_g_0', W_h_g_first)

        self.layer_params.append((W_x_z_first, W_h_z_first, W_x_r_first, W_h_r_first, W_x_g_first, W_h_g_first))

        for i in range(1, n_layers):
            self.layer_params.append(inner_layer_params(str(i)))

        # last layer -
        W_h_y = nn.Linear(h_dim, out_dim, bias=True)
        self.add_module('W_h_y', W_h_y)

        self.layer_params.append(W_h_y)

        # TODO make sure function is good

    def forward(self, input: Tensor, hidden_state: Tensor=None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: Implement the model's forward pass.
        # You'll need to go layer-by-layer from bottom to top (see diagram).
        # Tip: You can use torch.stack() to combine multiple tensors into a
        # single tensor in a differentiable manner.

        s = nn.Sigmoid()
        t = nn.Tanh()

        #layer_output2 = torch.empty((batch_size, seq_len, self.out_dim))
        layer_output = []

        for char_index in range(seq_len):
            x = layer_input[:, char_index, :]

            for layer_num, params, state in zip(range(self.n_layers), self.layer_params[:-1], layer_states):
                W_x_z, W_h_z, W_x_r, W_h_r, W_x_g, W_h_g = params
                z = s(W_x_z(x) + W_h_z(state))
                r = s(W_x_r(x) + W_h_r(state))
                g = t(W_x_g(x) + W_h_g(state*r))
                x = self.dropout(z*state + (1-z)*g)
                layer_states[layer_num] = z*state + (1-z)*g

            W_h_y = self.layer_params[-1]
            y = W_h_y(x)
            #layer_output2[:, char_index, :] = y
            layer_output.append(y)


        # hidden_state2 = torch.empty((batch_size, self.n_layers, self.h_dim))
        # for i, layer_state in enumerate(layer_states):
        #    hidden_state2[:, i, :] = layer_state

        hidden_state = torch.stack(layer_states, dim=1).reshape((batch_size, self.n_layers, self.h_dim))
        layer_output = torch.stack(layer_output, dim=1).reshape((batch_size, seq_len, self.out_dim))

        #b1 = torch.equal(layer_output, layer_output2)
        #b2 = torch.equal(hidden_state, hidden_state2)

        return layer_output, hidden_state

        # TODO make sure function is good

