from collections import defaultdict

import torch
import torch.nn as nn


def _get_coverage_tables(nets):
    coverage_tables = []
    for net in nets:
        coverage_table = _init_model_coverage_table(net=net)
        coverage_tables.append(coverage_table)
    return coverage_tables


def _init_model_coverage_table(net):
    coverage_table = defaultdict(bool)
    flatten_net = list(_flatten_net(net))
    for name, module in flatten_net:
        if isinstance(module, nn.Conv2d):
            output_channels = module.out_channels
        elif isinstance(module, nn.Linear):
            output_channels = module.out_features
        else:
            continue

        for index in range(output_channels):
            coverage_table[(name, index)] = False
    return coverage_table


def _flatten_net(module, parent_name=None):
    def is_flatten_module(module_):
        return isinstance(module_, nn.Linear) or isinstance(module_, nn.Conv2d)

    def get_module_name(module_name):
        return "%s.%s" % (parent_name, module_name) if parent_name else module_name

    # flatten_net = []
    for name, child_module in module.named_children():
        if is_flatten_module(module_=child_module):
            # flatten_net.append((get_module_name(name), child_module))
            yield get_module_name(name), child_module
        elif isinstance(child_module, nn.Sequential):
            # flatten_net.extend(_flatten_net(child_module, parent_name=name))
            yield from _flatten_net(child_module, parent_name=name)
        else:
            continue
        # for index in range(output_channels):
        #     coverage_table[(name, index)] = False
    # return coverage_table
    # return flatten_net


def _register_activation_hooks(name, activation_hooks):
    def hook(_, __, output):
        activation_hooks[name] = output

    return hook


def _get_neuron_to_cover(ct):
    table_keys = [(layer_name, index)
                  for (layer_name, index), v in ct.items()
                  if not v]
    if not table_keys:
        table_keys = tuple(ct.keys())

    high = len(table_keys)
    idx_choice = torch.randint(high, (1,)).item()
    layer_name, neuron_idx = table_keys[idx_choice]
    return layer_name, neuron_idx


class _HiddenOutput:
    def __init__(self, hidden_outputs, module_name):
        self.__hidden_outputs = hidden_outputs
        self.__module_name = module_name

    def __call__(self, m, i, o):
        self.__hook_fn(m, i, o)

    def __hook_fn(self, m, i, o):
        self.__hidden_outputs[self.__module_name] = o


class CoverageData(defaultdict):
    def __init__(self, nets):
        super(CoverageData, self).__init__()
        self.__hidden_outputs = []
        self.__neurons_to_cover = []
        self.__coverage_tables = _get_coverage_tables(nets)
        self.__get_neurons_to_cover()
        self.__init_activation_hooks(nets)

    def __getitem__(self, item):
        return self.__hidden_outputs[item]

    def __get_neurons_to_cover(self):
        for coverage_table in self.__coverage_tables:
            neuron_to_cover = _get_neuron_to_cover(coverage_table)
            self.__neurons_to_cover.append(neuron_to_cover)

    def __init_activation_hooks(self, nets):
        for net in nets:
            flatten_net = list(_flatten_net(net))
            hidden_outputs = defaultdict(torch.Tensor)
            for module_name, module in flatten_net:
                module.register_forward_hook(_HiddenOutput(hidden_outputs, module_name))
            self.__hidden_outputs.append(hidden_outputs)

    @property
    def coverage_tables(self):
        return self.__coverage_tables

    @property
    def hidden_outputs(self):
        return self.__hidden_outputs

    @property
    def neurons_to_cover(self):
        return self.__neurons_to_cover
