# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os

def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./results/NTU/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')
    elif dataset == 'NTU_ID':
        output_dir = os.path.join('./results/NTU_ID/')
    elif dataset == 'NTU_PRIM':
        output_dir = os.path.join('./results/NTU_PRIM/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60
    elif dataset == 'NTU120':
        return 120
    elif dataset == 'NTU_ID':
        return 40
    elif dataset == 'NTU_PRIM':
        return 32

    
