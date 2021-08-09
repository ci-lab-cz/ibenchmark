#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from rdkit import Chem
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
import re

def read_contrib_spci(fname,
                      model_names=("gbm", "svm", "rf", "pls"),
                      contr_names="overall",
                      mol_frag_sep="###",
                      frag=False,
                      filter_rel_frag_size=1): # keep all frags regardless of size
    d = defaultdict(dict)
    res = {}

    with open(fname) as f:

        names = f.readline().strip().split('\t')[1:]

        frag_names = []
        mol_names = []
        FragUID = []
        for n in names:
            mol_name, frag_name = n.split(mol_frag_sep)
            frag_names.append(frag_name)
            mol_names.append(mol_name)

        for i, v in enumerate(frag_names):
            FragUID.append(re.search("#\d+$", frag_names[i]).group(0)[1:]) # numeric after last #
            frag_names[i] = re.sub("#\d+$", "", frag_names[i])  # value without last # & numeric

        for line in f:
            tmp = line.strip().split('\t')
            if tmp[0] != "relative_frag_size":
                model_name, prop_name = tmp[0].rsplit("_", 1)
                # skip contributions which are not selected
                if prop_name not in contr_names:
                    continue
                if "all" in model_names or model_name in model_names:
                    values = list(map(float, tmp[1:]))
                    # filter out values by keep_ids
                    values = [values[i] for i in keep_ids]
                    d[prop_name][model_name] = values
            else: # line with relative frag size
                if filter_rel_frag_size < 1:
                    rel_frag_size =  list(map(float, tmp[1:]))
                    rel_frag_size = [rel_frag_size[i] for i in keep_ids]

        for i, v in d.items():
            res[i] = pd.DataFrame(v)
            if not frag:
                res[i]["atom"] = list(map(int, frag_names))
            else:
                res[i]["frag"] = list(frag_names)
                res[i]["FragUID"] = list(map(int, FragUID))
                if filter_rel_frag_size<1:
                    res[i]["rel_frag_size"] = rel_frag_size

            res[i]["molecule"] = mol_names
            if filter_rel_frag_size < 1:
                res[i] = res[i][res[i]["rel_frag_size"] <= filter_rel_frag_size]
    return res