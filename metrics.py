#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from rdkit import Chem
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, mean_squared_error
import argparse
import re
from utils import read_contrib_spci


def calc_auc(merged_df,
             which_lbls=("positive", "negative"),
             contrib_col_name="contrib",
             lbl_col_name="lbl"):
    # auc will be calculated only for  mols having lbld atoms, otherwise -1 returned
    def auc_wrapper(y_true, y_score, inverse=False):
        if len(set(y_true)) > 1:
            if not inverse: return roc_auc_score(y_true, y_score)
            else: return 1 - roc_auc_score(y_true, y_score)
        else:
            return -1

    res = {}
    if "positive" in which_lbls:
        res["auc_pos"] = merged_df.groupby(by="molecule").apply(lambda gr: auc_wrapper(y_true=gr[lbl_col_name]>0,
                                                                          y_score=gr[contrib_col_name]))
    if "negative" in which_lbls:
        res["auc_neg"] = merged_df.groupby(by="molecule").apply(lambda gr: auc_wrapper(y_true=gr[lbl_col_name]<0,
                                                                          y_score=gr[contrib_col_name],inverse=True))
    return res


def merge_lbls_contribs(contribs, lbls, lbl_col_name="lbl"):
    merged_df = pd.merge(
        contribs, lbls,
        how="inner")
    # next lines (left join) potentially lead to incorrect rmse if setdiff(mols_from_contribs, mols_from_sdf)>0!
    # because if SOME ATOMS of given mol aren't in sdf - we mustnot use their contribs, they maybe nonexisitng atoms!
    # however, in this case (inner join) only molecules with labels for all atoms, ie not NA,will be used
        # how="left")  # left join: all atoms with contribs will be used
    # merged_df.loc[
    #     pd.isna(merged_df[lbl_col_name]),
    #     lbl_col_name] = 0  # set zero lbl to atoms missing in ids table

    return merged_df

def read_contrib(contrib, sep=","):
    contrib = pd.read_csv(contrib,sep=sep)
    return contrib


# version with  averging of individual molecules expected p
# smapling of atoms  with replacement is accounted for, but it doesnt change formula for variable top_n logic,
# because n=K for hypergeometric distr (see wiki)
# def calc_baseline(merged_df, top=True, bottom=True, lbl_col_name="lbl", act_field_name=None):
#
#
#
#     res = {}
#     if act_field_name is not None:
#         mask = merged_df[act_field_name]
#     else:
#         mask = [True] * merged_df.shape[0]
#         print("Warning: baseline will be calculated for all molecules, may be incorrect when some mols dont have any labelled atoms!")
#     if top:
#         res["baseline_top"] = np.mean(merged_df.loc[mask,:].groupby(by="molecule").apply(lambda gr:sum(gr[lbl_col_name] > 0) / len(
#             gr[lbl_col_name])))
#     if bottom:
#         res["baseline_bottom"] = np.mean(merged_df.loc[mask,:].groupby(by="molecule").apply(lambda gr:sum(gr[lbl_col_name] < 0) / len(
#             gr[lbl_col_name])))
#     return res

# version with flat baseline calulation, less correct then averging of individual molecules p.
# smapling of atoms  with replacement is accounted for explicitly
def calc_baseline(merged_df, top=True, bottom=True, lbl_col_name="lbl", act_field_name=None):
    res = {}

    if act_field_name is not None:
        mask = merged_df[act_field_name]
    else:
        mask = [True] * merged_df.shape[0]
        print("Warning: baseline will be calculated for all molecules, may be incorrect when some mols dont have any labelled atoms!")

    if top:
        # res = sum(n*k/N) / sum(n), where n=k number of true atoms in a mol, N=number of atoms in a mol
         res["baseline_top"] = np.sum(merged_df.loc[mask, :].groupby(by="molecule").apply(lambda gr: (sum(gr[lbl_col_name] > 0)**2 )/ len(
                      gr[lbl_col_name]))) /(np.sum(merged_df.loc[mask][lbl_col_name] > 0)+0.0000001) # zerodivision corr.
    if bottom:
         res["baseline_bottom"] = np.sum(merged_df.loc[mask, :].groupby(by="molecule").apply(lambda gr: (sum(gr[lbl_col_name] < 0)**2 )/ len(
                      gr[lbl_col_name]))) / (np.sum(merged_df.loc[mask][lbl_col_name] < 0)+0.0000001)# zerodivision corr.
    return res

def summarize(data):
    res = {}
    for d in data:
        for i, v in d.items():
            if i in ("auc_pos", "auc_neg", "rmse"):
                res[i] = round(np.mean(v[v != -1]), 2)
            elif "baseline" in i:
                res[i] = round(v, 2)
            elif (("top" in i) and ("select" not in i)):
                res[i] = round(
                    sum(v.top_score * v.top_sum) / sum(v.top_sum), 2)
            elif (("bottom" in i) and ("select" not in i)):
                res[i] = round(
                    sum(v.bottom_score * v.bottom_sum) / sum(v.bottom_sum), 2)
    return res


def calc_rmse(merged_df, contrib_col_name="contrib", lbl_col_name="lbl"):
    return {"rmse":merged_df.groupby(by="molecule").apply(lambda gr: np.sqrt(mean_squared_error(gr[lbl_col_name], gr[contrib_col_name])))}


def read_lbls_from_sdf(input_sdf, lbls_field_name="lbls",act_field_name=None, sep=",", find_eq=True):

    def ranks(m):
        r = []
        for i,j in enumerate(list(Chem.CanonicalRankAtoms(m, breakTies=False, includeChirality=False,includeIsotopes=False))):
            r.append([mol.GetProp("_Name"), i+1,j]) # 1based!
        return r

    # ids are one based
    res = []  # 1 based
    rnks_all = []
    sdf = Chem.SDMolSupplier(input_sdf)
    for mol in sdf: # loop over mols
        if mol is not None:
            rnks = ranks(mol)
            rnks_all.extend(rnks)
            props = mol.GetPropsAsDict()
            if act_field_name is not None: # readout activity T/F for this mol
                if act_field_name in props:
                    act_tmp = bool(int(props[act_field_name]))
                else:
                    print("warning: bad activity field name.SDF was not read")
                    return None

            if lbls_field_name in props: # readout lblbs for this mol
                lbls = str(props[lbls_field_name]).split(
                    sep)  # convert to str to use split

                if (lbls[0] != "NA"):
                    for i, j in enumerate(lbls): # loop over atoms and add to big list their labels with mol name and act
                        res.append([mol.GetProp("_Name"), i + 1, # 1-based
                             int(j), act_tmp])

            else:
                print("warning: bad labels field_name.SDF was not read")
                return None

    res = pd.DataFrame(res)

    if len(res.columns) == 3: res.columns = ["molecule", "atom", "lbl"]
    if len(res.columns) == 4: res.columns = ["molecule", "atom", "lbl", act_field]
    if rnks_all:
        res = pd.merge(res,pd.DataFrame(rnks_all, columns=["molecule", "atom", "rank"]))
    # print(res.head(55))
    return res

def calc_top_n(merged_df,
               n_list=(-np.inf, np.inf, 3, 5, -3, -5),
               contrib_col_name="contrib",
               lbl_col_name="lbl"):
    # n_list  -inf means variable (adjustable) top n (minus inf - bottom n).
    # values other than +-inf determine fixed n, but for atoms (unlike fragments)
    # they are not recommended (if chosen, then fixed n will be used,
    # unless if overall number of positive/negative atoms in mol is less than n, min(overall n,n) will be used then
    # negative n means "calculate bottom n".  Ties for n are broken according param "keep" in nlargest/nsmallest
    def get_summary(df, sgn=True, lbl_col_name=lbl_col_name):
        if sgn:
            df = df.groupby(by="molecule").agg({
                lbl_col_name:
                lambda l: sum(l > 0) / len(l),
                "molecule":
                lambda m: m.shape[0]
            })
            df.columns = ["top_score", "top_sum"]
        else:
            df = df.groupby(by="molecule").agg({
                lbl_col_name:
                lambda l: sum(l < 0) / len(l),
                "molecule":
                lambda m: m.shape[0]
            })
            df.columns = ["bottom_score", "bottom_sum"]
        return df

    res = {}
    merged_df[
        "top_sum"] = merged_df.loc[:, ["molecule", lbl_col_name]].groupby(
            by="molecule").transform(lambda x: sum(x > 0))[lbl_col_name]
    merged_df[
        "bottom_sum"] = merged_df.loc[:, ["molecule", lbl_col_name]].groupby(
            by="molecule").transform(lambda x: sum(x < 0))[lbl_col_name]

    for n in n_list:

        if n == np.inf:
            res["select_top_n"] = merged_df.groupby(by="molecule").apply(lambda gr: gr.nlargest(int(gr.top_sum.unique()),contrib_col_name, keep="all"))
            res["select_top_n"].reset_index(drop=True, inplace=True)
            # add  new summarized one
            res["variable_top_n"] = get_summary(res["select_top_n"], True)
        elif n == -np.inf:
            res["select_bottom_n"]  = merged_df.groupby(by="molecule").apply(lambda gr: gr.nsmallest(int(gr.bottom_sum.unique()),contrib_col_name, keep="all"))
            res["select_bottom_n"].reset_index(drop=True, inplace=True)
            #add new summarized one
            res["variable_bottom_n"] = get_summary(res["select_bottom_n"],
                                                   False)
        elif n > 0:
            res["select_top_sum"+str(n)] = merged_df.groupby(by="molecule").apply(lambda gr: gr.nlargest(int(min(n,gr.top_sum.unique())),
                                         contrib_col_name, keep="all"))
            res["select_top_sum" + str(n)].reset_index(drop=True, inplace=True)
            res["top" + str(n)] = get_summary(
                res["select_top_sum" + str(n)], True)
        elif n < 0:
            res["select_bottom_sum"+str(-n)] = merged_df.groupby(by="molecule").apply(lambda gr: gr.nsmallest(int(min(-n, gr.bottom_sum.unique())),
                                          contrib_col_name, keep="all"))
            res["select_bottom_sum" + str(-n)].reset_index(
                drop=True, inplace=True)
            res["bottom" + str(-n)] = get_summary(
                res["select_bottom_sum" + str(-n)], False)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Calculate  performance of QSAR model interpretation.
                        Applicable to any method of interpretation/attribution/explanation which produces results in the form of contributions of atoms (fragments) in a given molecule.
                        Informally, interpretation performance here means closeness of atom contributions to expected/"ground truth"
                        values. For instance, when an atom is important for molecule's activity, its expected contribution is
                        non-zero. The exact "ground truth" value is, of course, defined by the user, for classification tasks it's typically 1 for important atoms and 0 for the rest;
                        for regression though it depends on quantitative impact of an atom. The value can also be negative. (See description)'''
    )
    parser.add_argument(
        '--contrib_fname',
        metavar='contrib.txt',
        required=True,
        help=
        'File name (with full path) for contributions.Should contain at least these columns (named): "molecule", "atom", "contribution". Atom: '
        '1-based atom numbers; must be consistent with order of atoms in sdf. Molecule: molecule name. Contribtuion: numeric value of atom contributions '
    )
    parser.add_argument(
        '--sdf_fname',
        metavar='molecules_with_labels.sdf',
        required=True,
        help=
        'File name (with full path) for sdf with molecules. Should contain molecule title and field with atom ground truth labels (expected contributions)'
    )
    parser.add_argument(
        '--contrib_col',
        metavar='contrib',
        required=False,
        default="contribution",
        help='Column name in contributions file, where contributions are given'
    )
    parser.add_argument(
        '--remove_eq',
        required=False,
        default=0,
        help='Should equivalent atoms (by Canonical rank without chirality/isotope considerations) be filtered out (keep only one)?'
    )
    parser.add_argument(
        '--lbls_field',
        metavar='lbls',
        required=False,
        default="lbls",
        help=
        'field name in sdf file, where ground truth labels of all atoms are given (without explicit atom numbers, atom order must hold)'
    )

    parser.add_argument(
        '--act_field',
        metavar='act',
        required=False,
        default=None,
        help=
        'field name in sdf file, where activities (int type) of molecules are given (optional).Needed only'
        'to correctly calculate baseline'
    )
    parser.add_argument(
        '-sep_for_lbls',
        metavar=',',
        required=False,
        default=",",
        help='separator for lbls in sdf field (default comma)')
    parser.add_argument(
        '--metrics',
        metavar='AUC_positive AUC_negative Top_n Bottom_n RMSE',
        required=False,
        default=("AUC_positive", "Top_n", "RMSE"),
        nargs='*',
        help='Which metrics to compute? Allowed names are: '
        'AUC_positive, AUC_negative, Top_..., Bottom_..., RMSE , where "..."  be replaced by either "n" or any integer. "n" (recommended) means adjustable top/bottom n (see description)'
    )
    parser.add_argument(
        '--output_fname',
        metavar='out.txt',
        required=True,
        help='output file name (with path)')
    parser.add_argument(
        '--per_molecule_metrics_fname',
        required=False,
        metavar='per_mol.txt',
        default=None,
        help='Should metrics for each molecule be returned, or only aggregated values per dataset? If yes,please provide a filename')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "contrib_fname": contrib_fname = v
        if o == "sdf_fname": sdf_fname = v
        if o == "contrib_col": contrib_col = v
        if o == "remove_eq": remove_eq = bool(int(v))
        if o == "lbls_field": lbls_field = v
        if o == "act_field": act_field = v
        if o == "sep_for_lbls": sep_for_lbls = v
        if o == "metrics": metrics = v
        if o == "output_fname": output_fname = v
        if o == "per_molecule_metrics_fname": per_mol_fname = v

    lbls = read_lbls_from_sdf(
        sdf_fname, lbls_field_name=lbls_field,act_field_name=act_field, sep=sep_for_lbls, find_eq=remove_eq)
    contribs = read_contrib(contrib_fname)
    # contribs = read_contrib_spci(contrib_fname)["overall"] # spci input
    merged = merge_lbls_contribs(contribs, lbls)
    if remove_eq:
        merged = merged.drop_duplicates(["molecule", "rank"])
    # print(merged.head(55))
    auc_ind = sum(["AUC" in i for i in metrics])
    top_ind = sum([("Top" in i or "Bottom" in i) for i in metrics])
    rmse_ind = ("RMSE" in metrics)
    metr = []

    if auc_ind:
            which_lbls = [i[4:] for i in metrics if "AUC" in i]
            auc = calc_auc(
                merged, which_lbls=which_lbls, contrib_col_name=contrib_col)
            print("calculated auc")
            metr.append(auc)
    if rmse_ind:
            rmse = calc_rmse(merged, contrib_col_name=contrib_col)
            print("calculated rmse")
            metr.append(rmse)
    if top_ind:
            n_list_1 = [i[4:] for i in metrics if "Top" in i]
            n_list_1 = [float(i) if i != "n" else np.inf for i in n_list_1]
            n_list_2 = [i[7:] for i in metrics if "Bottom" in i]
            n_list_2 = [-float(i) if i != "n" else -np.inf for i in n_list_2]
            n_list_1.extend(n_list_2)

            baseline = calc_baseline(merged, act_field_name=act_field)
            top_n = calc_top_n(
                merged, n_list=n_list_1, contrib_col_name=contrib_col)
            print("calculated baseline, top_n (bottom_n)")
            metr.extend([baseline, top_n])
    if per_mol_fname is not None:
        fin = None
        # print(len(metr))
        for i in metr:

            for k,v in i.items():
                if ("select" not in k) and ("baseline" not in k):
                    if fin is  None:
                        if isinstance(v, pd.DataFrame):
                            v = v.iloc[:, 0]  # case when v is not series but DF : drop top_sum
                        fin = pd.Series(v, name=k)
                    else:
                        # print(type(v))
                        if isinstance(v, pd.DataFrame):
                            v = v.iloc[:,0] # case when v is not series but DF : drop top_sum
                        # print(k,v)
                        tmp = pd.Series(v, name=k)
                        fin = pd.merge(fin, tmp, how="outer", right_index=True, left_index=True)
                        # print(fin)
        pd.DataFrame(fin).to_csv(per_mol_fname, sep = "\t")
    res = summarize(metr)
    pd.DataFrame(res, index=[0]).to_csv(output_fname, sep = "\t")