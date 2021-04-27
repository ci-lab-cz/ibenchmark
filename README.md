# iBenchmark

**iBenchmark** is a collection of  datasets and performance metrics (implemented in python) for benchmarking structural interpretation of QSAR models.

### Description of datasets
There are six synthetic datasets with pre-defined patterns determining end-point values and with control over possible biases. Regression and classification datasets represent different scenarios related to the following cases:  
- simple additive end-points on per-atom basis
- additive properties depending on local chemical context
- pharmacophore-like setting, where the property depends on distant features of a particular type, i.e. “global” context.

These data sets are purposed for evaluation of interpretation approaches which estimate atom or fragment contributions (attributions) by comparing calculated contributions of atoms or fragments with expected values. The source of compounds was the ChEMBL23 database. Structures were standardized and duplicates were removed. End-point values were assigned to compounds according to rules defined below. Compounds were randomly sampled from whole ChEMBL set until the desired data set size was achieved. Distribution of end-point values was controlled for being balanced (classification) or close to normal (regression). All atoms in each molecule were assigned “ground truth” labels, i.e. expected contributions.

**N dataset**. The end-point was the sum of nitrogen atoms. Thus, the expected contributions of nitrogen atoms were 1 and all other atoms - 0.  

**N-O dataset**. The end-point was the sum of nitrogen atoms minus the sum of oxygen atoms. Thus, oxygen represented a negatively contributing pattern. Expected contribution of any nitrogen was 1, any oxygen -1, and all others 0.  

**N+O dataset**. The end-point was the sum of nitrogen and oxygen atoms divided by two. The number of nitrogen and oxygen atoms in each molecule was strictly equal. Thus, two positively contributing patterns were co-occurring and both contributed equally to the target property. This represents a specific case to verify how a model treats correlated patterns and how this affects interpretation output.  

**Amide dataset** represented additive end-point depending on local chemical context. The end-point  was the number of amide groups encoded with SMARTS NC=O.  This end-point was similar to properties like lipophilicity, polar surface area, etc. This was a regression task.  

**Amide_class dataset** was a classification one, where compounds were assigned active if they had at least one amide pattern and inactive otherwise. The expected contribution of any atom of an amide group for either data set was 1, because upon removing of such an atom the whole pattern disappears.  

**Pharmacophore dataset** was designed based on a pharmacophore hypothesis and represents property, depending on whole-molecule context. Compounds were labeled as active if at least one of their conformers had a pair of an H-bond donor and an H-bond acceptor 9-10 Å apart. If the pattern occurred in more than one conformer of a molecule, this had to be the same pair of atoms. Therefore, actives contained the same pharmacophore pair consistent across all conformers. If this pattern was absent in all conformers a compound was labeled inactive. We generated up to 25 conformers for each compound using RDKit.

### Description of metrics.py  
A command line python tool to calculate performance of QSAR model interpretation based on the calculated **atom** contributions.  

### Performance metrics
All metrics below can be calculated for each molecule and for the whole dataset. In the latter case values are aggregated across individual molecules. Metrics (where applicable) can be computed in two modes: for positively contributing atoms (hereafter positive atoms) and for negatively contributing atoms (hereafter negative atoms).

**ROC-AUC**. Interpretation method is treated as a binary classifier: *predictor* is calculated contribution, *observed* is ground truth.
For *AUC_positive* ground truth label is set to 1 for positive atoms and 0 for the rest.
For *AUC_negative* ground truth label is set to -1 for positive atoms and 0 for the rest. Aggregation method for the whole dataset is mean.

**Top_n**. Overall number of positive atoms in *top n*  atoms sorted by contribution divided by n. N can be set by the user to any integer. In this case metric name should be specified as *"Top_3"*, *"Top_5"* etc. If not specified, *n* varies according to the number of positive atoms in each molecule.  
E.g.  for a molecule with 2 positive atoms *n = 2*. This metric is aggregated in a  cumulative  way: sum over all numerators for molecules is dividded by sum over denominators.

**Bottom_n** defined analogously to *top_n* but performance is calculated for negatively contributed atoms.

**RMSE**. *Root mean squared error* between observed (ground truth) and predicted (contributions) values. Aggregated by *mean*.

## Examples
To perform metrics calculation for the whole dataset, run the script with your files, for example:

python metrics.py  --contrib_fname example_notebook_data/contrib_per_atom_dc.txt   --sdf_fname  example_notebook_data/N_train_lbl.sdf --contrib_col contribution  --lbls_field lbls --metrics AUC_positive --output_fname example_notebook_data/out.txt

To perform metrics calculation per molecule, add key --per_molecule_metrics_fname, e.g:
python metrics.py  --contrib_fname example_notebook_data/contrib_per_atom_dc.txt   --sdf_fname  example_notebook_data/N_train_lbl.sdf --contrib_col contribution  --lbls_field lbls --metrics AUC_positive --output_fname example_notebook_data/out.txt --per_molecule_metrics_fname example_notebook_data/per_mol_out.txt

Alternatively, you can use functions from the script in your own pipelines, example is given in *metrics_for_individual_molecules.ipynb*
