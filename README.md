# InterpretBenchmark

**InterpretBenchmark** is a collection of  datasets and performance metrics (implemented in python) for benchmarking atom-based interpretation of QSAR models.

**Description of datasets**
There are six synthetic datasets with pre-defined patterns being end-point values and with control over possible biases. Regression and classification data sets represent different scenarios related to the following cases: i) simple additive end-points on per-atom basis, ii) additive properties depending on local chemical context and iii) pharmacophore-like setting, where the property depends on distant features of particular type, i.e. “global” context. These data sets are purposed for evaluation of any interpretation approach which estimates atom or fragment contributions (attributions), because calculated contributions of any atom or fragment can be compared with expected values. The source of compounds was the ChEMBL23 database. Structures were standardized and duplicates were removed. End-point values were assigned to compounds according to rules defined below. Compounds were randomly sampled until the desired data set size was achieved. Distribution of end-point values was controlled for being balanced (classification) or close to normal (regression). All atoms in each molecule were assigned “ground truth” labels, i.e. expected contributions, to compare them with the calculated contributions.
*N dataset*. The end-point was the sum of nitrogen atoms. Thus, the expected contributions of nitrogen atoms were 1 and all other atoms - 0.
*N-O dataset*. The end-point was the sum of nitrogen atoms minus the sum of oxygen atoms. Thus, oxygen represented a negatively contributing pattern. Expected contribution of any nitrogen was 1, any oxygen -1, and all others 0.
*N+O dataset*. The end-point was the sum of nitrogen and oxygen atoms divided by two. The number of nitrogen and oxygen atoms in a molecule was strictly equal. Thus, two positively contributing patterns were co-occurring and both contributed equally to the target property. This represents a specific case to verify how a model treats correlated patterns and how this affects interpretation output.
*Amide dataset* represented additive end-point depending on local chemical context. The end-point  was the number of amide groups encoded with SMARTS NC=O.  This end-point was similar to properties like lipophilicity, polar surface area, etc. This was a regression task.
*Amide_class dataset* was a classification one, where compounds were assigned active if they had at least one amide pattern and inactive otherwise. The expected contribution of any atom of an amide group for either data set was 1, because upon removing of such an atom the whole pattern disappears.
*Pharmacophore dataset* was designed based on a pharmacophore hypothesis and represents property, depending on whole-molecule context. Compounds were labeled as active if at least one of their conformers had a pair of an H-bond donor and an H-bond acceptor 9-10 Å apart. If the pattern occurred in more than one conformer of a molecule, this had to be the same pair of atoms. Therefore, actives contained the same pharmacophore pair consistent across all conformers. If this pattern was absent in all conformers a compound was labeled inactive. We generated up to 25 conformers for each compound using RDKit.


**Description of metrics.py**
A command line python tool to evaluate performance of QSAR model interpretation.  
Applicable to any method of interpretation/attribution/explanation which produces results in the form of contributions of atoms (fragments) in a given molecule.
Informally, interpretation performance here means closeness of atom contributions to expected/"ground truth"
values. For instance, when an atom is important for molecule's activity, its expected contribution is
non-zero. The exact "ground truth" value is, of course, defined by the user; for classification tasks it's typically 1 for important atoms and 0 for the rest;
for regression though it depends on quantitative impact of an atom. For example, let's consider  3-atomic molecule with activity=10 units, then  expected atom contributions can be e.g. 0, 5, 5 units.

**Performance metrics**
All below metrics can be calculated for each molecule and for the whole dataset.
In the latter case values are aggregated for the whole dataset, using mean or other method (see below).
Metrics (where applicable) can be computed  in two modes (for 2 types of atoms):
for positively contributing atoms (hereafter positive atoms) and for negatively contributing atoms
(hereafter negative atoms). Positive atoms are atoms which increase molecules' activity, or favor positive class prediction. (Negative atoms - conversely.)

**ROC-AUC**.  Attribution method is treated as a binary classifier: *predictor* is contribution, *observed* is
ground_truth. ROC AUC is then computed. For *AUC_positive*   ground truth label is set to 1 for positive atoms and 0 for the rest.
For *AUC_negative* ground truth label is set to -1 for positive atoms and 0 for the rest. Aggregation method for the whole dataset is mean.

**Top_n**. Overall number of positive atoms in *top n*  atoms sorted by contribution divided by n. N can be  set by the user to any sensible integer. In this case metric name should be specified as *"Top_3"*, *"Top_5"* etc. If not specified, *n* varies according to the number of positive atoms in each molecule.  
E.g.  for a molecule with 2 positive atoms *n = 2*. This metric is aggregated in a  cumulative  way: sum over all numerators for molecules is dividded by sum over denominators.
**Bottom_n**. Defined analogously to *top_n*. Sorting order is inverse.

**RMSE**. *Root mean squared error* between observed (ground truth) and predicted (contributions) values. Aggregated by *mean*.
