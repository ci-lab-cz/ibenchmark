# InterpretBenchmark

**InterpretBenchmark** is a collection of  datasets and performance metrics (implemented in python) for benchmarking atom-based interpretation of QSAR models.

**Description of datasets**

**Description of metrics.py**
A command line python tool to evaluate performance of QSAR model interpretation.  
Applicable to any method of interpretation/attribution/explanation which produces results in the form of contributions of atoms (fragments) in a given molecule.
Informally, interpretation performance here means closeness of atom contributions to expected/"ground truth"
values. For instance, when an atom is important for molecule's activity, its expected contribution is
non-zero. The exact "ground truth" value is, of course, defined by the user; for classification tasks it's typically 1 for important atoms and 0 for the rest;
for regression though it depends on quantitative impact of an atom. (Example: Let's consider  3-atomic molecule with activity=10 units, then  
expected atom contributions can be e.g. 0,5,5 units.)
**Performance metrics**
All below metrics can be calculated for each molecule and for the whole dataset.
In the latter case values are aggregated for the whole dataset, using mean or other method (see below).
Metrics (where applicable) can be computed  in two modes (for 2 types of atoms):
for positively contributing atoms (hereafter positive atoms) and for negatively contributing atoms
(hereafter negative atoms). Positive atoms are atoms which increase molecules' activity, or favor positive class prediction. (Negative atoms - conversely.)

**ROC-AUC**.  Attribution method is treated as a binary classifier: predictor is contribution, observed is
ground_truth. ROC AUC is then computed. For AUC_positive   ground truth label is set to 1 for positive atoms and 0 for the rest.
 When computed for negative atoms (AUC_negative): negative atoms label set to -1, and the rest to 0. For the whole dataset aggregated using mean.

**Top_n**. Overall number of positive atoms in top n  atoms sorted by contribution divided by n. N can be  set by the user to any sensible integer. In this case metric name should be specified as "Top_3", "Top_5" etc. If not specified, n varies according to the number of positive atoms in each given molecule.  
E.g.  for a molecule with 2 positive atoms n = 2. This metric is aggregated in a  cumulative  way: sum over all numerators for molecules is dividded by sum over denominators.
**Bottom_n**. Defined analogously to top_n. Sorting order is inverse.

**RMSE**. RMSE between observed (ground truth) and predicted (contributions) values. Aggregated using mean.
