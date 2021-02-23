Description of  fields present in every SDF:
activity - target activity. (Some  other fields e.g. "class" can duplicate its value.)
ids (ids_n, ids_o etc.)  -  1-based ids of important atoms (N, O etc.)
lbls (lbls_n, lbls_o etc.) - another format to represent information contained in ids. More convenient for metrics calculation. 
"0" = non-important atom, "1"  = positively contributing atom; "-1" = negatively contributing atom. 
