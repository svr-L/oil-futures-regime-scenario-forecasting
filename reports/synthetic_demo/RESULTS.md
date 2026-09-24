# Rolling pseudo-OOS results (simulated data)

- origins used: **85** (dropped: 0)
- horizons: [1, 4, 12, 20] weeks, stride 4 weeks, 1500 scenarios per forecast
- models: RW_equal, AR_equal, AR_time, AR_macro, AR_macro_nc, VECM_equal, VECM_time, VECM_macro, FHS_equal, FHS_time, FHS_macro, CTS_equal, CTS_time, CTS_macro, CTSF_equal, CTSF_time, CTSF_macro, VARL_equal, VARL_time
- baseline for all comparisons: `RW_equal`
- balanced panel: **True**

All models are scored on identical origins with common random numbers. p-values are Diebold-Mariano with Newey-West HAC variance at the overlap lag implied by horizon and stride, plus the HLN small-sample correction.

## Mean CRPS (lower is better)

### BRENT

| model       |       1 |       4 |      12 |      20 |
|:------------|--------:|--------:|--------:|--------:|
| AR_equal    | 0.0196  | 0.03793 | 0.06265 | 0.07987 |
| AR_macro    | 0.01956 | 0.03793 | 0.06264 | 0.08019 |
| AR_macro_nc | 0.01958 | 0.03807 | 0.06391 | 0.07992 |
| AR_time     | 0.01961 | 0.03813 | 0.06278 | 0.0802  |
| CTSF_equal  | 0.01935 | 0.03769 | 0.06119 | 0.0784  |
| CTSF_macro  | 0.01942 | 0.03768 | 0.0612  | 0.07839 |
| CTSF_time   | 0.01951 | 0.03788 | 0.06125 | 0.07828 |
| CTS_equal   | 0.01935 | 0.03794 | 0.06181 | 0.07935 |
| CTS_macro   | 0.01943 | 0.03795 | 0.06183 | 0.07934 |
| CTS_time    | 0.01952 | 0.03814 | 0.06188 | 0.07924 |
| FHS_equal   | 0.01944 | 0.03739 | 0.06183 | 0.07946 |
| FHS_macro   | 0.01946 | 0.03759 | 0.0622  | 0.07993 |
| FHS_time    | 0.01951 | 0.03766 | 0.06225 | 0.07989 |
| RW_equal    | 0.01968 | 0.03737 | 0.0616  | 0.07715 |
| VARL_equal  | 0.01936 | 0.03776 | 0.06085 | 0.07598 |
| VARL_time   | 0.01945 | 0.03806 | 0.06096 | 0.07586 |
| VECM_equal  | 0.01964 | 0.03779 | 0.06055 | 0.07526 |
| VECM_macro  | 0.01953 | 0.03767 | 0.06039 | 0.07547 |
| VECM_time   | 0.01959 | 0.03792 | 0.06069 | 0.0754  |

### SPREAD

| model       |       1 |       4 |      12 |      20 |
|:------------|--------:|--------:|--------:|--------:|
| AR_equal    | 0.01132 | 0.01858 | 0.02816 | 0.03304 |
| AR_macro    | 0.01134 | 0.01854 | 0.02818 | 0.03299 |
| AR_macro_nc | 0.01144 | 0.01896 | 0.02979 | 0.03589 |
| AR_time     | 0.01137 | 0.01854 | 0.02798 | 0.03307 |
| CTSF_equal  | 0.01064 | 0.0164  | 0.02052 | 0.02108 |
| CTSF_macro  | 0.01063 | 0.01637 | 0.0205  | 0.02116 |
| CTSF_time   | 0.01062 | 0.01637 | 0.02042 | 0.02117 |
| CTS_equal   | 0.01075 | 0.01626 | 0.01984 | 0.02022 |
| CTS_macro   | 0.01077 | 0.01626 | 0.01984 | 0.02034 |
| CTS_time    | 0.01075 | 0.01625 | 0.01976 | 0.02035 |
| FHS_equal   | 0.01163 | 0.01907 | 0.0291  | 0.03442 |
| FHS_macro   | 0.01163 | 0.019   | 0.02905 | 0.03439 |
| FHS_time    | 0.01167 | 0.01905 | 0.02888 | 0.03455 |
| RW_equal    | 0.01142 | 0.01847 | 0.02748 | 0.03244 |
| VARL_equal  | 0.01084 | 0.01674 | 0.02057 | 0.021   |
| VARL_time   | 0.01083 | 0.01683 | 0.0206  | 0.02103 |
| VECM_equal  | 0.01093 | 0.01682 | 0.02058 | 0.02091 |
| VECM_macro  | 0.01092 | 0.01687 | 0.02072 | 0.02098 |
| VECM_time   | 0.01092 | 0.01684 | 0.0206  | 0.02096 |

### WTI

| model       |       1 |       4 |      12 |      20 |
|:------------|--------:|--------:|--------:|--------:|
| AR_equal    | 0.01941 | 0.03792 | 0.06513 | 0.08282 |
| AR_macro    | 0.01952 | 0.03778 | 0.06502 | 0.0828  |
| AR_macro_nc | 0.01955 | 0.038   | 0.06574 | 0.08189 |
| AR_time     | 0.01958 | 0.03796 | 0.06529 | 0.08297 |
| CTSF_equal  | 0.01931 | 0.03765 | 0.06538 | 0.08348 |
| CTSF_macro  | 0.01947 | 0.03745 | 0.06513 | 0.08339 |
| CTSF_time   | 0.01945 | 0.0376  | 0.06521 | 0.0834  |
| CTS_equal   | 0.01932 | 0.03739 | 0.06467 | 0.08249 |
| CTS_macro   | 0.01947 | 0.03718 | 0.06442 | 0.08238 |
| CTS_time    | 0.01946 | 0.03734 | 0.0645  | 0.0824  |
| FHS_equal   | 0.01905 | 0.03681 | 0.06383 | 0.0816  |
| FHS_macro   | 0.01919 | 0.0367  | 0.06363 | 0.08133 |
| FHS_time    | 0.01923 | 0.0367  | 0.06363 | 0.08114 |
| RW_equal    | 0.01958 | 0.03737 | 0.06321 | 0.07897 |
| VARL_equal  | 0.0194  | 0.03712 | 0.0634  | 0.07992 |
| VARL_time   | 0.01949 | 0.03714 | 0.06354 | 0.07999 |
| VECM_equal  | 0.01952 | 0.03717 | 0.06325 | 0.07918 |
| VECM_macro  | 0.01956 | 0.0369  | 0.0632  | 0.07935 |
| VECM_time   | 0.0196  | 0.0371  | 0.06338 | 0.07933 |

## Model Confidence Set (90%)

How many models survive per cell. Where most of them survive, the
data cannot separate the models and no winner should be claimed.

| variable   |   horizon_weeks |   n_models |   n_in_mcs_90 | best_model   | mcs_90_members                                                                                                                                                                                                  |
|:-----------|----------------:|-----------:|--------------:|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BRENT      |               1 |         19 |            19 | CTS_equal    | CTS_equal, CTSF_equal, VARL_equal, CTSF_macro, CTS_macro, FHS_equal, VARL_time, FHS_macro, CTSF_time, FHS_time, CTS_time, VECM_macro, AR_macro, AR_macro_nc, VECM_time, AR_equal, AR_time, VECM_equal, RW_equal |
| BRENT      |               4 |         19 |            19 | RW_equal     | RW_equal, FHS_equal, FHS_macro, FHS_time, VECM_macro, CTSF_macro, CTSF_equal, VARL_equal, VECM_equal, CTSF_time, VECM_time, AR_macro, AR_equal, CTS_equal, CTS_macro, VARL_time, AR_macro_nc, AR_time, CTS_time |
| BRENT      |              12 |         19 |            19 | VECM_macro   | VECM_macro, VECM_equal, VECM_time, VARL_equal, VARL_time, CTSF_equal, CTSF_macro, CTSF_time, RW_equal, CTS_equal, CTS_macro, FHS_equal, CTS_time, FHS_macro, FHS_time, AR_macro, AR_equal, AR_time, AR_macro_nc |
| BRENT      |              20 |         19 |            19 | VECM_equal   | VECM_equal, VECM_time, VECM_macro, VARL_time, VARL_equal, RW_equal, CTSF_time, CTSF_macro, CTSF_equal, CTS_time, CTS_macro, CTS_equal, FHS_equal, AR_equal, FHS_time, AR_macro_nc, FHS_macro, AR_macro, AR_time |
| SPREAD     |               1 |         19 |            16 | CTSF_time    | CTSF_time, CTSF_macro, CTSF_equal, CTS_equal, CTS_time, CTS_macro, VARL_time, VARL_equal, VECM_time, VECM_macro, VECM_equal, AR_equal, AR_macro, AR_time, RW_equal, AR_macro_nc                                 |
| SPREAD     |               4 |         19 |            11 | CTS_time     | CTS_time, CTS_equal, CTS_macro, CTSF_macro, CTSF_time, CTSF_equal, VARL_equal, VECM_equal, VARL_time, VECM_time, VECM_macro                                                                                     |
| SPREAD     |              12 |         19 |            11 | CTS_time     | CTS_time, CTS_equal, CTS_macro, CTSF_time, CTSF_macro, CTSF_equal, VARL_equal, VECM_equal, VECM_time, VARL_time, VECM_macro                                                                                     |
| SPREAD     |              20 |         19 |            11 | CTS_equal    | CTS_equal, CTS_macro, CTS_time, VECM_equal, VECM_time, VECM_macro, VARL_equal, VARL_time, CTSF_equal, CTSF_macro, CTSF_time                                                                                     |
| WTI        |               1 |         19 |            19 | FHS_equal    | FHS_equal, FHS_macro, FHS_time, CTSF_equal, CTS_equal, VARL_equal, AR_equal, CTSF_time, CTS_time, CTSF_macro, CTS_macro, VARL_time, VECM_equal, AR_macro, AR_macro_nc, VECM_macro, AR_time, RW_equal, VECM_time |
| WTI        |               4 |         19 |            19 | FHS_macro    | FHS_macro, FHS_time, FHS_equal, VECM_macro, VECM_time, VARL_equal, VARL_time, VECM_equal, CTS_macro, CTS_time, RW_equal, CTS_equal, CTSF_macro, CTSF_time, CTSF_equal, AR_macro, AR_equal, AR_time, AR_macro_nc |
| WTI        |              12 |         19 |            19 | VECM_macro   | VECM_macro, RW_equal, VECM_equal, VECM_time, VARL_equal, VARL_time, FHS_macro, FHS_time, FHS_equal, CTS_macro, CTS_time, CTS_equal, AR_macro, CTSF_macro, AR_equal, CTSF_time, AR_time, CTSF_equal, AR_macro_nc |
| WTI        |              20 |         19 |            19 | RW_equal     | RW_equal, VECM_equal, VECM_time, VECM_macro, VARL_equal, VARL_time, FHS_time, FHS_macro, FHS_equal, AR_macro_nc, CTS_macro, CTS_time, CTS_equal, AR_macro, AR_equal, AR_time, CTSF_macro, CTSF_time, CTSF_equal |

## Winner per cell

| variable   |   horizon_weeks | best_model   |   mean_crps |   improvement_pct |   dm_pvalue | beats_baseline_at_5pct   |
|:-----------|----------------:|:-------------|------------:|------------------:|------------:|:-------------------------|
| BRENT      |               1 | CTS_equal    |      0.0193 |            1.6992 |      0.3004 | False                    |
| BRENT      |               4 | RW_equal     |      0.0374 |            0      |    nan      | False                    |
| BRENT      |              12 | VECM_macro   |      0.0604 |            1.9632 |      0.5827 | False                    |
| BRENT      |              20 | VECM_equal   |      0.0753 |            2.4429 |      0.5685 | False                    |
| SPREAD     |               1 | CTSF_time    |      0.0106 |            6.9637 |      0.0288 | True                     |
| SPREAD     |               4 | CTS_time     |      0.0163 |           12.0329 |      0.0062 | True                     |
| SPREAD     |              12 | CTS_time     |      0.0198 |           28.1096 |      0      | True                     |
| SPREAD     |              20 | CTS_equal    |      0.0202 |           37.6712 |      0      | True                     |
| WTI        |               1 | FHS_equal    |      0.019  |            2.7315 |      0.1177 | False                    |
| WTI        |               4 | FHS_macro    |      0.0367 |            1.8147 |      0.5322 | False                    |
| WTI        |              12 | VECM_macro   |      0.0632 |            0.0134 |      0.9918 | False                    |
| WTI        |              20 | RW_equal     |      0.079  |            0      |    nan      | False                    |

## Statistically significant improvements over the baseline

| variable   |   horizon_weeks | model      |   improvement_pct |   dm_stat |   dm_pvalue |   eff_independent_n |
|:-----------|----------------:|:-----------|------------------:|----------:|------------:|--------------------:|
| SPREAD     |               1 | CTSF_time  |            6.9637 |   -2.2248 |      0.0288 |                85   |
| SPREAD     |               1 | CTSF_macro |            6.8499 |   -2.198  |      0.0307 |                85   |
| SPREAD     |               1 | CTSF_equal |            6.7937 |   -2.2115 |      0.0297 |                85   |
| SPREAD     |               1 | CTS_equal  |            5.8213 |   -2.0046 |      0.0482 |                85   |
| SPREAD     |               4 | CTS_time   |           12.0329 |   -2.8079 |      0.0062 |                85   |
| SPREAD     |               4 | CTS_equal  |           12.0088 |   -2.8693 |      0.0052 |                85   |
| SPREAD     |               4 | CTS_macro  |           11.9565 |   -2.8005 |      0.0063 |                85   |
| SPREAD     |               4 | CTSF_macro |           11.4058 |   -2.6673 |      0.0092 |                85   |
| SPREAD     |               4 | CTSF_time  |           11.4013 |   -2.6588 |      0.0094 |                85   |
| SPREAD     |               4 | CTSF_equal |           11.2347 |   -2.6736 |      0.009  |                85   |
| SPREAD     |              12 | CTS_time   |           28.1096 |   -5.0302 |      0      |                28.3 |
| SPREAD     |              12 | CTS_equal  |           27.8123 |   -4.9544 |      0      |                28.3 |
| SPREAD     |              12 | CTS_macro  |           27.8068 |   -4.9548 |      0      |                28.3 |
| SPREAD     |              12 | CTSF_time  |           25.6918 |   -5.0613 |      0      |                28.3 |
| SPREAD     |              12 | CTSF_macro |           25.4008 |   -4.9683 |      0      |                28.3 |
| SPREAD     |              12 | CTSF_equal |           25.3331 |   -4.9066 |      0      |                28.3 |
| SPREAD     |              12 | VARL_equal |           25.1603 |   -3.8024 |      0.0003 |                28.3 |
| SPREAD     |              12 | VECM_equal |           25.1268 |   -3.813  |      0.0003 |                28.3 |
| SPREAD     |              12 | VECM_time  |           25.0617 |   -3.7425 |      0.0003 |                28.3 |
| SPREAD     |              12 | VARL_time  |           25.0605 |   -3.7188 |      0.0004 |                28.3 |
| SPREAD     |              12 | VECM_macro |           24.627  |   -3.6539 |      0.0004 |                28.3 |
| SPREAD     |              20 | CTS_equal  |           37.6712 |   -6.3895 |      0      |                17   |
| SPREAD     |              20 | CTS_macro  |           37.2927 |   -6.3609 |      0      |                17   |
| SPREAD     |              20 | CTS_time   |           37.2775 |   -6.3883 |      0      |                17   |
| SPREAD     |              20 | VECM_equal |           35.5521 |   -5.2623 |      0      |                17   |
| SPREAD     |              20 | VECM_time  |           35.3788 |   -5.2143 |      0      |                17   |
| SPREAD     |              20 | VECM_macro |           35.3396 |   -5.2217 |      0      |                17   |
| SPREAD     |              20 | VARL_equal |           35.2753 |   -5.223  |      0      |                17   |
| SPREAD     |              20 | VARL_time  |           35.1887 |   -5.1817 |      0      |                17   |
| SPREAD     |              20 | CTSF_equal |           35.0275 |   -6.1799 |      0      |                17   |
| SPREAD     |              20 | CTSF_macro |           34.7725 |   -6.1744 |      0      |                17   |
| SPREAD     |              20 | CTSF_time  |           34.749  |   -6.2227 |      0      |                17   |

## State space vs VECM on the spread (baseline `VECM_time`)

The common-trend model forecasts from the *filtered* relative
component; the VECM error-correction term is a function of the raw
observed prices and therefore inherits their measurement noise.

|   horizon_weeks | model     |   mean_crps |   improvement_pct |   dm_stat |   dm_pvalue |
|----------------:|:----------|------------:|------------------:|----------:|------------:|
|               1 | CTS_equal |      0.0108 |            1.5194 |   -1.4568 |      0.1489 |
|               1 | CTS_time  |      0.0108 |            1.4932 |   -1.7717 |      0.0801 |
|               1 | CTS_macro |      0.0108 |            1.3537 |   -1.4742 |      0.1442 |
|               4 | CTS_time  |      0.0163 |            3.4848 |   -2.6429 |      0.0098 |
|               4 | CTS_equal |      0.0163 |            3.4584 |   -2.3878 |      0.0192 |
|               4 | CTS_macro |      0.0163 |            3.4011 |   -2.4437 |      0.0166 |
|              12 | CTS_time  |      0.0198 |            4.0671 |   -1.7993 |      0.0756 |
|              12 | CTS_equal |      0.0198 |            3.6705 |   -1.5714 |      0.1199 |
|              12 | CTS_macro |      0.0198 |            3.6631 |   -1.602  |      0.1129 |
|              20 | CTS_equal |      0.0202 |            3.5474 |   -1.3162 |      0.1917 |
|              20 | CTS_macro |      0.0203 |            2.9617 |   -1.0958 |      0.2763 |
|              20 | CTS_time  |      0.0203 |            2.9383 |   -1.1143 |      0.2683 |

## 90% interval calibration

| model       |   ('BRENT', 1) |   ('BRENT', 4) |   ('BRENT', 12) |   ('BRENT', 20) |   ('SPREAD', 1) |   ('SPREAD', 4) |   ('SPREAD', 12) |   ('SPREAD', 20) |   ('WTI', 1) |   ('WTI', 4) |   ('WTI', 12) |   ('WTI', 20) |
|:------------|---------------:|---------------:|----------------:|----------------:|----------------:|----------------:|-----------------:|-----------------:|-------------:|-------------:|--------------:|--------------:|
| AR_equal    |          0.835 |          0.929 |           0.894 |           0.929 |           0.918 |           0.929 |            0.976 |            1     |        0.871 |        0.871 |         0.871 |         0.871 |
| AR_macro    |          0.847 |          0.918 |           0.918 |           0.906 |           0.859 |           0.941 |            0.976 |            1     |        0.882 |        0.882 |         0.906 |         0.871 |
| AR_macro_nc |          0.859 |          0.906 |           0.906 |           0.918 |           0.859 |           0.941 |            0.953 |            1     |        0.882 |        0.871 |         0.882 |         0.882 |
| AR_time     |          0.835 |          0.918 |           0.906 |           0.906 |           0.871 |           0.941 |            0.988 |            1     |        0.871 |        0.894 |         0.894 |         0.859 |
| CTSF_equal  |          0.882 |          0.929 |           0.906 |           0.918 |           0.906 |           0.894 |            0.847 |            0.894 |        0.894 |        0.882 |         0.871 |         0.871 |
| CTSF_macro  |          0.882 |          0.894 |           0.918 |           0.894 |           0.882 |           0.906 |            0.859 |            0.882 |        0.859 |        0.882 |         0.894 |         0.859 |
| CTSF_time   |          0.871 |          0.906 |           0.918 |           0.894 |           0.894 |           0.894 |            0.847 |            0.859 |        0.882 |        0.894 |         0.894 |         0.871 |
| CTS_equal   |          0.882 |          0.918 |           0.894 |           0.906 |           0.882 |           0.882 |            0.894 |            0.906 |        0.882 |        0.894 |         0.871 |         0.871 |
| CTS_macro   |          0.882 |          0.894 |           0.918 |           0.894 |           0.859 |           0.882 |            0.894 |            0.894 |        0.859 |        0.894 |         0.894 |         0.859 |
| CTS_time    |          0.882 |          0.906 |           0.918 |           0.894 |           0.871 |           0.871 |            0.882 |            0.894 |        0.882 |        0.894 |         0.882 |         0.871 |
| FHS_equal   |          0.835 |          0.929 |           0.906 |           0.929 |           0.882 |           0.965 |            1     |            1     |        0.847 |        0.894 |         0.894 |         0.871 |
| FHS_macro   |          0.871 |          0.906 |           0.906 |           0.894 |           0.871 |           0.965 |            0.988 |            1     |        0.871 |        0.894 |         0.894 |         0.871 |
| FHS_time    |          0.859 |          0.918 |           0.906 |           0.894 |           0.871 |           0.965 |            0.988 |            1     |        0.835 |        0.894 |         0.882 |         0.871 |
| RW_equal    |          0.847 |          0.929 |           0.894 |           0.906 |           0.894 |           0.953 |            0.976 |            1     |        0.847 |        0.894 |         0.871 |         0.894 |
| VARL_equal  |          0.871 |          0.929 |           0.894 |           0.894 |           0.871 |           0.871 |            0.824 |            0.835 |        0.847 |        0.906 |         0.882 |         0.882 |
| VARL_time   |          0.882 |          0.929 |           0.894 |           0.894 |           0.871 |           0.882 |            0.835 |            0.812 |        0.847 |        0.894 |         0.859 |         0.894 |
| VECM_equal  |          0.847 |          0.918 |           0.894 |           0.894 |           0.882 |           0.859 |            0.824 |            0.835 |        0.847 |        0.906 |         0.882 |         0.882 |
| VECM_macro  |          0.871 |          0.918 |           0.918 |           0.906 |           0.859 |           0.859 |            0.824 |            0.812 |        0.871 |        0.894 |         0.871 |         0.894 |
| VECM_time   |          0.859 |          0.918 |           0.894 |           0.906 |           0.847 |           0.882 |            0.824 |            0.824 |        0.847 |        0.906 |         0.871 |         0.906 |

## VaR / ES backtests (5% level, overlap-thinned)

| variable   |   horizon_weeks | model       |   n_used |   exception_rate |   kupiec_p |   christoffersen_cc_p |   es_z2 |   es_z2_p |
|:-----------|----------------:|:------------|---------:|-----------------:|-----------:|----------------------:|--------:|----------:|
| BRENT      |               1 | AR_equal    |       85 |           0.0824 |     0.2087 |                0.2637 | -0.6734 |    0.503  |
| BRENT      |               1 | AR_macro    |       85 |           0.0706 |     0.4109 |                0.4864 | -0.987  |    0.5555 |
| BRENT      |               1 | AR_macro_nc |       85 |           0.0588 |     0.7162 |                0.7278 | -0.839  |    0.608  |
| BRENT      |               1 | AR_time     |       85 |           0.0824 |     0.2087 |                0.2637 | -1.4027 |    0.562  |
| BRENT      |               1 | CTSF_equal  |       85 |           0.0706 |     0.4109 |                0.4864 | -0.8086 |    0.3875 |
| BRENT      |               1 | CTSF_macro  |       85 |           0.0588 |     0.7162 |                0.7278 | -0.5461 |    0.5695 |
| BRENT      |               1 | CTSF_time   |       85 |           0.0706 |     0.4109 |                0.4864 | -0.74   |    0.5475 |
| BRENT      |               1 | CTS_equal   |       85 |           0.0706 |     0.4109 |                0.4864 | -0.9349 |    0.3445 |
| BRENT      |               1 | CTS_macro   |       85 |           0.0588 |     0.7162 |                0.7278 | -0.6738 |    0.538  |
| BRENT      |               1 | CTS_time    |       85 |           0.0706 |     0.4109 |                0.4864 | -0.8869 |    0.505  |
| BRENT      |               1 | FHS_equal   |       85 |           0.0941 |     0.0944 |                0.1186 | -0.6345 |    0.4655 |
| BRENT      |               1 | FHS_macro   |       85 |           0.0588 |     0.7162 |                0.7278 | -0.496  |    0.6245 |
| BRENT      |               1 | FHS_time    |       85 |           0.0706 |     0.4109 |                0.4864 | -0.8391 |    0.5515 |
| BRENT      |               1 | RW_equal    |       85 |           0.0706 |     0.4109 |                0.4864 | -0.336  |    0.6605 |
| BRENT      |               1 | VARL_equal  |       85 |           0.0706 |     0.4109 |                0.4864 | -0.7171 |    0.411  |
| BRENT      |               1 | VARL_time   |       85 |           0.0588 |     0.7162 |                0.7278 | -0.416  |    0.626  |
| BRENT      |               1 | VECM_equal  |       85 |           0.0824 |     0.2087 |                0.2637 | -0.7246 |    0.4    |
| BRENT      |               1 | VECM_macro  |       85 |           0.0588 |     0.7162 |                0.7278 | -0.2554 |    0.7305 |
| BRENT      |               1 | VECM_time   |       85 |           0.0588 |     0.7162 |                0.7278 | -0.245  |    0.7385 |
| BRENT      |               4 | AR_equal    |       85 |           0.0353 |     0.5123 |                0.7219 | -1.08   |    0.5    |
| BRENT      |               4 | AR_macro    |       85 |           0.0353 |     0.5123 |                0.7219 | -0.5106 |    0.65   |
| BRENT      |               4 | AR_macro_nc |       85 |           0.0353 |     0.5123 |                0.7219 | -0.4542 |    0.706  |
| BRENT      |               4 | AR_time     |       85 |           0.0353 |     0.5123 |                0.7219 | -0.5422 |    0.694  |
| BRENT      |               4 | CTSF_equal  |       85 |           0.0235 |     0.2136 |                0.4394 | -1.2788 |    0.2855 |
| BRENT      |               4 | CTSF_macro  |       85 |           0.0588 |     0.7162 |                0.682  | -0.6375 |    0.571  |
| BRENT      |               4 | CTSF_time   |       85 |           0.0471 |     0.9    |                0.8122 | -0.6311 |    0.6335 |
| BRENT      |               4 | CTS_equal   |       85 |           0.0353 |     0.5123 |                0.7219 | -1.3652 |    0.2855 |
| BRENT      |               4 | CTS_macro   |       85 |           0.0588 |     0.7162 |                0.682  | -0.7611 |    0.508  |
| BRENT      |               4 | CTS_time    |       85 |           0.0471 |     0.9    |                0.8122 | -0.7133 |    0.5745 |
| BRENT      |               4 | FHS_equal   |       85 |           0.0353 |     0.5123 |                0.7219 | -0.7788 |    0.4975 |
| BRENT      |               4 | FHS_macro   |       85 |           0.0471 |     0.9    |                0.8122 | -0.2425 |    0.7845 |
| BRENT      |               4 | FHS_time    |       85 |           0.0471 |     0.9    |                0.8122 | -0.148  |    0.841  |
| BRENT      |               4 | RW_equal    |       85 |           0.0235 |     0.2136 |                0.4394 | -0.9313 |    0.5    |
| BRENT      |               4 | VARL_equal  |       85 |           0.0235 |     0.2136 |                0.4394 | -0.8366 |    0.5    |
| BRENT      |               4 | VARL_time   |       85 |           0.0235 |     0.2136 |                0.4394 | -0.276  |    0.723  |
| BRENT      |               4 | VECM_equal  |       85 |           0.0235 |     0.2136 |                0.4394 | -0.798  |    0.5    |
| BRENT      |               4 | VECM_macro  |       85 |           0.0235 |     0.2136 |                0.4394 | -0.2781 |    0.838  |
| BRENT      |               4 | VECM_time   |       85 |           0.0353 |     0.5123 |                0.7219 | -0.1798 |    0.8595 |
| SPREAD     |               1 | AR_equal    |       85 |           0.0353 |     0.5123 |                0.7219 |  0.2237 |    0.593  |
| SPREAD     |               1 | AR_macro    |       85 |           0.0588 |     0.7162 |                0.682  |  0.0335 |    0.9095 |
| SPREAD     |               1 | AR_macro_nc |       85 |           0.0588 |     0.7162 |                0.682  |  0.0163 |    0.9795 |
| SPREAD     |               1 | AR_time     |       85 |           0.0471 |     0.9    |                0.8122 |  0.0284 |    0.9315 |
| SPREAD     |               1 | CTSF_equal  |       85 |           0.0471 |     0.9    |                0.8122 |  0.0324 |    0.962  |
| SPREAD     |               1 | CTSF_macro  |       85 |           0.0588 |     0.7162 |                0.682  | -0.2282 |    0.741  |
| SPREAD     |               1 | CTSF_time   |       85 |           0.0588 |     0.7162 |                0.682  | -0.0821 |    0.866  |
| SPREAD     |               1 | CTS_equal   |       85 |           0.0471 |     0.9    |                0.8122 |  0.2334 |    0.6065 |
| SPREAD     |               1 | CTS_macro   |       85 |           0.0588 |     0.7162 |                0.682  |  0.0182 |    0.9775 |
| SPREAD     |               1 | CTS_time    |       85 |           0.0588 |     0.7162 |                0.682  |  0.1665 |    0.6945 |
| SPREAD     |               1 | FHS_equal   |       85 |           0.0588 |     0.7162 |                0.682  | -0.0497 |    0.931  |
| SPREAD     |               1 | FHS_macro   |       85 |           0.0588 |     0.7162 |                0.682  | -0.1845 |    0.797  |
| SPREAD     |               1 | FHS_time    |       85 |           0.0706 |     0.4109 |                0.4493 | -0.0506 |    0.917  |
| SPREAD     |               1 | RW_equal    |       85 |           0.0353 |     0.5123 |                0.7219 |  0.3076 |    0.493  |
| SPREAD     |               1 | VARL_equal  |       85 |           0.0471 |     0.9    |                0.8122 |  0.3924 |    0.4215 |
| SPREAD     |               1 | VARL_time   |       85 |           0.0471 |     0.9    |                0.8122 |  0.2912 |    0.5015 |
| SPREAD     |               1 | VECM_equal  |       85 |           0.0353 |     0.5123 |                0.7219 |  0.2733 |    0.609  |
| SPREAD     |               1 | VECM_macro  |       85 |           0.0471 |     0.9    |                0.8122 | -0.0594 |    0.9145 |
| SPREAD     |               1 | VECM_time   |       85 |           0.0588 |     0.7162 |                0.682  |  0.2463 |    0.585  |
| SPREAD     |               4 | AR_equal    |       85 |           0.0471 |     0.9    |                0.8122 |  0.3415 |    0.356  |
| SPREAD     |               4 | AR_macro    |       85 |           0.0471 |     0.9    |                0.8122 |  0.3028 |    0.4875 |
| SPREAD     |               4 | AR_macro_nc |       85 |           0.0471 |     0.9    |                0.8122 |  0.2637 |    0.5735 |
| SPREAD     |               4 | AR_time     |       85 |           0.0353 |     0.5123 |                0.7219 |  0.3809 |    0.3145 |
| SPREAD     |               4 | CTSF_equal  |       85 |           0.0706 |     0.4109 |                0.4493 | -0.6769 |    0.398  |
| SPREAD     |               4 | CTSF_macro  |       85 |           0.0706 |     0.4109 |                0.4493 | -0.7185 |    0.3575 |
| SPREAD     |               4 | CTSF_time   |       85 |           0.0706 |     0.4109 |                0.4493 | -0.7093 |    0.3325 |
| SPREAD     |               4 | CTS_equal   |       85 |           0.0588 |     0.7162 |                0.682  | -0.1089 |    0.868  |
| SPREAD     |               4 | CTS_macro   |       85 |           0.0588 |     0.7162 |                0.682  | -0.1369 |    0.8285 |
| SPREAD     |               4 | CTS_time    |       85 |           0.0588 |     0.7162 |                0.682  | -0.1571 |    0.7985 |
| SPREAD     |               4 | FHS_equal   |       85 |           0.0235 |     0.2136 |                0.4394 |  0.4139 |    0.246  |
| SPREAD     |               4 | FHS_macro   |       85 |           0.0235 |     0.2136 |                0.4394 |  0.2895 |    0.49   |
| SPREAD     |               4 | FHS_time    |       85 |           0.0235 |     0.2136 |                0.4394 |  0.3752 |    0.49   |
| SPREAD     |               4 | RW_equal    |       85 |           0.0353 |     0.5123 |                0.7219 |  0.3784 |    0.2745 |
| SPREAD     |               4 | VARL_equal  |       85 |           0.0588 |     0.7162 |                0.682  | -0.0985 |    0.8795 |
| SPREAD     |               4 | VARL_time   |       85 |           0.0588 |     0.7162 |                0.682  | -0.2137 |    0.732  |
| SPREAD     |               4 | VECM_equal  |       85 |           0.0706 |     0.4109 |                0.4493 | -0.1975 |    0.7585 |
| SPREAD     |               4 | VECM_macro  |       85 |           0.0706 |     0.4109 |                0.4493 | -0.3925 |    0.592  |
| SPREAD     |               4 | VECM_time   |       85 |           0.0588 |     0.7162 |                0.682  | -0.3383 |    0.6015 |
| WTI        |               1 | AR_equal    |       85 |           0.0353 |     0.5123 |                0.7219 |  0.0915 |    0.884  |
| WTI        |               1 | AR_macro    |       85 |           0.0353 |     0.5123 |                0.7219 | -0.5116 |    0.6405 |
| WTI        |               1 | AR_macro_nc |       85 |           0.0353 |     0.5123 |                0.7219 | -0.4468 |    0.636  |
| WTI        |               1 | AR_time     |       85 |           0.0353 |     0.5123 |                0.7219 | -0.622  |    0.6145 |
| WTI        |               1 | CTSF_equal  |       85 |           0.0353 |     0.5123 |                0.7219 |  0.0025 |    1      |
| WTI        |               1 | CTSF_macro  |       85 |           0.0353 |     0.5123 |                0.7219 | -0.7001 |    0.618  |
| WTI        |               1 | CTSF_time   |       85 |           0.0353 |     0.5123 |                0.7219 | -0.8355 |    0.561  |
| WTI        |               1 | CTS_equal   |       85 |           0.0353 |     0.5123 |                0.7219 |  0.06   |    0.9385 |
| WTI        |               1 | CTS_macro   |       85 |           0.0353 |     0.5123 |                0.7219 | -0.6753 |    0.624  |
| WTI        |               1 | CTS_time    |       85 |           0.0353 |     0.5123 |                0.7219 | -0.7771 |    0.601  |
| WTI        |               1 | FHS_equal   |       85 |           0.0471 |     0.9    |                0.8122 |  0.0423 |    0.878  |
| WTI        |               1 | FHS_macro   |       85 |           0.0588 |     0.7162 |                0.682  | -0.4707 |    0.6235 |
| WTI        |               1 | FHS_time    |       85 |           0.0706 |     0.4109 |                0.4493 | -0.6201 |    0.621  |
| WTI        |               1 | RW_equal    |       85 |           0.0353 |     0.5123 |                0.7219 |  0.2197 |    0.6955 |
| WTI        |               1 | VARL_equal  |       85 |           0.0353 |     0.5123 |                0.7219 |  0.2831 |    0.562  |
| WTI        |               1 | VARL_time   |       85 |           0.0353 |     0.5123 |                0.7219 | -0.3898 |    0.647  |
| WTI        |               1 | VECM_equal  |       85 |           0.0353 |     0.5123 |                0.7219 |  0.2368 |    0.6325 |
| WTI        |               1 | VECM_macro  |       85 |           0.0353 |     0.5123 |                0.7219 | -0.3438 |    0.7075 |
| WTI        |               1 | VECM_time   |       85 |           0.0353 |     0.5123 |                0.7219 | -0.4421 |    0.645  |
| WTI        |               4 | AR_equal    |       85 |           0.0824 |     0.2087 |                0.39   | -1.52   |    0.2995 |
| WTI        |               4 | AR_macro    |       85 |           0.0706 |     0.4109 |                0.5081 | -1.0632 |    0.367  |
| WTI        |               4 | AR_macro_nc |       85 |           0.0706 |     0.4109 |                0.5081 | -0.83   |    0.4555 |
| WTI        |               4 | AR_time     |       85 |           0.0588 |     0.7162 |                0.682  | -0.8416 |    0.375  |
| WTI        |               4 | CTSF_equal  |       85 |           0.0824 |     0.2087 |                0.39   | -1.6507 |    0.2965 |
| WTI        |               4 | CTSF_macro  |       85 |           0.0706 |     0.4109 |                0.4493 | -1.0886 |    0.3675 |
| WTI        |               4 | CTSF_time   |       85 |           0.0588 |     0.7162 |                0.682  | -0.8627 |    0.4025 |
| WTI        |               4 | CTS_equal   |       85 |           0.0588 |     0.7162 |                0.5018 | -1.4662 |    0.33   |
| WTI        |               4 | CTS_macro   |       85 |           0.0588 |     0.7162 |                0.682  | -0.9807 |    0.4065 |
| WTI        |               4 | CTS_time    |       85 |           0.0588 |     0.7162 |                0.682  | -0.769  |    0.462  |
| WTI        |               4 | FHS_equal   |       85 |           0.0588 |     0.7162 |                0.5018 | -0.7804 |    0.48   |
| WTI        |               4 | FHS_macro   |       85 |           0.0588 |     0.7162 |                0.682  | -0.6747 |    0.5075 |
| WTI        |               4 | FHS_time    |       85 |           0.0588 |     0.7162 |                0.682  | -0.5036 |    0.538  |
| WTI        |               4 | RW_equal    |       85 |           0.0353 |     0.5123 |                0.7219 | -0.8452 |    0.525  |
| WTI        |               4 | VARL_equal  |       85 |           0.0471 |     0.9    |                0.3511 | -1.0817 |    0.416  |
| WTI        |               4 | VARL_time   |       85 |           0.0588 |     0.7162 |                0.682  | -0.3949 |    0.619  |
| WTI        |               4 | VECM_equal  |       85 |           0.0471 |     0.9    |                0.3511 | -0.8325 |    0.491  |
| WTI        |               4 | VECM_macro  |       85 |           0.0588 |     0.7162 |                0.682  | -0.389  |    0.6895 |
| WTI        |               4 | VECM_time   |       85 |           0.0471 |     0.9    |                0.8122 | -0.2516 |    0.7415 |
