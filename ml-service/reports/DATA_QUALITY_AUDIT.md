# NFHS-5 Data Quality Audit

- Rows: **198,849**
- Columns: **33**
- Dataset SHA-256: `4215d66720c761695eb7f029962c4ebb5d0df5cf40c336a8fc0bc9e803a04abf`
- Exact duplicate rows: **0**
- Unique case IDs: **157,674**
- Records belonging to repeated case IDs: **79,312**

## Labels

- Stunting and wasting are binary.
- Composite malnutrition is deterministically `Stunting OR Wasting`.
- Composite consistency verified: **True**.

## Missingness

- `measles_vaccine`: 82,555
- `anc_visits`: 48,150
- `breastfeeding_duration`: 39,461
- `birth_weight`: 18,673
- `mother_bmi`: 1,146
- `mother_height`: 894
- `mother_weight`: 873
- `toilet_type`: 1

## Interpretation

No records were removed and no target labels were imputed. Missing values must be handled inside each training split. Repeated case IDs require a group-aware sensitivity analysis because random row splitting can expose related household records across partitions.

## Source columns currently omitted

- `v001`
- `v002`
- `v003`
- `age_hh_head_proxy`
- `wealth_score`
- `gender_hh_head`
- `dist_market_proxy`
- `birth_weight_source`
- `mother_weight`
- `mother_height`
- `mother_marital_status`
