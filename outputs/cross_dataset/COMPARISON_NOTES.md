# Cross-Dataset Pilot Notes

## Scope

The figures compare the same one-round, five-client experiment grid: no defense, SecAgg-only, and DP-only / DP+SecAgg at target epsilon 0.5, 1, and 2. MovieLens uses a seeded sample from the real MovieLens-32M file (2,965 retained users, 5,615 items, 445,773 ratings). Amazon uses the Electronics subset of the attached catalog as implicit positives (232 retained users, 111 items, 863 listed interactions).

These are pilot results, not final research estimates: each configuration has one seed and one FL round. Amazon's catalog has no individual review stars or timestamps, so its RMSE is intentionally undefined and its random implicit-feedback holdout is not methodologically identical to MovieLens's chronological rating holdout.

## Security Findings

- Threshold membership AUC is close to the 0.5 chance level on both datasets. DP+SecAgg averages 0.509 on Amazon versus 0.531 for its no-defense baseline. On the MovieLens sample, DP+SecAgg averages 0.499 versus 0.473 for no defense; the baseline is already below chance, so this is not evidence of a consistent cross-dataset improvement.
- Shadow-model AUC remains near chance on both datasets (approximately 0.50). These results do not establish a reliable defense ranking.
- Gradient support precision drops from 0.018 to 0.0018 on MovieLens and from 0.417 to 0.045 on Amazon when moving from no defense to DP+SecAgg in this pilot. This attack metric shows stronger degradation on Amazon here, but should be repeated across seeds and rounds.

## Utility Findings

- MovieLens Recall@5 and NDCG@5 are zero for this one-round sample across the plotted settings; DP also produces very large rating RMSE. This pilot is not adequately trained for a useful MovieLens utility comparison.
- Amazon's no-defense Recall@5 is 0.056; DP+SecAgg averages 0.050 across the three epsilon values. At epsilon 1, DP+SecAgg Recall@5 is 0.056, while Recall@10 is 0.095 versus 0.086 for no defense.
- Absolute ranking scores should not be compared as if the candidate sets and holdout protocols were identical. Compare within-dataset defense trends only.

## Interpretation

The relative security trends are **not consistently preserved** across these two pilot datasets. The DP+SecAgg threshold-membership score moves closer to chance on Amazon, while MovieLens starts below chance and moves slightly above it. The implicit Amazon catalog is much smaller and less complete than a review-event dataset, so these results support a preliminary cross-domain check only. A stronger claim needs repeated seeds and a longer MovieLens training schedule.
