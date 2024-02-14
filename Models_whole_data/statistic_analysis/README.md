# Statistical Analysis of Results

## Hierarchical vs Non-Hierarchical T-test
Given the predictions of two models:

Group A the hierarchical model (e.g. LSGBERT + BiGRU).
Group B the non-hierarchical model (e.g. LSGBERT).

Null hypothesis H0: The mean of A is not greater than that of B i.e. μ(A) <= μ(B)

Alternative hypothesis H1: The mean of A is greater than that of B i.e. μ(A) > μ(B)

The paired t-test (stats.ttest_rel()) is used since the models are tested on the same dataset.

The parameter 'alternative' is set to *'greater'* in this way a one-tailed test is performed to see if the mean of the distribution of GROUP A (hierarchical model) is not greater than the mean of the distribution of GROUP B (non-hierarchical model).

Considering alpha=0.05 and comparing it to the p-value, when:

- p-value<α: we reject the null hypothesis H0 and therefore A is actually better than B. This is the condition that determines that model A is better than model B.
- p-value>=α: there is not enough evidence to reject the null hypothesis (H0) and therefore we cannot say that A is actually better than B.
