# Score-Fusion Ablation: Learned vs. Hand-Tuned Weights

RAMA scored candidate venues by linearly combining three component scores with weights set by hand for TREC 2014. Would fitting those weights to data have done better? This experiment tests that on a surrogate task built from the track's profile data.

## Summary

The classical algorithm is reimplemented, validated against the user interest models published in the 2014 paper [1], and evaluated on a surrogate task built from the track's input data. Conditions differ only in how the general- and specific-interest scores are combined.

- Fitting the mixing weight beats both hand-set 2014 settings on graded ranking quality — consistently, and by a small margin.
- The gain lies in how the top of the ranking is ordered, not in which items reach it.
- The interest models carry real signal: every personalized condition ranks well clear of a random baseline, and for a large majority of users.
- The 2014 preference for general interest over specific interest replicates, descriptively.

The fitted weight's *value* is representation-dependent, and a population-level popularity model outperforms every per-user condition here; see Discussion.

## Evaluation Setup

This ablation reuses the track's *input* data rather than its relevance judgments (qrels), as the 2014 qrels only evaluated the top 5 suggestions of submitted runs and do not form a reusable test collection [2].

- **Data**: All 299 Mechanical Turk assessors rated the same 70 example attractions twice each (description and website) on a 0–4 scale. Three assessors gave zero positive ratings (both ratings $\ge 3$) across all 70 items, leaving 296 evaluable users with non-empty relevant sets.
- **Task & Split**: Per user, 70% of ratings (~49 items) build the interest models; the remaining 30% (~21 items) form a held-out ranking task scored against that assessor's own ratings. Evaluated across 10 random train/test splits, shared across conditions and paired by user.
- **Relevance Threshold**: An attraction is relevant if both description and website ratings are $\ge 3$, following the track's criteria. NDCG@5 uses $\min(\text{description rating}, \text{website rating})$ as graded gain.
- **Components & Reconstruction**: Context scoring is excluded because user-location distance and review metadata do not exist for the profile seed items. General-interest categories follow a reconstruction of the 2014 Yelp lookup; see [data/RECONSTRUCTION_METHOD.md](data/RECONSTRUCTION_METHOD.md). Reproduction of user 814's published interest models is tested in [tests/test_rama_gates.py](tests/test_rama_gates.py).

> **Comparability Note**: Scores here should not be read against the track leaderboard. In 2014, systems retrieved up to 50 suggestions from city-specific candidate pools of ~233 Yelp results per context (11,641 candidates across 50 context cities [1]) subject to geographic appropriateness [2]. Here, models rank ~21 held-out seed attractions with a much higher relevance base rate (random ranking yields P@5 = 0.469 here, compared to 0.0843 for the lowest open-web run in 2014 [2]). Comparisons are valid *between* conditions within this experiment, but absolute numbers are not comparable to the 2014 leaderboard.

---

## Results

### Summary of Conditions (Mean over 10 Seeds)

| Condition | Weight Scheme | NDCG@5 | P@5 | MRR |
|---|---|---|---|---|
| popularity (reference) | Population model: crowd average, no per-user parameters | 0.7994 | 0.6572 | 0.8848 |
| **learned** | Convex BPR fit ($W_g \approx 0.583$) | **0.7228** | **0.5670** | **0.7382** |
| fixed-general | 2014 RAMARUN2 renormalized ($W_g = 0.909$) | 0.7171 | 0.5620 | 0.7320 |
| fixed-specific | 2014 RUN1 renormalized ($W_g = 0.091$) | 0.7087 | 0.5472 | 0.7355 |
| random (reference) | Uniform shuffle | 0.6333 | 0.4694 | 0.6433 |

*Between-seed standard deviations for learned condition: 0.0069 (NDCG@5), 0.0081 (P@5), 0.0137 (MRR).*

*Note on 2014 Weights*: In TREC 2014 ([1], Table 7), RAMA used 3-component weights $(W_g, W_s, W_c)$ of $(0.90, 0.09, 0.01)$ for RAMARUN2 and $(0.09, 0.90, 0.01)$ for RUN1. Dropping the context component and renormalizing produces $W_g = 0.90 / 0.99 \approx 0.909$ and $W_g = 0.09 / 0.99 \approx 0.091$.

### Learned Fusion Weights

Across the 10 splits, the fitted general-interest weight converged to:
$$W_g = 0.583 \quad (\text{sd } 0.023, \text{ range } [0.554, 0.616])$$
Neither unconstrained component coefficient was negative in any seed (per-seed $u$ and $v$ are recorded in `results/learned_weights.csv`), so both components contribute positively to the fitted mixture. The fitted weight sits close to balanced, in contrast to the lopsided 90/10 manual splits submitted to the track.

### Paired Statistical Comparison (Learned vs. Fixed Baselines)

Evaluated across 296 users (averaged across splits to reduce partitioning noise), with Holm step-down correction across the two comparisons:

| Comparison | Metric | Mean Diff | 95% CI | Median Diff | Raw $p$ | Holm $p$ |
|---|---|---|---|---|---|---|
| **learned vs. fixed-general** | **NDCG@5** | **+0.0055** | **[+0.0030, +0.0081]** | **+0.0053** | $1.6\times 10^{-5}$ | **$3.2\times 10^{-5}$** |
| | P@5 | +0.0048 | [+0.0004, +0.0094] | +0.0000 | 0.129 | 0.129 |
| | MRR | +0.0061 | [-0.0008, +0.0132] | +0.0000 | 0.071 | 0.141 |
| **learned vs. fixed-specific** | **NDCG@5** | **+0.0143** | **[+0.0081, +0.0204]** | **+0.0138** | $7.5\times 10^{-5}$ | **$7.5\times 10^{-5}$** |
| | **P@5** | **+0.0198** | **[+0.0104, +0.0291]** | **+0.0200** | $6.3\times 10^{-5}$ | **$1.3\times 10^{-4}$** |
| | MRR | +0.0033 | [-0.0104, +0.0167] | +0.0000 | 0.260 | 0.260 |

*Holm correction is applied across the two baseline comparisons within each metric. For P@5 against `fixed-general`, the bootstrap interval for the mean excludes zero while the signed-rank test does not ($p = 0.129$); the difference distribution is tied for 65 of 296 users and skewed, and the rank test is treated as primary.*

### Sensitivity of the Fitted Weight

Each run repeats the full 10-split procedure with one input changed. Shifts are given in units of the baseline's between-seed standard deviation ($0.023$).

| Run | $W_g$ | sd | Shift |
|---|---|---|---|
| baseline | 0.583 | 0.023 | — |
| exclude user 814 | 0.587 | 0.029 | $+0.2$ sd |
| 10% of examples' labels randomized | 0.414 | 0.025 | $-7$ sd |
| 20% randomized | 0.377 | 0.015 | $-9$ sd |
| coarse 5-category taxonomy | 0.353 | 0.012 | $-10$ sd |
| 30% randomized | 0.190 | 0.037 | $-17$ sd |
| 50% randomized | 0.267 | 0.025 | $-14$ sd |

Excluding user 814 — whose published model supplied the 10 verified labels and who is also among the evaluated users — leaves the weight unchanged, closing that circularity concern.

Every degradation of the category representation, by contrast, moves the weight sharply downward. A 10% corruption rate alone accounts for a shift seven times the sampling spread, and adopting a coarser but entirely sensible taxonomy moves it further still. Since the reconstruction's own error rate is unknown and plausibly in this range (10 labels verified, 56 high-confidence, 29 medium, 5 low), the gap between 0.583 and the 2014 setting cannot be attributed to the fitting procedure rather than to label quality.

*Caveat on the sweep*: each noise level is a single corruption draw under a fixed seed, and the reported sd is across data splits, not across corruptions. The direction and magnitude are unambiguous; the non-monotonicity between 30% and 50% is an artifact of that design. Averaging over several corruption seeds per level would give a proper dose-response curve.

---

## Key Findings

1. **Learning the fusion weight provides a modest, statistically clear NDCG@5 gain.**
   The paired NDCG@5 improvement over `fixed-general` (+0.0055, Holm $p = 3.2\times 10^{-5}$) is consistent across users. The gain is smaller than the seed-to-seed variance and achieves statistical significance because pairing across identical splits removes partition noise. All conditions share the same category representation, so this comparison is unaffected by the reconstruction; it is a statement about fitting versus fixing, *given* the representation.
2. **The improvement reflects graded preference discrimination within the top 5.**
   Against `fixed-general`, neither binary P@5 ($p = 0.129$) nor first-hit MRR ($p = 0.141$) shows a statistically significant improvement, with median per-user differences of 0.0000. NDCG@5 captures the difference because it distinguishes between rating strengths (3 vs. 4) and rewards rank ordering across all top-5 positions, whereas binary P@5 treats all top-5 hits identically and MRR halts at the first hit.
3. **The 2014 general-interest prioritization replicates.**
   `fixed-general` outperforms `fixed-specific` on both NDCG@5 (0.7171 vs. 0.7087) and P@5 (0.5620 vs. 0.5472), mirroring the advantage of RAMARUN2 over RUN1 in 2014 [2]. On MRR (0.7320 vs. 0.7355) the order reverses by 0.0035; this comparison was not tested, and is reported descriptively only.
4. **Popularity — a population-level model — outperforms the per-user models on shared items.**
   Ranking held-out items purely by their mean training description rating from other users (`popularity`) achieves NDCG@5 = 0.7994, outperforming the learned per-user model by **0.0766 NDCG@5** (paired difference $-0.0766$, $p = 2\times 10^{-24}$; this and the random comparison are not Holm-adjusted). The learned model is superior for only **73 of 296 users**. This baseline is not an absence of modeling: it is a model of the population, estimated from roughly 208 other assessors' ratings per item, with no per-user parameters. On a closed set of items that every assessor has rated, consensus carries more signal than an individual profile does.

   *Why popularity could not exist in the 2014 track*: The 2014 competition required retrieving unrated venues in 50 unseen cities where no shared assessor rating history existed. In contrast, this ablation evaluates a closed set of 70 well-known attractions where crowd consensus is strong.

5. **Per-user modeling beats no modeling by a wide margin.**
   Every personalized condition ranks far above a random ordering (NDCG@5 0.7087–0.7228 against 0.6333). For the learned condition the paired gain is **+0.0904 NDCG@5** ($p = 8.3\times 10^{-32}$), better for **233 of 296 users**. The interest models therefore capture genuine individual preference; the comparison in finding 4 is between individual-level and population-level modeling, not between modeling and its absence.

6. **The fitted weight measures representation quality as much as component balance.**
   Randomizing 10% of category labels shifts $W_g$ by seven times its sampling spread, and always downward; a coarser taxonomy shifts it by ten. The fitted value is therefore not interpretable as RAMA's optimal general/specific balance, and the distance from the 2014 setting is not evidence that the 2014 choice was wrong. More generally: in a decomposed user model, a learned component weight absorbs how well each component is *represented*, not only how much it *matters*. Isolating the two requires holding representation quality fixed across components — which reconstructed features cannot guarantee.

---

## Discussion

**What the experiment establishes, and what it does not.** All conditions share one category representation, so the paired comparisons are internally fair: given this representation, fitting the mixture beats fixing it. The *value* of the fitted weight is a different matter. The general-interest component rests on labels that had to be reconstructed, and degrading those labels moves the weight by many times its sampling spread — always downward, the same direction as the observed shift away from the 2014 setting. The distance between 0.583 and 0.909 therefore cannot be attributed to the fitting procedure rather than to label quality, and nothing here shows the 2014 choice was wrong.

**A learned component weight absorbs representation quality, not only component importance.** This is the more transferable result. In any decomposed user model — general and specific interest here, but equally long- and short-term profiles, or content and collaborative signals — fitting a mixture over components whose features differ in quality will shift weight toward whichever component happens to be better represented. The fitted value then reads as a statement about the model when it is partly a statement about the features. Separating the two requires representation quality to be held equal across components, which reconstructed features cannot guarantee and which few real feature sets guarantee either.

**Population and individual modeling, not modeling and its absence.** The results separate into three tiers: random ordering, then the per-user interest models well above it, then the population model above those. A crowd average beating per-user profiles is the expected outcome for a dense, shared-item task rather than a discovery, and it does not mean personalization failed — the same models beat random ranking for 233 of 296 users. What it means is that when every item has been rated by nearly everyone, consensus is the stronger estimator, and an individual profile has little left to add. The useful question is not whether to personalize but how population and individual signal should be combined, which is the hybrid-recommender problem. The baseline's role here is to bound the fusion question: when a model-free baseline leads by an order of magnitude more than the gap between fusion settings, the fusion choice is a second-order concern on this task, whatever its optimal value turns out to be.

**Where this points.** The natural follow-ups need a dataset with all three components natively present and a task where personalization has room — the balance may depend on profile sparsity, on the user, or on the context, and a single global weight assumes it depends on none of them.

---

## Reproducibility & Limitations

- **Deterministic Optimization**: The BPR objective is convex with a unique optimum, located by numerical optimization; with fixed seeds, results are identical across platforms (verified on Python 3.12 and 3.14). Fitting a single mixing parameter needs no framework, but the same objective was also fitted by gradient descent in TensorFlow, converging to within 0.0001 — a check on this package's training path rather than a result.
- **Category Reconstruction — measured, and material**: Category labels for the 100 seed items are reconstructed (10 verified against the published user 814 model in [1]; 56 high-confidence, 29 medium, 5 low). The sensitivity runs above show label quality dominates the fitted weight: 10% corruption shifts it seven sampling deviations downward. The paired comparisons between conditions hold, since all conditions share the labels; the *value* of $W_g$ does not generalize beyond this representation.
- **Circularity — checked**: User 814 is among the evaluated users and their published model supplied the 10 verified labels. Excluding them entirely gives $W_g = 0.587$ against 0.583, within sampling spread.
- **Shared vocabulary**: Training and held-out items are drawn from the same 70 attractions, so term and category overlap is higher than in a retrieval setting. This inflates absolute scores for every personalized condition without favoring one over another.
- **Scope**: Evaluates interest-model fusion (general vs. specific); context scoring (distance, Yelp review count/rating) is omitted as it is not defined on seed profile ratings.
- **Code**: Reproducible via `python run_ablation.py`; see the commands in the repository README.

---

## References

[1] H. Li and R. Alonso. **User Modeling for Contextual Suggestion.** In *Proceedings of the Twenty-Third Text REtrieval Conference (TREC 2014)*, NIST Special Publication 500-308. National Institute of Standards and Technology, 2014. <https://trec.nist.gov/pubs/trec23/papers/pro-RAMA_cs.pdf>

[2] A. Dean-Hall, C. L. A. Clarke, J. Kamps, P. Thomas, and E. Voorhees. **Overview of the TREC 2014 Contextual Suggestion Track.** In *Proceedings of the Twenty-Third Text REtrieval Conference (TREC 2014)*, NIST Special Publication 500-308. National Institute of Standards and Technology, 2014. <https://trec.nist.gov/pubs/trec23/papers/overview-context.pdf>

Both appear in the TREC 2014 proceedings: <https://trec.nist.gov/pubs/trec23/trec2014.html>
