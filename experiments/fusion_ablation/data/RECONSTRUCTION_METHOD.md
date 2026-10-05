# Example-category reconstruction — method

**Status: reconstruction, not the original 2014 mapping.**

## What survives from 2014

Li & Alonso (2014), *User Modeling for Contextual Suggestion*, §2.2.2: each of the 100
example attractions was matched to a business by querying the Yelp API v2 with its title
and city, and that business's Yelp categories were taken as the example's categories. The
union across all 100 examples was exactly **72 categories**, printed in Figure 2.
They are in `yelp_categories_72.csv`. The per-example assignment was never published.

## How the assignment was reconstructed

1. **Semantic annotation.** Each example was labelled from its title and description,
   using only the 72 categories. An absent category is evidence too: `belgian` is not
   among the 72, so Hopleaf must have been filed under bars/pubs.

2. **Verification against the published model.** The paper prints user 814's general
   interest model (Table 1, top 24 categories; Table 4 adds `wineries` −0.083 and
   `horsebackriding` −0.25). User 814's ratings are in `profiles2014-100.csv`. Re-running
   RAMA with the paper's own rules — Table 6 relevance mapping, reinforcement factor 0.5,
   decay 0, description rating, examples in ID order — shows the mention frequency is
   **1/k for an example carrying k categories**. That model reproduces published weights
   exactly, e.g. `shoppingcenters` = 9/16.

3. **Inversion.** Only examples user 814 rated 1, 3 or 4 affect their model (a relevance
   of 0 leaves no trace), so 45 examples are constrained. Where a published weight admits
   exactly one solution, the label set is marked `verified`. Categories absent from
   Table 1 must weigh at most 0.234, which excludes further assignments.

## Limits

- One user's model cannot uniquely determine 100 label sets. Examples rated neutral by
  user 814 are unconstrained; several large categories (`museums`, `galleries`) have
  thousands of consistent solutions.
- The reconstruction reproduces 12 of the 26 published weights exactly and keeps every
  hidden category under the cutoff. The remainder are semantic best guesses.
- `steak` is unplaced. The 2014 union guarantees some example carried it, but no example
  supports it and user 814's model does not constrain it.
- Example 127 carrying `wineries` is forced by the data and semantically odd, which
  suggests the 2014 title-and-city lookup sometimes matched a different business.

## Confidence levels

| Level | Meaning |
|---|---|
| verified | Unique solution given user 814's published model |
| high | Unambiguous from title and description |
| medium | Plausible; one of several readings, or partly constrained by the inversion |
| low | Best available fit from the 72; weak evidence |

## Sources

- Categories and procedure: Li & Alonso (2014), NIST TREC 2014 proceedings
- `examples2014.csv`, `profiles2014-100.csv`: TREC 2014 Contextual Suggestion track data,
  retrieved from the University of Glasgow Terrier team's public repository
  `tthonet/composite-contextual-suggestion`
