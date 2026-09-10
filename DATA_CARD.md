# Dataset card

## Dataset

`Mobile Reviews Sentiment.csv` contains mobile-device reviews, product attributes, customer age, market, price, ratings, sentiment labels and review dates.

## Intended use

The bundled data supports exploratory analysis, sentiment experiments and demonstration of the inventory workflow. Review counts are used only as a demand proxy when observed units sold are unavailable.

## Not a production target

The bundled dataset does not contain a verified restocking-success outcome. V2 therefore uses transparent rule-based ranking for this dataset. Supervised XGBoost or Random Forest success scoring is enabled only when an external `restock_success` field is supplied with sufficient examples from both classes.

## Limitations

- Review activity is not equivalent to sales.
- Customer age may be incomplete or unrepresentative.
- Prices may not reflect current market prices.
- Sentiment labels may contain annotation noise.
- Heuristic return costs and fallback margins are not observed financial outcomes.
- Geographic coverage may not represent all target markets.

## Privacy and fairness

Do not upload direct personal identifiers. Age should be used only for aggregate market analysis, not individual eligibility or discriminatory pricing. Evaluate performance by market and remove demographic features if they do not provide justified business value.

## Source and license

The repository does not currently document the original external source or redistribution license for the CSV. The repository owner should add the source URL, collection method, license and attribution before public production use.

## Recommended production replacement

Use transaction-level data with units sold, selling price, purchase cost, inventory, returns, lead time, stock-outs, lost sales and an externally observed restocking outcome.
