from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


@dataclass
class AprioriResult:
    """Container for encoded transactions, display itemsets, and association rules."""

    encoded_transactions: pd.DataFrame
    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame


def _format_itemset(value: object) -> str:
    if isinstance(value, frozenset):
        return ", ".join(sorted(value))
    return str(value)


def _prepare_transactions(transactions: pd.Series) -> list[list[str]]:
    """Return non-empty transactions with duplicate items removed."""
    cleaned: list[list[str]] = []
    for transaction in transactions.tolist():
        if not isinstance(transaction, list):
            continue
        unique_items = sorted(set(item for item in transaction if item))
        if unique_items:
            cleaned.append(unique_items)
    return cleaned


def run_apriori(
    transactions: pd.Series,
    min_support: float,
    min_confidence: float,
    min_lift: float,
) -> AprioriResult:
    """Encode transactions, run Apriori once, and return display-ready results."""
    transaction_list = _prepare_transactions(transactions)
    if not transaction_list:
        return AprioriResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transaction_list).transform(transaction_list)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.columns_)

    frequent_itemsets_raw = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True,
    )
    if frequent_itemsets_raw.empty:
        return AprioriResult(encoded_df, frequent_itemsets_raw, pd.DataFrame())

    rules = association_rules(
        frequent_itemsets_raw,
        metric="confidence",
        min_threshold=min_confidence,
    )

    frequent_itemsets = frequent_itemsets_raw.copy()
    frequent_itemsets = frequent_itemsets.sort_values(
        ["support"],
        ascending=False,
    ).reset_index(drop=True)
    frequent_itemsets["jumlah_item"] = frequent_itemsets["itemsets"].map(len)
    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].map(_format_itemset)

    if rules.empty:
        return AprioriResult(encoded_df, frequent_itemsets, pd.DataFrame())

    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return AprioriResult(encoded_df, frequent_itemsets, pd.DataFrame())

    rules["antecedents"] = rules["antecedents"].map(_format_itemset)
    rules["consequents"] = rules["consequents"].map(_format_itemset)
    display_columns = ["antecedents", "consequents", "support", "confidence", "lift"]
    rules = (
        rules[display_columns]
        .sort_values(["confidence", "lift", "support"], ascending=False)
        .reset_index(drop=True)
    )

    return AprioriResult(encoded_df, frequent_itemsets, rules)
