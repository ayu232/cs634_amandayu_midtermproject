# ============================================================
# Part 4: User-Specified Association Rule Mining
#
# Author: Amanda Yu
# ============================================================

import os
import sys
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


# 
# Folder setup 

script_dir = os.path.dirname(os.path.abspath(__file__))

base_dir = script_dir
input_dir = os.path.join(base_dir, "transactions")
output_dir = os.path.join(base_dir, "results")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_dir):
    print(f"Error: '{input_dir}' not found.")
    print("Make sure the 'transactions' folder is in the same directory as this script.")
    sys.exit(1)

# Helper function

def transactions_to_df(transactions):
    all_items = sorted(set(item for t in transactions for item in t))
    one_hot = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)
    for i, t in enumerate(transactions):
        one_hot.loc[i, list(t)] = 1
    return one_hot


# Step 1: Display available datasets

available_datasets = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

if not available_datasets:
    print("No CSV datasets found in the 'transactions' folder.")
    sys.exit(1)

print("\n=== Available Retailer Datasets ===")
for i, dataset in enumerate(available_datasets, start=1):
    print(f"{i}. {dataset}")



# Step 2: Dataset selection (validated)

while True:
    try:
        choice = int(input("\nSelect a dataset by number: "))
        if 1 <= choice <= len(available_datasets):
            selected_dataset = available_datasets[choice - 1]
            break
        else:
            print(f"Please enter a number between 1 and {len(available_datasets)}.")
    except ValueError:
        print("Invalid input. Please enter a number.")


# Step 3: Get user-specified thresholds

while True:
    try:
        min_support = float(input("Enter minimum support (0–1): "))
        if 0 < min_support <= 1:
            break
        else:
            print("Support must be between 0 and 1.")
    except ValueError:
        print("Invalid input. Please enter a decimal number between 0 and 1.")

while True:
    try:
        min_confidence = float(input("Enter minimum confidence (0–1): "))
        if 0 < min_confidence <= 1:
            break
        else:
            print("Confidence must be between 0 and 1.")
    except ValueError:
        print("Invalid input. Please enter a decimal number between 0 and 1.")


# Step 4: load selected dataset

print("\n=== Execution Summary ===")
print(f"Selected Dataset : {selected_dataset}")
print(f"Minimum Support   : {min_support}")
print(f"Minimum Confidence: {min_confidence}\n")

selected_path = os.path.join(input_dir, selected_dataset)
df = pd.read_csv(selected_path)
transactions = [set(t.split(", ")) for t in df["ItemsPurchased"]]
df_onehot = transactions_to_df(transactions)


# Step 5: Apriori and FP-Growth Anal

print("Running Apriori and FP-Growth")

# Apriori
apriori_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
apriori_rules = association_rules(apriori_itemsets, metric="confidence", min_threshold=min_confidence)

# FP-Growth
fpg_itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
fpg_rules = association_rules(fpg_itemsets, metric="confidence", min_threshold=min_confidence)

# Step 5.5: Display results summary

def display_top_itemsets(itemsets_df, table_name, top_n=10):
    print(f"\n{table_name}:")
    print("Itemset | Count")
    for _, row in itemsets_df.head(top_n).iterrows():
        items = list(row["itemsets"])
        count = int(row["support"] * len(df_onehot))
        print(f"{items} : {count}")

def display_rules(rules_df):
    """Nicely formatted association rules"""
    print("\nFinal Association Rules:")
    if rules_df.empty:
        print("No rules found for given thresholds.")
        return
    for i, row in rules_df.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])
        confidence = row["confidence"] * 100
        print(f"Rule {i+1}: {antecedents} -> {consequents}")
        print(f"Confidence: {confidence:.2f}%\n")

# display Apriori 
print("\n=== Apriori Results Preview ===")
display_top_itemsets(apriori_itemsets, "Table Apriori")
display_rules(apriori_rules)

# display FP-Growth
print("\n=== FP-Growth Results Preview ===")
display_top_itemsets(fpg_itemsets, "Table FP-Growth")
display_rules(fpg_rules)


# Step 6: Save results
prefix = selected_dataset.replace(".csv", "")
apriori_itemsets.to_csv(f"{output_dir}/{prefix}_apriori_itemsets.csv", index=False)
apriori_rules.to_csv(f"{output_dir}/{prefix}_apriori_rules.csv", index=False)
fpg_itemsets.to_csv(f"{output_dir}/{prefix}_fpgrowth_itemsets.csv", index=False)
fpg_rules.to_csv(f"{output_dir}/{prefix}_fpgrowth_rules.csv", index=False)

print(f"\nResults saved in: {output_dir}")
print("Files generated:")
print(f"- {prefix}_apriori_itemsets.csv")
print(f"- {prefix}_apriori_rules.csv")
print(f"- {prefix}_fpgrowth_itemsets.csv")
print(f"- {prefix}_fpgrowth_rules.csv\n")

# Step 7: Display Final Association Rules (Compact View)
print("\nFinal Association Rules:")

# Combine Apriori and FP-Growth results (optional — or just pick one)
final_rules = pd.concat([apriori_rules, fpg_rules]).drop_duplicates(subset=["antecedents", "consequents"])

if final_rules.empty:
    print("No rules found with the given thresholds.")
else:
    for i, row in enumerate(final_rules.itertuples(), start=1):
        antecedents = list(row.antecedents)
        consequents = list(row.consequents)
        confidence = round(row.confidence, 4)
        print(f"Rule {i}: [{set(antecedents)}], [{set(consequents)}], {confidence}")


print("Analysis complete!")
