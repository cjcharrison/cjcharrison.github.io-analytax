"""
Generate a realistic synthetic R&D tax relief dataset for UK-focused tax analytics.

Produces ~120 company-year records with correlated features and a dependent
variable (qualifying_rd_expenditure) built from a realistic formula that
mirrors actual UK R&D tax relief mechanics.

Output: /Users/chrisharrison/Projects/analytax/assets/data/rd_tax_relief.csv
"""

import csv
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N = 120  # number of observations
OUTPUT_PATH = "/Users/chrisharrison/Projects/analytax/assets/data/rd_tax_relief.csv"

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Generate base features with realistic correlations
# ---------------------------------------------------------------------------

# rd_headcount — primary driver, log-normal so we get a right-skewed
# distribution (many small teams, fewer large ones)
rd_headcount_raw = rng.lognormal(mean=3.2, sigma=0.85, size=N)
rd_headcount = np.clip(rd_headcount_raw, 5, 200).astype(int)

# avg_salary — weakly correlated with headcount (larger teams tend to be
# at bigger firms that pay a bit more, but the relationship is noisy)
avg_salary_base = 35_000 + (rd_headcount / 200) * 20_000  # structural part
avg_salary_noise = rng.normal(0, 8_000, size=N)
avg_salary = np.clip(avg_salary_base + avg_salary_noise, 35_000, 85_000).astype(int)

# num_projects — positively correlated with headcount
num_projects_raw = 1 + (rd_headcount / 200) * 20 + rng.normal(0, 3, size=N)
num_projects = np.clip(num_projects_raw, 1, 25).astype(int)

# subcontractor_costs — many companies use none; those that do, spend varies
# Use a zero-inflated approach: ~30% of companies have zero subcontractor spend
has_subcontractors = rng.random(size=N) > 0.30
subcontractor_costs_raw = rng.lognormal(mean=10.5, sigma=1.2, size=N)  # median ~36k
subcontractor_costs = np.where(has_subcontractors, subcontractor_costs_raw, 0)
subcontractor_costs = np.clip(subcontractor_costs, 0, 500_000).astype(int)

# consumables_spend — loosely scales with headcount and number of projects
consumables_base = 5_000 + rd_headcount * 300 + num_projects * 1_500
consumables_noise = rng.normal(0, 10_000, size=N)
consumables_spend = np.clip(consumables_base + consumables_noise, 5_000, 150_000).astype(int)

# software_costs — scales with headcount (licences per head) + project complexity
software_base = 2_000 + rd_headcount * 250 + num_projects * 800
software_noise = rng.normal(0, 6_000, size=N)
software_costs = np.clip(software_base + software_noise, 2_000, 100_000).astype(int)

# pct_time_qualifying — beta-distributed, centred around 0.75
alpha, beta_param = 6, 2.5  # gives a left-skewed distribution peaking ~0.75
pct_time_qualifying = rng.beta(alpha, beta_param, size=N)
pct_time_qualifying = np.clip(pct_time_qualifying, 0.50, 0.95)
pct_time_qualifying = np.round(pct_time_qualifying, 2)

# ---------------------------------------------------------------------------
# 2. Inject a handful of realistic outliers
# ---------------------------------------------------------------------------
outlier_indices = rng.choice(N, size=6, replace=False)

# 2 companies with unusually high subcontractor reliance
for idx in outlier_indices[:2]:
    subcontractor_costs[idx] = rng.integers(350_000, 500_000)

# 2 companies with unusually low qualifying-time percentages
for idx in outlier_indices[2:4]:
    pct_time_qualifying[idx] = round(rng.uniform(0.50, 0.58), 2)

# 2 companies with disproportionately large consumables spend
for idx in outlier_indices[4:]:
    consumables_spend[idx] = rng.integers(120_000, 150_000)

# ---------------------------------------------------------------------------
# 3. Compute the dependent variable
# ---------------------------------------------------------------------------

# Staff costs (biggest component): headcount * salary * qualifying-time fraction
staff_costs = rd_headcount * avg_salary * pct_time_qualifying

# Subcontractor costs qualify at 65% under UK rules
qualifying_subcontractor = subcontractor_costs * 0.65

# Software costs partially qualify (80%)
qualifying_software = software_costs * 0.80

# Base qualifying expenditure
qualifying_base = (
    staff_costs
    + qualifying_subcontractor
    + consumables_spend
    + qualifying_software
)

# Add realistic noise (~5-10% relative standard deviation)
noise_pct = rng.normal(0, 0.07, size=N)  # ~7% average noise
qualifying_rd_expenditure = qualifying_base * (1 + noise_pct)

# Ensure no negative values and round to nearest pound
qualifying_rd_expenditure = np.maximum(qualifying_rd_expenditure, 0).astype(int)

# ---------------------------------------------------------------------------
# 4. Write CSV
# ---------------------------------------------------------------------------
columns = [
    "record_id",
    "rd_headcount",
    "avg_salary",
    "subcontractor_costs",
    "consumables_spend",
    "software_costs",
    "num_projects",
    "pct_time_qualifying",
    "qualifying_rd_expenditure",
]

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    for i in range(N):
        writer.writerow([
            i + 1,
            int(rd_headcount[i]),
            int(avg_salary[i]),
            int(subcontractor_costs[i]),
            int(consumables_spend[i]),
            int(software_costs[i]),
            int(num_projects[i]),
            float(pct_time_qualifying[i]),
            int(qualifying_rd_expenditure[i]),
        ])

print(f"Dataset written to {OUTPUT_PATH}")
print(f"Shape: {N} rows x {len(columns)} columns\n")

# ---------------------------------------------------------------------------
# 5. Print diagnostics
# ---------------------------------------------------------------------------

# Gather all arrays into a dict for easy reporting
data = {
    "rd_headcount": rd_headcount.astype(float),
    "avg_salary": avg_salary.astype(float),
    "subcontractor_costs": subcontractor_costs.astype(float),
    "consumables_spend": consumables_spend.astype(float),
    "software_costs": software_costs.astype(float),
    "num_projects": num_projects.astype(float),
    "pct_time_qualifying": pct_time_qualifying.astype(float),
    "qualifying_rd_expenditure": qualifying_rd_expenditure.astype(float),
}

print("=" * 80)
print("FIRST 10 ROWS")
print("=" * 80)
header_fmt = "{:>3s} {:>12s} {:>11s} {:>18s} {:>17s} {:>14s} {:>12s} {:>10s} {:>18s}"
row_fmt    = "{:>3d} {:>12d} {:>11d} {:>18d} {:>17d} {:>14d} {:>12d} {:>10.2f} {:>18d}"
print(header_fmt.format("id", "rd_headcount", "avg_salary", "subcontractor_costs",
                         "consumables_spend", "software_costs", "num_projects",
                         "pct_time", "qual_rd_expend"))
for i in range(10):
    print(row_fmt.format(
        i + 1,
        int(rd_headcount[i]),
        int(avg_salary[i]),
        int(subcontractor_costs[i]),
        int(consumables_spend[i]),
        int(software_costs[i]),
        int(num_projects[i]),
        float(pct_time_qualifying[i]),
        int(qualifying_rd_expenditure[i]),
    ))

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
stat_fmt = "{:<28s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}"
print(stat_fmt.format("Feature", "Mean", "Std", "Min", "Median", "Max"))
print("-" * 92)
for name, arr in data.items():
    print(stat_fmt.format(
        name,
        f"{np.mean(arr):,.1f}",
        f"{np.std(arr):,.1f}",
        f"{np.min(arr):,.1f}",
        f"{np.median(arr):,.1f}",
        f"{np.max(arr):,.1f}",
    ))

print("\n" + "=" * 80)
print("CORRELATION WITH qualifying_rd_expenditure")
print("=" * 80)
target = data["qualifying_rd_expenditure"]
for name, arr in data.items():
    if name == "qualifying_rd_expenditure":
        continue
    corr = np.corrcoef(arr, target)[0, 1]
    print(f"  {name:<28s}  r = {corr:+.3f}")
