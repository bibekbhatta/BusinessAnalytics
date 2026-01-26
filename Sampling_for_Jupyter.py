# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:39:18 2026

@author: bibek
"""

############################################################
# ACC7013 – Lab Session (Week 2)
# Statistical Inference: Sampling, CLT, CIs, Hypothesis Testing

############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, binomtest, ttest_ind, ttest_rel, chi2_contingency
np.random.seed(42)
plt.style.use('default')

############################################################
# 1. Sampling Distribution of the Mean & CLT
############################################################

# Learning goal:
# Understand how sample means vary and why the CLT makes them approximately normal.

# Create a skewed population (Exponential distribution)
population = np.random.exponential(scale=1, size=500000)

# Visualise the population distribution
plt.figure(figsize=(10, 5))
plt.hist(population, bins=60, alpha=0.6, color='purple')
plt.title("Underlying Population (Exponential, Skewed)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

# Function to draw repeated sample means
def sample_means(n, reps=5000):
    return [np.mean(np.random.choice(population, size=n)) for _ in range(reps)]

# Show sampling distributions for different sample sizes
plt.figure(figsize=(14, 6))
sample_sizes = [5, 30, 100]

for n in sample_sizes:
    means = sample_means(n)
    plt.hist(means, bins=40, alpha=0.5, density=True, label=f"n={n}")

plt.title("Sampling Distribution of the Mean (CLT in Action)")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

############################################################
# 2. Normal Approximation for Large n
############################################################

means_100 = sample_means(100)
mu = np.mean(means_100)
sigma = np.std(means_100)

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)

plt.figure(figsize=(10, 6))
plt.hist(means_100, bins=40, density=True, alpha=0.5, label="Simulated Means")
plt.plot(x, norm.pdf(x, mu, sigma), 'r--', linewidth=2, label="Normal Curve")
plt.title("Sampling Distribution (n=100) with Normal Approximation")
plt.legend()
plt.show()

############################################################
# 3. Confidence Intervals
############################################################

# Simple CI example (students can compute by hand)
sample = np.random.normal(50, 10, 30)
xbar = np.mean(sample)
se = 10 / np.sqrt(30)
lower = xbar - 1.96 * se
upper = xbar + 1.96 * se

print("Simple CI Example:")
print(f"Sample mean = {xbar:.2f}")
print(f"95% CI = ({lower:.2f}, {upper:.2f})\n")

############################################################
# CI Coverage Simulation
############################################################

true_mean = 50
true_sd = 10

def one_ci(n=30):
    sample = np.random.normal(true_mean, true_sd, n)
    xbar = np.mean(sample)
    se = true_sd / np.sqrt(n)
    lower = xbar - 1.96 * se
    upper = xbar + 1.96 * se
    return lower <= true_mean <= upper

coverage = np.mean([one_ci() for _ in range(1000)])
print("Proportion of intervals containing the true mean:", coverage)

############################################################
# Visualising 50 Confidence Intervals
############################################################

n = 30
num_intervals = 50
intervals = []
contains = []

for i in range(num_intervals):
    sample = np.random.normal(true_mean, true_sd, n)
    xbar = np.mean(sample)
    se = true_sd / np.sqrt(n)
    lower = xbar - 1.96 * se
    upper = xbar + 1.96 * se
    intervals.append((lower, upper))
    contains.append(lower <= true_mean <= upper)

plt.figure(figsize=(14, 10))
for i, ((lower, upper), ok) in enumerate(zip(intervals, contains)):
    color = 'blue' if ok else 'red'
    plt.plot([lower, upper], [i, i], color=color, linewidth=3)
    plt.plot([true_mean], [i], 'ko')

plt.axvline(true_mean, color='black', linestyle='--')
plt.title("Confidence Intervals: Blue = Contains True Mean, Red = Misses")
plt.xlabel("Value")
plt.ylabel("Interval Index")
plt.grid(alpha=0.3)
plt.show()

############################################################
# CI Width vs Sample Size
############################################################

sizes = [10, 30, 100, 500]
widths = []

for n in sizes:
    se = true_sd / np.sqrt(n)
    widths.append(2 * 1.96 * se)

plt.plot(sizes, widths, marker='o')
plt.title("CI Width Shrinks as Sample Size Increases")
plt.xlabel("Sample Size")
plt.ylabel("CI Width")
plt.grid(alpha=0.3)
plt.show()

############################################################
# 4. Hypothesis Testing
############################################################

# Learning goal:
# Understand how hypothesis tests compare data to a null model.

############################################################
# Rejection Region Visualisation for a Binomial Test
############################################################

p_true = 0.5
n_coins = 50
n_experiments = 1000

heads_counts = []
for _ in range(n_experiments):
    coins = np.random.rand(n_coins) < p_true
    heads_counts.append(np.sum(coins))

alpha = 0.05
lower_cutoff = binom.ppf(alpha/2, n_coins, 0.5)
upper_cutoff = binom.ppf(1 - alpha/2, n_coins, 0.5)

plt.figure(figsize=(12, 6))
plt.hist(heads_counts, bins=range(0, n_coins+2), edgecolor='black', alpha=0.7)
plt.axvspan(0, lower_cutoff, color='red', alpha=0.2)
plt.axvspan(upper_cutoff, n_coins, color='red', alpha=0.2)
plt.axvline(lower_cutoff, color='red', linestyle='--')
plt.axvline(upper_cutoff, color='red', linestyle='--')
plt.title("Rejection Regions for H0: p = 0.5")
plt.show()

############################################################
# One-Sample Proportion Test (Fair Coin Example)
############################################################

import math
from scipy.stats import norm

print("\n--- One-Sample Proportion Test (Fair Coin Example) ---")

n = 50
heads = 32
p0 = 0.5

phat = heads / n
SE = math.sqrt(p0 * (1 - p0) / n)
z = (phat - p0) / SE
p_value = 2 * (1 - norm.cdf(abs(z)))

print(f"Observed proportion: {phat:.3f}")
print(f"Standard Error: {SE:.4f}")
print(f"z-statistic: {z:.3f}")
print(f"p-value: {p_value:.4f}")
print("Interpretation: Evidence against fairness if p < 0.05\n")

############################################################
# Two-Sample Comparison (Email Campaign Example)
############################################################

import math
from scipy.stats import t

print("--- Two-Sample Comparison (Independent Means) ---")

# Group A (old design)
n1 = 40
mean1 = 5.2
sd1 = 2.0

# Group B (new design)
n2 = 45
mean2 = 6.1
sd2 = 2.4

diff = mean2 - mean1
SE = math.sqrt(sd1**2 / n1 + sd2**2 / n2)
t_stat = diff / SE

df = (sd1**2 / n1 + sd2**2 / n2)**2 / (
    (sd1**2 / n1)**2 / (n1 - 1) +
    (sd2**2 / n2)**2 / (n2 - 1)
)

p_value = 2 * (1 - t.cdf(abs(t_stat), df))

print(f"Difference in means: {diff:.3f}")
print(f"Standard Error: {SE:.3f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"Degrees of freedom: {df:.1f}")
print(f"p-value: {p_value:.4f}")
print("Interpretation: At α = 0.05, evidence is not strong enough to conclude a difference.\n")

############################################################
# Paired t-test 
############################################################

print("--- Paired t-test ---")
before = np.random.normal(70, 8, 20)
after = before + np.random.normal(3, 2, 20)

t_stat_rel, p_val_rel = ttest_rel(before, after)
print(f"Mean Difference: {np.mean(after - before):.2f}")
print(f"P-value: {p_val_rel:.4f}")

############################################################
# Summary
############################################################

print("""
Summary:
- Sampling distributions show how statistics vary across samples.
- CLT explains why sample means become normal.
- Confidence intervals quantify uncertainty in estimates.
- Hypothesis tests compare data to a null model.
- Power increases with sample size.
- Statistical significance does not imply practical significance.
""")


##### another quiz
### what can you infer from the following experiment?
import numpy as np
from scipy.stats import norm

true_mean = 170          # null hypothesis mean
sample_mean = 168        # observed sample mean
true_sd = 10             # population SD

sizes = [10, 30, 100, 500]

print("Sample Size   SE       95% CI Width     95% CI Interval              p-value")
print("-------------------------------------------------------------------------------")

for n in sizes:
    se = true_sd / np.sqrt(n)
    ci_width = 2 * 1.96 * se
    lower = sample_mean - 1.96 * se
    upper = sample_mean + 1.96 * se
    
    # z-test for H0: mu = 170
    z = (sample_mean - true_mean) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    
    print(f"{n:<13} {se:>6.3f} {ci_width:>15.3f}   [{lower:6.2f}, {upper:6.2f}]      {p:>8.4f}")

print("For sample sizes of 100 and 500, the data provide enough evidence \
to reject the null hypothesis that the population mean is 170. \
The sample evidence is inconsistent with the claim that the \
true mean is 170.")
