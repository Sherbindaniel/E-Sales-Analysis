# Statistical Modeling of High-Volume Website Traffic Patterns Using Probability Distributions with Python

**Project Title**: Statistical Modeling of High-Volume Website Traffic Patterns Using Probability Distributions with Python  
**Role**: Lead Data Scientist  
**Date**: September 2026 (Validated & Refined)

---

## 1. Executive Summary & Research Objectives

Modern web applications frequently experience dramatic traffic fluctuations driven by product releases, promotional campaigns, or high-demand online events. In such scenarios, website capacity planning and resource provisioning are critical. If planning is based on flawed assumptions—such as treating traffic as a simple, independent Poisson process—servers will be severely under-provisioned, leading to service degradation or catastrophic failure during peak surges.

This project investigates the following primary research question:
> **"How can probability distributions statistically model high-volume website traffic patterns, particularly during peak usage periods, and how does this modeling differ from normal background usage?"**

### Research Hypotheses & Evaluative Criteria
1. **Hypothesis 1 (Goodness-of-Fit & Model Selection)**:
   - $H_0$: The selected probability distribution adequately represents the observed traffic pattern.
   - $H_1$: The selected probability distribution differs significantly from the observed traffic pattern.
   - *Evaluative Distinction*: We distinguish between **statistically acceptable fit** (passing $p > 0.05$ threshold on GoF tests) and **best relative model** (minimizing AIC/BIC information loss).
2. **Hypothesis 2 (Discrete Models & Dispersion)**:
   - $H_0$: Traffic count data is equidispersed ($\text{Variance} \approx \text{Mean}$) and follows a Poisson process.
   - $H_1$: Traffic count data is overdispersed ($\text{Variance} \gg \text{Mean}$), requiring a Negative Binomial distribution.
3. **Hypothesis 3 (Regime Shift)**:
   - $H_0$: The same probability distribution minimizes relative information loss across both normal (below 90th percentile) and peak (90th percentile and above) traffic regimes.
   - $H_1$: Normal traffic and peak traffic regimes are best represented by distinct probability distributions.

---

## 2. Dataset Profile Assessment & Preprocessing

This study profiles and compares two independent log datasets to analyze traffic behavior under different operational loads:

| Metric | Dataset 1: Web Server Access Logs | Dataset 2: E-Commerce Website Logs |
| :--- | :--- | :--- |
| **File Source** | `Web Server Log - DS dataset/` (`access_logs.csv`) | `E-Commerce Website Logs.zip` (`E-commerce Website Logs.csv`) |
| **Row Count** | 5,000 rows | 172,838 rows |
| **Timestamp Field** | `timestamp` (Resolution: Seconds) | `accessed_date` (Resolution: Milliseconds) |
| **Time Span** | 2024-01-01 to 2024-02-06 (36 days) | 2017-03-14 to 2017-03-21 (7.25 days) |
| **Traffic Vol. (Raw)**| Sparse: Mean of ~1.07 req/min, max of 3 | High: Mean of ~17.35 req/min, max of 288 |
| **Project Role** | **Secondary (Baseline)**: Low-volume background traffic | **Primary (Target)**: High-volume traffic representative of flash-sales |

### Preprocessing & Alignment
- **Data Cleaning**: String anomalies in E-Commerce `age` (`'--'`, 14,743 rows) were converted to `NaN`. Web Server access logs were joined with server metadata (`servers.csv`).
- **Temporal Resolution**: Both datasets were aggregated to 1-minute intervals across their full time span ($N = 10,457$ minutes for E-Commerce; $N = 52,625$ minutes for Web Server). Unobserved minutes were filled with `0` to prevent artificial inflation of mean traffic.

---

## 3. Exploratory Data Analysis & Empirical Results

Traffic was partitioned into operational regimes:
- **Normal Regime**: Observations below the 90th percentile of requests/minute.
- **Peak Regime**: Observations at or above the 90th percentile of requests/minute (operationally defined for this study; for Web Server logs where $P_{90} = 0.0$, peak is defined as active minutes $\ge 1.0$ req/min).

### Empirical Requests/Minute Summary

#### E-Commerce (Primary Dataset, $N = 10,457$)
- **Overall Traffic**: Mean = $16.5285$, Median = $14.0$, Variance = $221.6593$, Std Dev = $14.8882$, Skewness = $3.9765$, Kurtosis = $34.3407$, Dispersion $D = 13.4108$
- **Normal Regime (<30 req/min, $N = 9,255$)**: Mean = $12.7302$, Median = $12.0$, Variance = $53.3792$, Skewness = $0.1848$, Kurtosis = $-0.7766$, Dispersion $D = 4.1931$
- **Peak Regime ($\ge 30$ req/min, $N = 1,202$)**: Mean = $45.7737$, Median = $38.0$, Variance = $551.4975$, Skewness = $3.5095$, Kurtosis = $20.1850$, Dispersion $D = 12.0483$

#### Web Server (Secondary Dataset, $N = 52,625$)
- **Overall Traffic**: Mean = $0.0950$, Median = $0.0$, Variance = $0.1005$, Std Dev = $0.3170$, Skewness = $3.5185$, Kurtosis = $13.5215$, Dispersion $D = 1.0574$

---

## 4. Statistical Modeling & Model Evaluation

### Dispersion & Goodness-of-Fit Validation

#### A. Primary Dataset (E-Commerce)
- **Dispersion Index**: $D = 13.4108 \gg 1.0$, proving severe overdispersion ($\text{Variance} \gg \text{Mean}$).
- **GoF Test Behavior**: Due to the large sample size ($N = 10,457$), formal Chi-Square and Kolmogorov-Smirnov tests reject $H_0$ ($p < 0.05$) for all candidate distributions. Therefore, no theoretical distribution represents a "statistically perfect fit" under formal hypothesis testing. Model evaluation relies on relative performance via **AIC/BIC**.
- **Model Comparisons**:
  - **All Traffic**: **Log-Normal** is the best relative model ($\text{AIC} = 78,824.03$), followed by **Negative Binomial** ($\text{AIC} = 79,029.45$) and **Exponential** ($\text{AIC} = 79,583.51$). Poisson performs worst ($\text{AIC} = 150,650.87$).
  - **Peak Traffic ($\ge 30$ req/min)**: **Gamma** is the best relative model ($\text{AIC} = 1,082.39$), outperforming Log-Normal ($\text{AIC} = 1,462.37$) and Negative Binomial ($\text{AIC} = 10,292.40$). This supports the regime shift hypothesis ($H_1$ of Hypothesis 3).

#### B. Secondary Dataset (Web Server)
- **Dispersion Index**: $D = 1.0574 \approx 1.0$, suggesting approximate equidispersion.
- **GoF Validation**: Although $D \approx 1.057$, the Poisson distribution is statistically rejected by Chi-Square GoF ($\chi^2 = 101.45, p = 9.33\times 10^{-23} < 0.05$). In contrast, **Negative Binomial** yields $\chi^2 = 1.57, p = 0.210 > 0.05$, representing a **statistically acceptable fit**. This confirms that $D \approx 1$ alone does not guarantee a Poisson fit.

---

## 5. Capacity Planning & Extreme Event Probabilities

We evaluate the mathematical survival probabilities $P(X > T)$ for extreme traffic thresholds ($T = 100$ req/min) using actual fitted parameters on E-Commerce All Traffic ($N = 10,457$).

### Empirical Benchmark
- **Count of intervals $> 100$ req/min**: $45$
- **Empirical Probability $P_{\text{emp}}(X > 100)$**: $45 / 10,457 = 4.3033 \times 10^{-3}$ ($0.4303\%$)
- **Average Time Between Spikes**: $10,457 / 45 = 232.38\text{ minutes} \approx 3.87\text{ hours}$ (i.e., **once every 3.87 hours**)

### Fitted Survival Probabilities $P(X > 100)$
- **Poisson** ($\lambda = 16.5285$): $9.2369 \times 10^{-45}$ (Catastrophic underestimation by 42 orders of magnitude).
- **Normal** ($\mu = 16.5285, \sigma = 14.8882$): $1.0320 \times 10^{-8}$ (Severe underestimation).
- **Gamma** (Shifted, $a = 0.0894, \text{scale} = 3.6430$): $5.3176 \times 10^{-15}$ (Underestimates tail when fitted on all traffic).
- **Negative Binomial** ($n = 1.7170, p = 0.0941$): $2.7321 \times 10^{-4}$ ($0.0273\%$).
- **Log-Normal** (Shifted, $s = 0.5611, \text{loc} = -4.4413, \text{scale} = 18.6821$): $1.0205 \times 10^{-3}$ ($0.1021\%$).
- **Exponential** ($\text{scale} = 16.5285$): $2.3575 \times 10^{-3}$ ($0.2357\%$).

### Capacity Planning Takeaways
1. **Failure of Poisson**: The Poisson model estimates $P(X > 100) \approx 10^{-45}$, declaring extreme surges impossible. Sizing infrastructure on Poisson assumptions leads to severe under-provisioning.
2. **Tail Estimates**: **Exponential** ($2.36\times 10^{-3}$), **Log-Normal** ($1.02\times 10^{-3}$), and **Negative Binomial** ($2.73\times 10^{-4}$) provide realistic tail estimates orders of magnitude closer to the empirical rate ($4.30\times 10^{-3}$). While Exponential is closest at threshold 100, **Log-Normal** maintains the best overall relative fit across the entire distribution ($\text{AIC} = 78,824.03$).
