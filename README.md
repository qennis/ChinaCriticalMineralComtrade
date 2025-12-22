# ChinaCriticalMineralComtrade

A comprehensive pipeline for analyzing China’s critical mineral exports using UN Comtrade data. This project goes beyond basic trade flows to detect **strategic behavior**, **trade deflection**, **evasion (leakage)**, and **price discrimination** following China's 2023–2024 export controls on Gallium, Germanium, and Graphite.

The repository performs three main functions:
1.  **Pulls** granular HS6 trade data from the Comtrade v1 API.
2.  **ETLs** raw data into cleaned panels, aggregating by material groups and calculating concentration metrics (HHI).
3.  **Analyzes** the data through a suite of advanced economic forensics modules (Stockpiling, Leakage, Price Divergence, etc.).

---

## Key Features

### 1. Core Trade Analysis
* **Concentration:** Calculates Herfindahl-Hirschman Indices (HHI) at both the material-group and HS6 product-line levels.
* **Regime Classification:** Distinguishes between competitive "commodity-like" markets and speculative "scarcity" regimes using price-volume correlations.

### 2. Strategic Behavior Forensics (`make_strategic_analysis.py`)
* **Anticipatory Stockpiling:** Detects abnormal volume surges in the 3-month window preceding export controls (with propagated error bars).
* **Trade Deflection:** Quantifies the net volume shift ($kg$) to third-party countries post-ban.
* **Mass & Price Outliers:** Identifies specific partners experiencing significant absolute deviations in quantity or unit price.
* **The "Friendship Premium":** Measures the volume-weighted price divergence between "Adversary" blocs (e.g., US, Japan, EU) and "Intermediary" blocs.

### 3. Leakage & Evasion Tracking (`make_leakage_figures.py`)
* **"Black Hole" Analysis:** Explicitly tracks flows into "Unspecified" (Area 999) and "Bonded Zones" (China 156) to estimate evasion magnitude.
* **True Leakage:** Calculates the gap between reported global totals and tracked destination flows.

### 4. Advanced Dynamics (`make_advanced_figures.py`)
* **Event Studies:** Generates $\pm$12-month value indexes centered on implementation dates (Aug 1, 2023 for Ga/Ge; Dec 1, 2023 for Graphite).
* **Peer Comparisons:** Benchmarks strategic minerals against non-controlled industrial peers (e.g., Graphite vs. Polysilicon) to isolate policy effects.
* **Price Divergence:** Splits analysis by material family to show specific premiums for Gallium/Germanium vs. Graphite.

### 5. Deep Dives (`make_deep_dive.py`)
* **Composition Evolution:** Visualizes how the product mix (e.g., Unwrought vs. Compounds) changes for key partners over time.
* **Volatility Bands:** Plots HS6-level unit values with rolling standard deviation bands to highlight pricing uncertainty.

---

## Repository Structure

* `src/china_ir/`: Core library code.
    * `comtrade.py`: API client.
    * `etl.py`: Data cleaning and aggregation pipeline.
* `scripts/`: Analysis workflows.
    * `pull_from_map.py`: **Main data fetcher.**
    * `make_figures.py`: Generates standard overview figures (concentration, growth).
    * `make_strategic_analysis.py`: Generates stockpiling, deflection, and outlier figures.
    * `make_leakage_figures.py`: Generates leakage stacks and destination tracking.
    * `make_advanced_figures.py`: Generates event studies and peer comparisons.
    * `make_deep_dive.py`: Generates partner-specific composition and price drilldowns.
* `data_work/`: Storage for raw Parquet pulls and processed tables.
* `figures/`: Output directory for all generated plots.

---

## Installation

Requires Python 3.10+ (3.11 recommended).

```bash
# Create environment
conda create -n china-ir python=3.11
conda activate china-ir

# Install dependencies
conda env create -f environment.yml
````

-----

## End-to-End Workflow

### 1\. Pull Data

Fetch monthly HS6 data for all codes defined in `notes/hs_map.csv`.

```bash
PYTHONPATH=src ./scripts/pull_from_map.py \
  --freq M \
  --period 201801-202412 \
  --flow X \
  --reporter 156 \
  --partner 0
```

*Note: Run `scripts/pull_partners.py` if you need granular partner-level data for the strategic/leakage scripts.*

### 2\. Build Tables

Clean raw data and generate master tables (`materials_monthly.parquet`, `hs6_hhi.parquet`, etc.).

```bash
python src/china_ir/etl.py \
  --in-glob 'comtrade_CM_HS_*_MAP_*.parquet' \
  --hs-map notes/hs_map.csv \
  --outdir data_work
```

### 3\. Generate Analysis & Figures

Run the suite of analysis scripts to populate `figures/`.

**Standard Overview:**

```bash
python scripts/make_figures.py
```

**Strategic Forensics (Stockpiling, Deflection, Premiums):**

```bash
python scripts/make_strategic_analysis.py
```

*Outputs to `figures/strategic/`*

**Leakage & Evasion:**

```bash
python scripts/make_leakage_figures.py
```

*Outputs to `figures/leakage/`*

**Advanced Market Dynamics:**

```bash
python scripts/make_advanced_figures.py
```

*Outputs to `figures/advanced/`*

**Deep Dives (Composition & Volatility):**

```bash
python scripts/make_deep_dive.py
```

*Outputs to `figures/deep_dives/`*

-----

## Outputs

After running the full pipeline, your `figures/` directory will contain:

  * **Strategic:** `stockpiling_gallium_germanium.png`, `deflection_graphite.png`, `premium_gallium.png`
  * **Leakage:** `leakage_unwrought_ga_ge.png`, `leakage_natural_graphite.png`
  * **Advanced:** `event_study_comparison.png`, `divergence_overview_graphite.png`
  * **Deep Dives:** `composition_USA.png`, `price_hs6_811292.png`
  * **Overview:** `hhi_monthly.png`, `monthly_stack.png`, `price_quantity_regimes.png`

-----

## License & Citation

If you use this code or the derived figures in a paper, please cite the corresponding project/paper.
