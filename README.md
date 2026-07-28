# Predicting Gallstone Disease from Body Composition & Blood Chemistry

End-to-end SQL + Python analytics project: raw clinical data → relational
database → SQL risk analysis → machine learning classification.

## Why this dataset

Most portfolio projects lean on Titanic, Iris, or housing prices. This one
uses real de-identified patient data (bioelectrical impedance body-composition
scans + standard blood panels, n=319) to predict gallstone disease — a
problem with genuine clinical stakes and messy, correlated features, which
is a better proxy for real DS work than a clean toy dataset.

**Source:** UCI Machine Learning Repository — Gallstone Disease Dataset.

**[Open the full analysis in Google Colab](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/gallstone_analysis.ipynb)**
*(replace `<your-username>/<your-repo>` once this is pushed — see "Publishing this yourself" below)*

## Pipeline

```
data/raw/gallstone_patients.csv   →   data/gallstone.db (SQLite)
                                            │
                    sql/01_exploratory_analysis.sql
                    (risk stratification, cohort analysis,
                     window functions, percentile ranking)
                                            │
                    sql/02_advanced_analysis.sql
                    (self-joins, correlated subqueries,
                     matched-pair comparison, composite risk scoring)
                                            │
                    scripts/model.py
                    (logistic regression, random forest,
                     gradient boosting + 5-fold CV)
                                            │
                    scripts/tune_and_explain.py
                    (GridSearchCV tuning + SHAP explainability,
                     including a per-patient explanation)
                                            │
                    outputs/ (ROC curves, feature importance,
                               SHAP summary, tuned results)
                                            │
                    gallstone_analysis.ipynb
                    (single notebook narrating the whole pipeline —
                     Colab-runnable end to end)
```

## Key SQL techniques demonstrated

- Window functions: `RANK()`, `PERCENT_RANK()`, `NTILE()`, running totals with `SUM() OVER`
- CTEs for readable multi-step transformations
- Correlated subqueries (within-cohort standardization, matched-pair lookups)
- Self-joins to find near-identical patients with opposite outcomes (confounder-spotting)
- Composite risk scoring by combining multiple `NTILE()` quartiles
- Risk-stratification logic (BMI bands, age cohorts, comorbidity load) translated into `CASE` + `GROUP BY`
- Business-style framing: "which risk band has the highest incidence rate?" rather than raw `SELECT *`
- One documented, real SQLite quirk (correlated `ORDER BY` inside a scalar subquery) and its portable workaround — worth mentioning in an interview as a debugging story

## Modeling depth

- Baseline comparison across 3 model families with 5-fold cross-validation
- Hyperparameter tuning via `GridSearchCV`
- SHAP for both global feature importance and a concrete per-patient explanation — "here's exactly why the model flagged this person," not just a bar chart

## Key findings

| Question | Answer |
|---|---|
| Baseline prevalence | 158 / 319 patients (49.5%) are gallstone-positive — balanced classes |
| Highest-risk BMI band | Overweight/obese bands carry disproportionately higher positive rates than normal-BMI patients |
| Best predictor overall | **C-Reactive Protein (CRP)**, an inflammation marker — outweighs BMI and even cholesterol markers |
| Second predictor | Vitamin D level |
| Best model | Random Forest / Gradient Boosting, ROC-AUC ≈ 0.85–0.89 on held-out test data |

The CRP finding is the standout insight: it suggests **systemic inflammation
tracks with gallstone risk more strongly than body-fat measures alone** —
a more interesting takeaway than "obesity causes gallstones," which is what
most people would guess going in.

## How to run

```bash
# 1. Load data into SQLite (already done — data/gallstone.db is included)
# 2. Explore the SQL
sqlite3 data/gallstone.db < sql/01_exploratory_analysis.sql
sqlite3 data/gallstone.db < sql/02_advanced_analysis.sql

# 3. Run the baseline modeling pipeline
pip install pandas scikit-learn matplotlib shap
python3 scripts/model.py

# 4. Run hyperparameter tuning + SHAP explainability
python3 scripts/tune_and_explain.py
```

Or just open `gallstone_analysis.ipynb` in Colab and run all cells — it does everything above in one narrated notebook.

### Interactive demo (Streamlit)

```bash
pip install streamlit joblib shap
python3 scripts/train_final_model.py   # trains + saves the model the app uses
streamlit run app.py
```

Lets anyone type in patient values and see a live risk score plus a SHAP
waterfall chart explaining that specific prediction — this is the part
worth linking directly in a recruiter message, since it's something they
can click and interact with in 10 seconds rather than reading code.

## Project structure

```
├── data/
│   ├── raw/gallstone_patients.csv   # cleaned source data
│   └── gallstone.db                 # SQLite database
├── sql/
│   ├── 01_exploratory_analysis.sql  # 7 core analysis queries
│   └── 02_advanced_analysis.sql     # self-joins, correlated subqueries
├── scripts/
│   ├── model.py                     # baseline model comparison
│   ├── tune_and_explain.py          # GridSearchCV + SHAP
│   └── train_final_model.py         # trains + saves the model used by app.py
├── app.py                           # Streamlit interactive risk predictor
├── app_model.joblib                 # saved tuned model (generated)
├── app_feature_info.json            # feature medians/ranges for sliders (generated)
├── gallstone_analysis.ipynb         # full pipeline, Colab-runnable
├── build_notebook.py                # regenerates the notebook from source
├── outputs/
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── example_patient_explanation.csv
│   ├── model_results.csv
│   ├── gridsearch_results.csv
│   └── tuned_model_summary.csv
└── README.md
```

## Publishing this yourself

1. `git init`, commit everything in this folder, push to a new GitHub repo under your account — this is what gives you a real, dated commit history as proof you built it.
2. In `README.md` and `gallstone_analysis.ipynb`, replace `<your-username>/<your-repo>` with your actual GitHub path so the "Open in Colab" link works.
3. Pin the repo on your GitHub profile.
4. Deploy the Streamlit app for free on [Streamlit Community Cloud](https://streamlit.io/cloud) (point it at `app.py` in your repo) so you have a live URL, not just code.
5. Post the CRP finding + a screenshot (or live link) of the app on LinkedIn — a real dataset, a counterintuitive finding, and something recruiters can click and use themselves is what gets attention.

## Next steps (roadmap)

- Migrate SQLite → Postgres and containerize with Docker for a "production-shaped" version
- Add GitHub Actions CI that re-runs `scripts/model.py` on every push
- Deploy `app.py` to Streamlit Community Cloud for a public live link
- Widen the hyperparameter grid on a machine with more CPU cores
