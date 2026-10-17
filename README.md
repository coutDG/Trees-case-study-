# Random Forest vs XGBoost vs LightGBM vs CatBoost

A practical comparison of tree-based models across **Home Credit, Adult Census Income, Amazon Employee Access and HIGGS**.

The study explores how predictive performance, training time and inference time change across mixed data, categorical IDs and large numerical data. It includes preprocessing, feature engineering, native categorical handling, fixed-budget Optuna searches and held-out evaluation.

**Main finding:** the strongest model depends on the dataset and the operational goal. The fastest model to train is not necessarily the fastest to predict.

## Questions explored

- Where do Random Forest, XGBoost, LightGBM and CatBoost perform well?
- Does native categorical handling improve results over one-hot encoding?
- What does tuning achieve within a fixed time budget?
- How do training and inference costs differ?
- How do the findings change between categorical and numerical datasets?

## Datasets and experiments

| Dataset | Prediction task and data | Experiment |
|---|---|---|
| **Home Credit** | Payment difficulty; 307,511 original application records with mixed predictors and substantial missingness | EDA, feature engineering, engineered versus filtered features, one-hot versus native inputs, baselines and tuned models; approximately 30 minutes per tuning study |
| **Adult Census Income** | Income above $50K; 32,537 records after removing 24 exact duplicates; 5 numerical and 7 categorical predictors | Random Forest with one-hot encoding; boosting models with native categories; tuned models only, approximately 10 minutes per model |
| **Amazon Employee Access** | Access approval; 32,769 labelled records and 9 categorical ID predictors | Random Forest with one-hot encoding; boosting models with native categories; tuned models only, approximately 10 minutes per model |
| **HIGGS TFRecord shards** | Simulated signal versus background; 28 numerical predictors; 500,000 validation and 500,000 test events | Shards converted to NumPy; three fixed 100-tree boosting baselines; no Optuna or early stopping; Random Forest excluded |

The HIGGS experiment uses the downloaded TFRecord version. Its converted row count and training size should be taken from the preprocessing outputs, rather than assumed to equal the original UCI dataset's 11 million rows. Its random split is not claimed to reproduce the original UCI test ordering.

## Main results

### Test ROC-AUC

| Dataset | Random Forest | XGBoost | LightGBM | CatBoost |
|---|---:|---:|---:|---:|
| Home Credit: tuned one-hot pipelines | 0.7506 | **0.7636** | 0.7635 | 0.7635 |
| Adult Income: tuned pipelines | 0.9127 | 0.9228 | **0.9231** | 0.9211 |
| Amazon Access: tuned pipelines | 0.8023 | 0.8536 | 0.8607 | **0.8809** |
| HIGGS: fixed baselines | Not tested | **0.8135** | 0.8100 | 0.8008 |

Bold values identify the highest observed score, not necessarily a statistically decisive winner. AUC values across different prediction tasks should not be used to rank task difficulty.

### What changed across datasets?

- **Home Credit:** the top three boosting models were closely matched. CatBoost led average precision, while XGBoost had the highest observed AUC. Native categorical handling showed no clear AUC advantage under the study budgets.
- **Adult Income:** LightGBM led the observed test metrics, with XGBoost very close. Their paired AUC interval included zero. Several categorical columns did not automatically make CatBoost the winner.
- **Amazon Access:** CatBoost showed its clearest quality advantage. Its test log loss was **0.1426**, approximately **6.6% lower** than LightGBM's. Paired bootstrap intervals supported lower loss than each alternative.
- **HIGGS:** XGBoost led prediction quality. LightGBM fitted **3.8% faster** than XGBoost in the recorded runs, a modest observed advantage. CatBoost had the fastest measured inference.

These results describe the fitted pipelines. They do not isolate dataset size or categorical encoding as the sole cause of a performance difference.

## Training and inference trade-offs

| Experiment | Fastest recorded final fit | Fastest observed batch inference |
|---|---|---|
| Home Credit | LightGBM among the one-hot finalists | One-hot CatBoost; inference thread defaults require an audit |
| Adult Income | XGBoost: **1.19 s** | CatBoost: approximately **8.8 ms per 1,000 rows**, with XGBoost close at 9.5 ms |
| Amazon Access | LightGBM: **1.68 s** | CatBoost: **19.90 ms per 1,000 rows**, including input preparation |
| HIGGS | LightGBM: **92.59 s**, versus XGBoost at 96.25 s | CatBoost: **0.432 ms per 1,000 rows**, approximately 2.14 times XGBoost's measured throughput |

Timings should be compared **within each experiment**. Batch sizes and measurement boundaries differ: Amazon includes input preparation, while the reported Adult and HIGGS benchmarks measure model prediction on prepared inputs. Batch measurements scaled to 1,000 rows are not single-request latency measurements.

Final fit time and Optuna search time are reported separately. Equal time budgets allow different numbers of completed trials, and a running trial can finish after the nominal timeout.

## Workflow

1. **Split the data:** create fixed training, validation and test assignments shared by every model in an experiment.
2. **Preprocess:** learn imputation, category definitions and feature filtering from training data only.
3. **Train:** use validation data for configuration choices, tuning and early stopping where enabled.
4. **Freeze decisions:** select classification thresholds on validation data before test evaluation.
5. **Evaluate:** report final test metrics and timing without retuning on the test set.

EDA was performed on training data for Home Credit and Adult. Amazon and HIGGS used simpler preprocessing workflows. TensorFlow was used only to decode the HIGGS TFRecord shards; model training uses classical ML libraries and numerical arrays.

### Evaluation outputs

- ROC-AUC and average precision (AP)
- Log loss and Brier score
- Precision, recall, F1 and accuracy at validation-selected thresholds
- ROC, precision-recall, confusion-matrix and calibration plots
- Training and repeated, warmed-up inference timings
- Feature importance and paired bootstrap comparisons where included

AP is the average-precision summary, not trapezoidal PR-AUC. The primary selection metric also differs: Home Credit and Adult use validation ROC-AUC, while Amazon uses validation log loss. HIGGS reports fixed baseline configurations.

## Project organisation

Each dataset uses its own study folder. Notebook names differ between experiments; run them in workflow order.

| Location inside a study folder | Contents |
|---|---|
| `data/raw/` | Original downloaded data |
| `data/splits/` | Saved split files or row assignments |
| `data/processed/` | Prepared inputs where materialized, including HIGGS NumPy arrays |
| `notebooks/` | Splitting, EDA where applicable, preprocessing, training and evaluation |
| `artifacts/` | Saved models, preprocessing objects and input metadata |
| `optuna/` | Persistent tuning studies for experiments using Optuna |
| `results/validation/` | Validation metrics and tuning histories |
| `results/test/` | Final comparison tables |
| `results/predictions/` | Saved probabilities and row identifiers |
| `results/figures/` | Evaluation graphs |

## Running the notebooks

**Python 3.11** was used for the study. A starting environment can be installed with:

```bash
python -m pip install numpy pandas scikit-learn xgboost lightgbm catboost optuna joblib matplotlib seaborn jupyter ipykernel
```

HIGGS TFRecord conversion additionally requires TensorFlow:

```bash
python -m pip install tensorflow
```

These commands install dependencies; they are not an exact reproduction lockfile. Match the package versions recorded by each evaluation notebook when reproducing published results.

1. Download the relevant dataset from its source and place it in the study's `data/raw/` folder.
2. Set `PROJECT_DIR` in the notebooks to the local study directory.
3. Run splitting and preprocessing before model training.
4. Run model training before evaluation; evaluation expects the saved artifacts.
5. Check the generated CSV tables and figures in `results/`.

For HIGGS, preserve the downloaded shard filenames, decode them in batches and save compact numerical arrays. Train models sequentially to control memory pressure. Memory-mapped arrays reduce loading copies but do not eliminate model working-memory requirements. Restart the preprocessing kernel before training to release TensorFlow's memory.

Record CPU/GPU device settings, hardware and thread counts alongside timings. GPU results should be labelled separately from CPU comparisons.

## Dataset sources

- [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- [Adult — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) — this experiment uses the 32,561-row input version before deduplication, not the full combined UCI train/test collection.
- [Amazon Employee Access Challenge — Kaggle](https://www.kaggle.com/competitions/amazon-employee-access-challenge/data)
- [Higgs Boson TFRecord dataset — Kaggle](https://www.kaggle.com/datasets/ryanholbrook/higgs-boson)
- [Original HIGGS dataset — UCI](https://archive.ics.uci.edu/dataset/280/higgs)

Original datasets, generated arrays and large model files should be downloaded or regenerated locally rather than bundled into the repository. Dataset access and reuse remain subject to the original providers' terms.

## Limitations

- This is a practical case study, not one uniform benchmark: preprocessing, tuning objectives, budgets and configurations differ between datasets.
- Most comparisons use one split and one training seed. Bootstrap intervals describe test-sample uncertainty for fixed models, not variability from retraining.
- Home Credit representation comparisons sometimes use different selected feature sets, so they do not isolate encoding alone.
- A suspected erroneous Home Credit income row was removed from training without independent confirmation of the error.
- Random Forest was not evaluated in the final HIGGS comparison.
- Equal tree counts do not imply equal capacity or computational work.
- Training timings are single recorded fits; small differences need repeated measurements.
- Findings do not establish a universally superior algorithm or production readiness.

## Conclusion

**CatBoost showed its strongest quality advantage on Amazon's categorical IDs. XGBoost and LightGBM remained strong on mixed and numerical data. Training speed, inference speed and prediction quality often favoured different models.**

Model choice should reflect the data, compute budget and operational objective, with quality and timing assessed together.
