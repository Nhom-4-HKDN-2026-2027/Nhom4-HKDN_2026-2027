# Research on Adaptive Machine Learning Model Updating in Automated Data Pipelines

## 1. Research Background and Problem

In real-world systems, data changes over time. Such changes (Concept Drift/Data Drift) can cause a model trained on historical data to lose predictive performance. A system must therefore monitor data and model performance, detect degradation, and decide whether and when to update the model. Updating too frequently increases computational cost, processing time, and resource consumption; updating too slowly allows inaccurate predictions to persist. This research investigates adaptive model-update policies that balance predictive performance with retraining cost, operational latency, and resource constraints in automated data pipelines.

## 2. Theoretical Background and Related Studies

Key concepts include Concept Drift (changes over time in data distributions or relationships that can affect predictions), Data Drift (changes in input-data distributions), Drift Detection (identifying changes), Model Retraining (retraining with new or recent data), Incremental Learning (updating a model as new data arrives), Prequential Evaluation (predict first, then evaluate using the actual label), and Retraining Cost (the time and resources required to update a model).

The two main studies provide complementary perspectives. Dasari (2026) compares periodic, error-threshold, ADWIN-triggered, and no-retrain policies under different drift, budget, and latency conditions. The findings indicate that, without Incremental Learning, policy choice can substantially affect post-drift performance; no single policy is best in every setting. Mahadevan and Mathioudakis (2024) propose CARA, which balances model-obsolescence costs against retraining costs. In reported experiments, CARA achieved comparable or better query accuracy than baseline drift-detection methods while making fewer retraining decisions in many cases.

Supporting studies emphasize correct handling of delayed labels; distinguishing data drift from actual model degradation; preserving chronological order and preventing leakage in time-series forecasting; using statistical tests when comparing multiple policies; and checking the suitability of each dataset for stream-learning tasks.

## 3. Research Gaps and Proposed Direction

- Evaluate predictive performance together with update frequency, training time, resource consumption, and deployment latency.
- Test policy robustness across datasets and drift patterns: sudden, gradual, and recurring.
- Account for delayed labels and distinguish drift detection from the decision to retrain.
- Integrate update decisions into the pipeline, including data ingestion, monitoring, training, version management, evaluation, deployment, and rollback.

## 4. Proposed Data Branches

| Branch | Task / Dataset | Candidate Models and Methods |
| --- | --- | --- |
| ELEC2 | Streaming classification | GaussianNB, Hoeffding Tree; ADWIN/DDM; Prequential Evaluation |
| Beijing | One-hour-ahead PM2.5 forecasting | Persistence, LightGBM, Random Forest (subject to confirmation against the contract) |
| Vietnam | PM2.5 forecasting; Hanoi and Ho Chi Minh City | Persistence, LightGBM, River Linear Regression (subject to confirmation) |

ELEC2 is a separate classification branch and must not be combined with PM2.5 regression metrics. Under the Data/AI/MLOps Contract mapping, PM2.5 uses LightGBM Regressor as the primary model and Persistence as the mandatory baseline; additional models require team agreement.

*Research on Adaptive Machine Learning Model Updating | 1*

## PRELIMINARY EXPERIMENTAL DESIGN

## 5. Quantitative Experimental Matrix for PM2.5

Objective: compare update policies under matched dataset, label-delay, and random-seed conditions, measuring forecast quality as well as update cost. Each configuration is run with five random seeds: 11, 22, 33, 42, and 55 (seed 42 is the default specified in CONTRACTS.md).

| Factor | Experimental Configuration | Notes |
| --- | --- | --- |
| Policies (4) | 1) No-retrain/static model; 2) Weekly retraining (every 7 days); 3) ADWIN-triggered; 4) Hybrid: weekly + ADWIN | Hybrid cooldown: at most one update every 24 hours. |
| Label delay | 0, 24, and 72 hours; add 1 hour | 0/24/72 hours are stress tests; 1 hour is the operational reference in the Contract. Do not calculate supervised errors or update before labels arrive. |
| Datasets | Beijing Multi-Site Air Quality; Vietnam–Hanoi; Vietnam–Ho Chi Minh City | Three datasets for PM2.5 regression. |
| Random seeds | 11, 22, 33, 42, 55 | Five runs to assess stability. |
| Drift scenarios | Sudden; Gradual; Recurring | Evaluate separately when suitable data or controlled scenarios are available. |
| Total runs | 180 per scenario with 3 label delays; 240 per scenario with the additional 1-hour delay | 540 or 720 runs across three scenarios. These are theoretical counts from the proposed matrix, not completed experiments. |

Run-count formula: 4 policies × number of label delays × 3 datasets × 5 seeds. DDM is based on labeled errors and should therefore be evaluated separately in the ELEC2 classification branch or another suitable labeled setting; it is not treated as an unsupervised detector for PM2.5.

## 6. Experimental Workflow

1. Receive data in chronological order; validate the schema, data types, features, target, and potential leakage.
2. Train the initial model using the time-ordered training data.
3. Feed new data into the pipeline and generate predictions; record timestamps and model versions.
4. Monitor performance or drift signals; calculate errors only after the actual labels become available.
5. Apply the assigned policy to decide whether to keep the model, update incrementally, or retrain.
6. Evaluate the candidate model before replacing the champion; record timing, resources, results, and rollback status.

## 7. Evaluation Metrics and Deployment Criteria

| Evaluation Group | Metrics | Purpose |
| --- | --- | --- |
| PM2.5 performance | MAE, RMSE (µg/m³) | Measure one-hour-ahead forecast error. |
| Drift detection | Detection Delay; False Alarm Rate; number of alerts | Measure detection delay and false alarms. |
| Update cost | Retraining Count; model_retrain_duration_seconds | Measure update frequency and CPU time per training run. |
| Resources / latency | CPU, resident memory; model_prediction_duration_seconds | Monitor resource use and /predict processing time. |
| Deployment | Model Version; Candidate/Champion; rollback status | Track versions, promotion decisions, and rollback. |

The mandatory model_retrain_duration_seconds metric must be measured using CPU process time via time.process_time(). Record initial-training CPU time and each retraining event separately, and calculate cumulative CPU seconds per policy/run. Distinguish CPU time from wall-clock retraining duration and prediction latency. The /predict latency target is ≤500 ms.

Under the Contract, the initial candidate must achieve lower MAE than Persistence on the same test set. Compared with the champion, a candidate must improve MAE by at least 2%, have RMSE no more than 1% higher than the champion, and meet latency ≤500 ms. Otherwise, retain the current champion.

## 8. Statistical Evaluation and Items Requiring Confirmation

Compare policies using the same dataset, drift scenario, label delay, and seed. For two policies, use a paired t-test if assumptions are met; otherwise use the Wilcoxon signed-rank test. For three or more policies across multiple conditions, use the Friedman test; if an overall difference is found, conduct pairwise post-hoc Wilcoxon tests with Holm–Bonferroni correction. Report p-values, effect sizes, and confidence intervals where available. Do not treat autocorrelated timestamps as independent observations.

Before implementation, the Data Pipeline, AI, and MLOps teams should agree on feature/target definitions and leakage-prevention rules; model interfaces and version management; drift-event schema and timestamp semantics; trigger conditions, minimum data volume, and candidate acceptance criteria; and data ordering and label availability. For PM2.5 forecasting, use weather features only if they were available at prediction time; future observed weather must not be used as an input.

Conclusion: The project aims to develop and evaluate adaptive update strategies that consider not only drift signals but also predictive benefit, training cost, resource use, latency, and safe integration into automated pipelines.

*Research on Adaptive Machine Learning Model Updating | 2*