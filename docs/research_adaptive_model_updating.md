# Research on Adaptive Machine Learning Model Updating in Automated Data Pipelines

## 1. Introduction to the Research Problem

In real-world machine learning systems, data frequently changes over time. Such changes may cause models trained on historical data to gradually lose predictive accuracy. This phenomenon is commonly referred to as Concept Drift.

To maintain model performance, a system must be able to monitor data changes, detect signs of performance degradation, and determine when to update or retrain a model (Model Retraining). However, retraining too frequently can increase computational costs, processing time, and system resource consumption. Conversely, if updates are too slow, the model may continue to produce inaccurate predictions.

Therefore, the research problem involves not only detecting Concept Drift, but also determining when a model should be updated, which method should be used, and how predictive performance can be balanced against operational costs.

This study focuses on adaptive machine learning model updating methods in automated data pipelines, aiming to maintain model performance as data changes over time while taking system resource and latency constraints into account.

## 2. Theoretical Background

| Concept | Item | Role in the Research |
| --- | --- | --- |
| Concept Drift | A change over time in relationships or data distributions that may degrade the performance of the current model. | Defines the problem to be detected and addressed. |
| Data Drift | A change over time in the distribution of input data. | Monitors changes in data within the pipeline. |
| Drift Detection | The use of statistical methods or performance metrics to detect changes. | Provides signals for model evaluation and updating. |
| Model Retraining | Retraining a model using new or recent data. | Restores or improves predictive performance. |
| Incremental Learning | Updating a model incrementally as new data arrives, rather than retraining it entirely. | An adaptive approach that may reduce the need for full retraining. |
| Adaptive Learning | The ability to adjust a model or learning strategy to adapt to a changing data environment. | The overall direction of this project. |
| Prequential Evaluation | Sequential evaluation: make predictions first, then use actual labels to calculate errors and update. | Suitable for evaluating models on sequential data streams. |
| Retraining Cost | Resource, time, and processing costs incurred during retraining. | A factor to consider when selecting an update policy. |

## 3. Analysis of Two Key Studies

### 3.1. Study 1: When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints (2026)

| Criteria | Analysis |
| --- | --- |
| Authors | Sawan Dasari |
| Objective | Investigates when to retrain models in streaming-data environments with Concept Drift, limited retraining budgets, and training/deployment latency. |
| Method | Compares Periodic Retraining, Error-threshold Retraining, ADWIN-triggered Retraining, and No-retrain. |
| Experimental Data | Synthetic data, LUFlow, and LendingClub. |
| Experimental Setup | Three drift types: abrupt, gradual, and recurring; multiple budget and latency levels; comparison of models with and without Incremental Learning. |
| Main Results | When the model updates each sample through Incremental Learning, practical differences between retraining policies are not significant for the studied linear models and immediate-label setting. Without Incremental Learning, retraining-policy choice significantly affects post-drift performance. |
| Notable Findings | Periodic Retraining performs better than reactive methods in many abrupt and gradual drift cases without Incremental Learning. Reactive methods retain advantages in some recurring-drift cases. |
| Limitations | Results depend on linear models, the update mechanism, and the assumption of immediate labels. They cannot be assumed to generalize to other models or delayed-label environments. |
| Relevance to This Project | Provides a basis for comparing model-update policies and highlights the need to evaluate Incremental Learning, retraining costs, and deployment latency. |

Commentary: The study shows that no single retraining policy is optimal in every situation. Effectiveness depends on the type of drift, the ability to update incrementally, budget, and latency. Therefore, this project should evaluate update policies under specific operating conditions rather than relying solely on model accuracy.

### 3.2. Study 2: Cost-Effective Retraining of Machine Learning Models / Cost-aware retraining for machine learning (2023–2024)

| Criteria | Analysis |
| --- | --- |
| Authors | Ananth Mahadevan and Michael Mathioudakis |
| Publication | Preprint in 2023; published in Knowledge-Based Systems in 2024 as Cost-aware retraining for machine learning. |
| Objective | Develops a method for deciding whether to retrain a model or continue using the current model by balancing model-obsolescence costs against retraining costs. |
| Method | Proposes the Cost-Aware Retraining Algorithm (CARA), with variants CARA-T, CARA-CT, and CARA-P. |
| Decision Basis | Considers performance degradation on prediction queries and the resources required for retraining. |
| Experimental Data | Synthetic and real-world datasets, including Electricity, Covertype, and Airlines. |
| Main Results | CARA balances cost and performance; in reported experiments, it achieved query accuracy comparable to or better than baseline drift-detection methods while making fewer retraining decisions in many cases. |
| Notable Findings | CARA-P is a periodic update policy that may work well when data or queries are cyclical, but may cause unnecessary retraining when clear cyclicality is absent. |
| Limitations | Pairwise similarity computation can be resource-intensive for large datasets; retraining costs are assumed fixed in some settings. |
| Relevance to This Project | Provides a basis for update policies that consider not only drift signals but also the practical costs and benefits of retraining. |

Commentary: This study extends the approach from drift detection to cost-aware update decisions. It provides an important basis for considering resources, training time, and performance improvements when deciding whether to update a model.

## 4. Summary of Five Supporting Studies

| No. | Study | Key Content | Relevance to This Project |
| --- | --- | --- | --- |
| 1 | Grzenda et al. (2020) – Delayed Labelling Evaluation for Data Streams | • Evaluates models on data streams where actual labels do not arrive immediately after prediction.<br>• Emphasizes verification latency: the time between prediction and label confirmation.<br>• Evaluation must associate predictions with labels only when labels become available; future labels must not be used prematurely for error calculation or updates.<br>• Project relevance: test label delays of 0, 24, and 72 hours to measure their effects on performance monitoring and retraining decisions. | Helps model delayed labels and determine when MAE/RMSE can be calculated, models updated, or error-based signals generated. |
| 2 | Bayram, Ahmed & Kassler (2022) – From Concept Drift to Model Degradation: An Overview on Performance-Aware Drift Detectors | • Reviews performance-based detectors that use changes in error or prediction quality to identify model degradation.<br>• Distinguishes changes in data distribution from actual model-performance degradation; these phenomena are related but not identical.<br>• Helps classify monitoring signals and select detectors based on label availability.<br>• Project relevance: combine feature drift (e.g., PSI/ADWIN on features) with error monitoring when labels arrive; DDM is a supervised error-based detector. | Supports drift-monitoring choices, distinguishes data drift from model degradation, and clarifies detector label requirements. |
| 3 | Cerqueira, Torgo & Mozetič (2020) – Evaluating Time Series Forecasting Models: An Empirical Study on Performance Estimation Methods | • Studies how to estimate time-series forecasting performance using different evaluation methods.<br>• Compares evaluation approaches such as cross-validation and out-of-sample evaluation for time series.<br>• Emphasizes preserving chronological order and preventing future information from entering the training set.<br>• Project relevance: use temporal split/rolling-origin for PM2.5; weather features must be information available at prediction time. | Provides a basis for temporal validation and leakage control in one-hour-ahead PM2.5 forecasting. |
| 4 | Demšar (2006) – Statistical Comparisons of Classifiers over Multiple Data Sets | • Presents a framework for statistically comparing machine-learning algorithms across datasets.<br>• Recommends the Friedman test for comparing multiple methods across datasets and the Wilcoxon signed-rank test for pairwise comparisons.<br>• Recommends post-hoc testing to identify differing method pairs and control for multiple comparisons.<br>• Project relevance: compare retraining policies using the same dataset, drift scenario, label delay, and seed; report effect sizes and p-values. | Provides a statistical basis for comparing policies rather than relying solely on mean MAE/RMSE. |
| 5 | Souza et al. (2020) – Challenges in Benchmarking Stream Learning Algorithms with Real-world Data | • Discusses challenges in benchmarking algorithms on real-world stream data.<br>• Emphasizes sequentiality, temporal dependence, and source-specific characteristics when designing evaluations.<br>• Suggests checking label quality, distributions, representativeness, and dataset suitability for the benchmark task.<br>• Project relevance: use Elec2 for classification/label-based drift and Beijing and Vietnam for PM2.5 forecasting; keep the two tasks separate. | Suggests assessing dataset suitability and keeping the Elec2 stream-classification benchmark separate from PM2.5 regression. |

## 5. Comparison and Critical Analysis

| Criteria | Study When to Retrain (2026) | Study CARA (2024) | Overall Assessment |
| --- | --- | --- | --- |
| Focus | Compares retraining policies under drift, budget, and latency constraints. | Optimizes the Retrain-or-Keep decision based on cost. | The two studies approach the problem from complementary perspectives. |
| Decision Mechanism | Periodic schedules, error thresholds, and the ADWIN detector. | Model-obsolescence costs and retraining costs. | Change detection can be combined with an assessment of update benefits. |
| Resource Considerations | Emphasizes retraining budgets and latency. | Emphasizes model-obsolescence costs and retraining resources. | Both computational costs and predictive effectiveness should be considered. |
| Adaptability | Also compares Incremental Learning and retraining policies. | Makes update decisions based on data, queries, and costs. | The appropriate method must be identified for each model and data type. |
| Limitations | Limited by model scope, the immediate-label assumption, and experimental setup. | Measurement computational costs and assumptions about retraining costs. | Must be validated in the project pipeline and datasets. |

The following observations can be drawn from these studies:

- Detecting Concept Drift does not necessarily mean that retraining must be performed immediately. A data change should trigger an update only when there is evidence that the update provides an appropriate benefit.
- Retraining policies should be evaluated simultaneously in terms of predictive performance, number of updates, training time, and resource consumption.
- Data and model characteristics substantially affect the effectiveness of update methods.
- For time-series data, splitting and evaluation must preserve chronological order.
- Label delay may affect when performance degradation can be detected and how the model can be evaluated.

## 6. Research Gaps

Based on the reviewed studies, several issues warrant further investigation within the scope of this project:

- Insufficient joint evaluation of performance and cost: Many methods focus on drift detection or accuracy improvement, while retraining costs and deployment latency also directly affect system effectiveness.
- Dependence on data characteristics: A policy may perform well under one type of drift but poorly under another. Therefore, evaluation is needed across datasets and tasks.
- Delayed labels: In real-world environments, labels may not arrive immediately after prediction, affecting performance monitoring and update decisions.
- Distinction between drift detection and retraining decisions: Detecting a change does not necessarily mean that the model has degraded to the point where retraining is required.
- Integration into automated pipelines: Update decisions must be connected to data ingestion, model evaluation, version management, and deployment.

These gaps represent research directions derived from the literature; they do not imply that prior studies have entirely failed to address these issues.

## 7. Proposed Research Direction

Based on the theoretical background and reviewed studies, this project proposes evaluating adaptive update methods within automated data pipelines.

| Research Branch | Data / Task | Models and Methods Awaiting Confirmation | Proposed Research Direction |
| --- | --- | --- | --- |
| ELEC2 | Stream classification | GaussianNB, Hoeffding Tree; ADWIN/DDM; Prequential Evaluation | Evaluate drift-detection ability and model-performance maintenance as data changes. |
| Beijing | One-hour-ahead PM2.5 forecasting | Persistence, LightGBM, Random Forest | Compare forecasting models and assess performance maintenance over time. |
| Vietnam | Air-quality forecasting | Persistence, LightGBM, River Linear Regression | Study model updating on real-world data over time. |

The models and methods listed in the table are subject to confirmation by the AI Team before approval and should not be considered a finalized approved list.

The overall direction includes:

- Monitor changes in data and model performance.
- Compare update policies such as Periodic Retraining, Error-threshold Retraining, Drift-triggered Retraining, and No-retrain.
- Consider Incremental Learning when supported by the model and deployment interface.
- Evaluate the effects of retraining cost, latency, and label availability timing.
- Develop a process for evaluating a new model before replacing the current model.

## 8. Preliminary Experimental Design

The design below is an initial proposal to guide experimentation. Parameters, thresholds, input data, and deployment criteria must be agreed upon with the Data Pipeline, AI, and MLOps teams before implementation.

### 8.1. Experimental Objectives

- Evaluate the ability to maintain model performance as data changes.
- Compare the effectiveness of different update policies.
- Measure the trade-off between predictive performance and retraining cost.
- Evaluate the integration of update policies into an automated pipeline.

### 8.2. Methods for Comparison

| Method | Mechanism | Purpose |
| --- | --- | --- |
| No-retrain | Keep the model unchanged without retraining during the experiment. | Baseline for assessing degradation without updates. |
| Periodic Retraining | Retrain at predefined intervals. | Evaluate periodic updating. |
| Error-threshold Retraining | Trigger retraining when an error metric exceeds a threshold. | Evaluate a policy based on performance degradation. |
| Drift-triggered Retraining | Trigger retraining when a drift detector issues an alert. | Evaluate a policy based on drift-detection signals. |
| Incremental Learning (if supported) | Update the model incrementally when new data and corresponding labels are available. | Evaluate adaptation without frequent full retraining. |

### 8.3. Proposed Experimental Workflow

1. Receive data from the Data Pipeline in chronological order.
2. Check the data, data types, features, target, and potential data leakage.
3. Train the initial model using the training dataset.
4. Feed new data into the system and generate predictions.
5. Monitor performance or drift signals using the selected method.
6. Trigger the corresponding update policy when its conditions are met.
7. Train or update the model using appropriate data.
8. Evaluate the new model before replacing the currently deployed model.
9. Record results, processing time, and model version information.

### 8.4. Evaluation Metrics

| Evaluation Group | Proposed Metrics | Meaning |
| --- | --- | --- |
| Classification Performance (ELEC2) | Accuracy, F1-score | Assess prediction quality for stream classification. |
| Regression Performance (PM2.5) | MAE, RMSE (µg/m³) | Assess one-hour-ahead PM2.5 forecasting error. |
| Drift Detection | Detection Delay; False Alarm Rate; number of alerts | Assess detection timing and false alarms; distinguish feature drift from error-based drift. |
| Update Cost | Retraining Count; model_retrain_duration_seconds (CPU seconds) | Measure retraining count and required CPU time per training run. |
| System Resources | time.process_time(); process_cpu_seconds_total; process_resident_memory_bytes | Monitor resources; retraining CPU time is measured using time.process_time(). |
| Prediction Latency | model_prediction_duration_seconds; latency (ms) | Measure wall-clock latency for /predict requests; the operational target is no more than 500 ms. |
| Deployment | Model Version; Candidate/Champion; rollback status | Track versions, promotion outcomes, and rollback to the champion. |

### 8.5. Evaluation and Comparison Methods

- Use Prequential Evaluation for streaming classification tasks where appropriate.
- For PM2.5 forecasting, prioritize chronological data splitting and evaluation on future periods; rolling evaluation or repeated holdout may be used.
- Ensure that all policies are compared under the same data conditions and experimental setup.
- Statistical tests appropriate to the comparison design may be used; Wilcoxon or Friedman tests are possible options depending on the number of methods and datasets.
- Report both the magnitude of differences and statistical results, rather than relying only on Accuracy or RMSE.

#### 8.5.1. Quantitative Experimental Design Matrix

The matrix below is the proposed configuration for the PM2.5 forecasting branch. Each policy × label-delay × dataset combination is run with 5 random seeds (11, 22, 33, 42, 55). Seed 42 is the default seed specified in CONTRACTS.md; the remaining seeds are used to assess stability. Sudden, gradual, and recurring drift are evaluated separately when drift periods are identified or controlled and generated.

The PM2.5 policies include: (1) No-retrain/static model; (2) Weekly retraining every 7 days; (3) ADWIN-triggered retraining; and (4) Hybrid weekly + ADWIN-triggered, with a cooldown of at most once every 24 hours. DDM is a labeled-error-based detector and is therefore evaluated separately in the ELEC2 classification branch or in an appropriate labeled setting; it is not treated as an unsupervised detector for PM2.5.

Label delays of 0, 24, and 72 hours are research stress tests. CONTRACTS.md specifies 1 hour as the operational reference delay for the PM2.5 pipeline; an additional test at 1 hour should therefore be included for Contract alignment. When labels are not yet available, supervised errors must not be calculated and those labels must not be used to update the model.

The PM2.5 datasets include Beijing Multi-Site Air Quality and Vietnam PM2.5 from Hanoi and Ho Chi Minh City. ELEC2 remains a separate streaming-classification experimental branch for evaluating label-based drift and is not combined with PM2.5 regression metrics.

#### 8.5.2. Statistical Testing

Policies are compared pairwise using the same dataset, drift scenario, label delay, and seed. For two policies, use a paired t-test when the distribution of differences meets the assumptions; otherwise, use the Wilcoxon signed-rank test. When comparing three or more policies across multiple conditions, use the Friedman test; if an overall difference is found, conduct pairwise post-hoc Wilcoxon tests with Holm–Bonferroni correction. Report p-values, effect sizes (paired Cohen’s d or rank-biserial correlation), and confidence intervals where available. Autocorrelated timestamps should not be treated as independent observations.

#### 8.5.3. Mandatory CPU-Time Measurement and Training Cost

According to Section 5.1 of CONTRACTS.md, model_retrain_duration_seconds is a mandatory metric and must be measured using CPU process time via time.process_time(). Record CPU time separately for initial training and each retraining event, and store cumulative CPU seconds for each policy and run. Example: cpu_start = time.process_time(); train_model(...); cpu_elapsed = time.process_time() - cpu_start. This value represents process CPU time, not elapsed wall-clock time.

CPU retraining time must be distinguished from wall-clock retraining duration and prediction latency. model_prediction_duration_seconds measures the actual time taken to process a /predict request; the Contract target latency is ≤500 ms. Where possible, report both CPU time and wall-clock time.

| Factor | Experimental Value |
| --- | --- |
| Policies | No-retrain; Weekly (7 days); ADWIN-triggered; Hybrid (Weekly + ADWIN) |
| Label Delay | 0, 24, and 72 hours (stress tests); add 1 hour for Contract alignment |
| PM2.5 Datasets | Beijing; Vietnam-Hanoi; Vietnam-Ho Chi Minh City |
| Number of Seeds | 5: 11, 22, 33, 42, 55 |
| Drift Scenario | Sudden; Gradual; Recurring (evaluated separately where suitable data/setup is available) |
| Experiment Runs | 4 policies × 3 label delays × 3 datasets × 5 seeds = 180 runs per scenario; across 3 scenarios (Sudden, Gradual, Recurring), the theoretical total is 540 runs. With the additional 1-hour delay (4 delay levels), this becomes 240 runs per scenario, or 720 runs in total. These are theoretical runs based on the proposed matrix, not experiments already conducted. |

### 8.6. Items Requiring Cross-Team Confirmation

| Item | Relevant Team | Items to Agree Upon |
| --- | --- | --- |
| Feature Contract | Data Pipeline + AI | Features, target, data types, and leakage-prevention rules for each dataset. |
| Model Interface | AI + MLOps | Training inputs, prediction outputs, and model-version management. |
| Evaluation | AI | Accuracy/F1 for classification; MAE/RMSE for regression; and time-based evaluation. |
| Drift Event | AI + MLOps | Event schema, detector identifier, and timestamp semantics. |
| Retraining | AI + MLOps | Trigger conditions, minimum data volume, and acceptance criteria for a new model. |
| Data Stream | Data Pipeline + AI | Chronological ordering, delayed data, and label availability timing. |

### 8.7. Mapping the Research Design to the Data/AI/MLOps Contract

For PM2.5, the task is one-hour-ahead forecasting (target_time = event_time + 1 hour), with LightGBM Regressor as the primary model and Persistence as the mandatory baseline. Moving Average and static LightGBM may be included as additional comparators if agreed upon by the team. The primary regression metrics are MAE and RMSE. The MLOps mock RandomForestClassifier on Breast Cancer is not the PM2.5 research model.

Input features must be consistently mapped between the standard pm25_hourly_features schema (e.g., pm25_lag_1h, pm25_lag_3h, pm25_lag_24h, rolling means, weekday, weather) and the aliases used in notebooks (e.g., pm25_lag1, dayofweek). Leakage must be prevented: weather features may only be used if they were available at prediction time; future observed weather must not be treated as if it were already available.

At the schema-adapter layer, retain valid missing values and do not arbitrarily impute or delete records; serving-time imputation is a separate step governed by the Model Interface Contract. Extreme but valid PM2.5 values must be retained for drift research. For forecasts with delayed labels, errors should only be calculated and models updated after the actual label becomes available.

The initial candidate must achieve a lower MAE than Persistence on the same test set. Compared with the champion, the candidate must improve MAE by at least 2%, have an RMSE no more than 1% higher than the champion, and meet latency ≤500 ms. Otherwise, retain the current champion; rollback is supported through the champion alias.

## 9. Conclusion

The reviewed studies show that adaptive machine learning model updating involves multiple factors and is not limited to Concept Drift detection. The choice of when and how to retrain must consider predictive performance, computational cost, latency, data characteristics, and the model’s update capabilities.

The two main studies provide complementary perspectives: Dasari’s study compares retraining policies under budget and latency constraints, while CARA focuses on balancing the cost of model obsolescence against retraining costs. The supporting studies provide a basis for stream-data evaluation, delayed-label handling, drift detection, and experimental-result testing.

Accordingly, this project aims to investigate and compare adaptive update methods across three data branches: ELEC2, Beijing, and Vietnam. The objective is to explore how model performance can be maintained in changing data environments while ensuring that the update process can be evaluated and integrated into automated data pipelines.

## 10. References

Dasari, S. (2026). When to Retrain: An Empirical Study of Retraining Policies for Streaming ML Under Concept Drift, Budget, and Latency Constraints. arXiv:2608.19488.

Mahadevan, A., & Mathioudakis, M. (2024). Cost-aware retraining for machine learning. Knowledge-Based Systems, 293, 111610. https://doi.org/10.1016/j.knosys.2024.111610

Bayram, F., Ahmed, B. S., & Kassler, A. (2022). From concept drift to model degradation: An overview on performance-aware drift detectors. Knowledge-Based Systems, 245, 108632.

Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. Machine Learning, 109, 1997–2028. https://doi.org/10.1007/s10994-020-05910-7

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.

Grzenda et al. (2020). Study on evaluating models on data streams with delayed labels.

Souza et al. (2020). Study on benchmarking and stream data.
