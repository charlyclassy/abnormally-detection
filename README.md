

# 🧠 AI Anomaly Detection — Deep Learning Autoencoder

This project uses an Autoencoder Neural Network to detect unusual patterns in sensor data.

## 🚀 Launch Live Application

[👉 CLICK HERE TO OPEN AI ANOMALY DETECTION](https://abnormally-detection-a2cgk2vqugf3mkhs675ag2.streamlit.app/)

## 🤖 Deep Learning Model

**Model:** Autoencoder Neural Network

**Framework:** TensorFlow / Keras

**Application:** Anomaly Detection

## 📊 Features

- Upload CSV datasets
- Train an Autoencoder neural network
- Detect unusual data patterns
- Visualise anomalies
- Train, validate and test the model
- Evaluate precision, recall and F1-score
- Download anomaly detection results




> **Holdout v4:** This release includes a visible 60/20/20 training-flow graphic at the top of the Streamlit page. If you do not see the graphic, check that the latest `app.py` was committed to GitHub and that Streamlit has redeployed.

# 🔎 AnomalyLab Pro — Deep Learning Anomaly Detection

An advanced Streamlit dashboard using a **TensorFlow/Keras dense autoencoder**, a deep learning neural network, to detect unusual patterns in numeric measurements. It is **not a CNN or a Transformer**.

## 🚀 Live app

[**Launch AnomalyLab Pro**](https://aianomaly-detection-ykrwhxcbwcvrrw6h78tx2m.streamlit.app/)

## Evaluation protocol (new)

1. Hold out **20% of all complete records** as an untouched test set (stratified for labelled data).
2. For labelled data, train the autoencoder on **normal records only** from the remaining portion; exclude known anomalies from training and validation.
3. Use a separate normal **validation split** for early stopping and selection of a reconstruction-error percentile threshold.
4. Score **test rows only**, and compute test precision, recall, F1, accuracy and a confusion matrix when labels are available.
5. Changing the threshold after viewing test metrics makes the test a tuning set; fix the threshold before evaluating it.

**Important:** The split is random and appropriate only for approximately independent rows. For time series, temporally adjacent records, or multiple machines/patients, use time-ordered or group-based splits to avoid leakage. This sample illustrates the method and is not a production validation.

## Features

- Built-in **labelled** sensor dataset, or upload your own CSV.
- Select model features, with labels deliberately excluded to avoid leaking answers.
- Train an autoencoder to reconstruct normal numeric patterns.
- **Change the anomaly threshold after training** without training again.
- Interactive reconstruction error, training loss, and flagged-row charts.
- Inspect **largest-error feature** for each detected record; this is a diagnostic hint, not a proven cause.
- If your CSV has a 0/1 ground-truth column, show **precision, recall, F1, and a confusion matrix**.
- Download all scored records or just flagged rows.

## Model architecture

Dense encoder → compressed latent features → dense decoder. Input variables are standardized based on the training subset. Reconstruction mean-squared error is used as an anomaly score; a percentile of **held-out normal reference errors** is used as the threshold. When 0/1 labels are available, training examples are selected from labelled normal rows, without using labels as input.

**Accuracy note:** The dashboard evaluates only held-out test rows, not training or validation rows. Synthetic labels and a single random split are educational; confidence in real-world deployment requires separate external testing. Anomalies are not necessarily real equipment faults.

## CSV format

Use at least **100 complete numeric rows** (and at least **80 normal plus 7 anomalies** for labelled evaluation). Optionally include a `timestamp` column and a binary `actual_anomaly` column (`0` = normal, `1` = anomaly). Without labels, the model still detects candidate anomalies, but evaluation scores are unavailable. Keep datasets within 10,000 rows and 25 MB for this community-cloud demo.

## Deploy on Streamlit Community Cloud

1. Create or open your GitHub repository.
2. Upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and this README, preserving `.streamlit` as a folder.
3. On [Streamlit Community Cloud](https://share.streamlit.io), select the repository, branch `main`, and entrypoint `app.py`.
4. **Select Python 3.12** in Advanced settings. Python 3.14 is incompatible with the specified TensorFlow range.
5. Deploy and test the sample dataset.

If updating an existing deployment, commit these files to the same GitHub repository; Streamlit Cloud normally redeploys automatically. If its Python version is still 3.14, redeploy with 3.12.

## Local use

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Responsible usage

Do not infer causality or production safety from reconstruction scores alone. Validate flags with domain-specific experts, and avoid uploading confidential datasets to public services without appropriate safeguards.
