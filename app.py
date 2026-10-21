"""AnomalyLab Pro: autoencoder sensor anomaly detection with optional evaluation."""
import hashlib
import io
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AnomalyLab Pro | Autoencoder", page_icon="🔎", layout="wide")
st.markdown("""<style>
.stApp {background-color:#0b1220; color:#f8fafc;}
.block-container {max-width:1250px;padding-top:1.4rem;}
.hero {padding:30px;border:1px solid #334155;border-radius:20px;
background:linear-gradient(120deg,#172554,#312e81 65%,#164e63);margin-bottom:20px;}
.hero h1 {color:white;margin:0;font-size:2.5rem;}
.hero p {color:#cbd5e1;margin:8px 0 0 0;}
[data-testid="stMetric"] {background:#18253a;border:1px solid #334155;padding:12px;border-radius:12px;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔎 AnomalyLab Pro <span style="font-size:0.82rem;vertical-align:middle;background:#0f766e;padding:6px 10px;border-radius:99px;">Holdout v4</span></h1><p>Autoencoder neural network • Sensor monitoring • Explainable alerts • Model evaluation</p></div>', unsafe_allow_html=True)

@st.cache_data
def example_dataset(n=600, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    df = pd.DataFrame({
        "timestamp":pd.date_range("2026-01-01", periods=n, freq="h"),
        "temperature":21+3*np.sin(t/24)+rng.normal(0,.65,n),
        "vibration":3+.4*np.cos(t/16)+rng.normal(0,.16,n),
        "pressure":101+.7*np.sin(t/20)+rng.normal(0,.25,n),
        "power_usage":54+6*np.sin(t/24)+rng.normal(0,1.8,n),
    })
    picks = rng.choice(np.arange(35,n-35),size=20,replace=False)
    df["actual_anomaly"] = 0
    df.loc[picks,"actual_anomaly"] = 1
    df.loc[picks,"temperature"] += rng.choice([-11,12],len(picks))
    df.loc[picks,"vibration"] += rng.uniform(2.5,5,len(picks))
    df.loc[picks,"power_usage"] += rng.uniform(18,38,len(picks))
    return df

def build_model(n_features):
    tf.keras.utils.set_random_seed(42)
    wide=max(8,min(32,n_features*3))
    mid=max(4,min(12,n_features*2))
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(wide,activation="relu"),
        tf.keras.layers.Dense(mid,activation="relu"),
        tf.keras.layers.Dense(max(2,min(6,n_features)),activation="relu",name="latent_layer"),
        tf.keras.layers.Dense(mid,activation="relu"),
        tf.keras.layers.Dense(wide,activation="relu"),
        tf.keras.layers.Dense(n_features),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss="mse")
    return model

def fit_and_score(df, features, label_col, epochs):
    """Split before preprocessing; fit and calibrate only with normal training/validation rows."""
    numeric = df[features].replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna().all(axis=1)
    clean = numeric.loc[valid]
    if len(clean) < 100:
        raise ValueError("At least 100 complete numeric rows are required for a reliable split.")

    labels = None
    if label_col:
        labels = pd.to_numeric(df.loc[valid, label_col], errors="coerce")
        if not labels.isin([0, 1]).all():
            raise ValueError("Labels must contain only 0 (normal) and 1 (anomaly).")
        counts = labels.value_counts()
        if counts.get(0, 0) < 80 or counts.get(1, 0) < 7:
            raise ValueError("Labelled evaluation requires at least 80 normal and 7 anomaly records. Gather more data before testing.")

    # First split removes the untouched test set (20%). Stratify so its anomaly
    # prevalence resembles the source dataset. Split the rest into train/val.
    rest_idx, test_idx = train_test_split(
        clean.index.to_numpy(), test_size=0.20, random_state=42,
        stratify=labels.to_numpy() if labels is not None else None
    )
    if labels is not None:
        normal_rest = np.array([idx for idx in rest_idx if labels.loc[idx] == 0])
        if len(normal_rest) < 40:
            raise ValueError("Too few normal rows remain for training and validation.")
        train_idx, val_idx = train_test_split(
            normal_rest, test_size=0.25, random_state=43
        )
        excluded_anomalies = int(len(rest_idx) - len(normal_rest))
    else:
        # Unlabelled rows are assumed mostly normal. This assumption is not verified.
        train_idx, val_idx = train_test_split(rest_idx, test_size=0.25, random_state=43)
        excluded_anomalies = 0

    scaler = StandardScaler()
    x_train = scaler.fit_transform(clean.loc[train_idx]).astype("float32")
    x_val = scaler.transform(clean.loc[val_idx]).astype("float32")
    x_test = scaler.transform(clean.loc[test_idx]).astype("float32")

    tf.keras.backend.clear_session()
    model = build_model(len(features))
    history = model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=epochs, batch_size=min(32, len(x_train)), verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        )]
    )
    val_rebuilt = model.predict(x_val, verbose=0)
    reference = np.mean((x_val - val_rebuilt) ** 2, axis=1)
    test_rebuilt = model.predict(x_test, verbose=0)
    per_feature = (x_test - test_rebuilt) ** 2
    errors = np.mean(per_feature, axis=1)
    best = np.argmax(per_feature, axis=1)

    # Only test rows receive predictions. No training or validation predictions
    # enter the confusion matrix or downloadable scored test set.
    scored = df.loc[test_idx].copy()
    scored["reconstruction_error"] = errors
    scored["largest_error_feature"] = [features[i] for i in best]
    scored["source_row"] = test_idx
    scored = scored.sort_index()
    loss = pd.DataFrame({
        "Training loss": history.history["loss"],
        "Validation loss": history.history["val_loss"]
    })
    details = {
        "valid": len(clean), "train": len(train_idx), "val": len(val_idx),
        "test": len(test_idx), "excluded_anomalies": excluded_anomalies,
        "labelled": labels is not None
    }
    return scored, reference, loss, details

with st.sidebar:
    st.header("⚙️ Setup")
    source=st.radio("Data source",["Built-in labelled sample","Upload your CSV"])
    st.divider()
    st.markdown("**Deep learning model:** Dense autoencoder (TensorFlow/Keras)")
    st.caption("The encoder compresses readings; the decoder reconstructs them. Large reconstruction errors suggest unusual patterns.")
    st.info("For labelled CSVs, use 0 = normal and 1 = anomaly. Labels are never used as input features.")
    st.caption("Evaluation uses an untouched 20% test split. Random split is intended for independent records, not time-dependent forecasting.")

if source=="Built-in labelled sample":
    df=example_dataset()
    sig="labelled-sample-holdout-v3"
    st.download_button("⬇️ Download labelled sample CSV",df.to_csv(index=False).encode(),"anomalylab_sample_labelled.csv",mime="text/csv")
else:
    file=st.file_uploader("Upload a CSV dataset",type="csv")
    if file is None:
        st.info("Upload a CSV, or choose the built-in labelled sample.")
        st.stop()
    if file.size>25*1024*1024:
        st.error("This demonstration supports CSVs up to 25 MB.")
        st.stop()
    blob=file.getvalue()
    sig=hashlib.sha256(blob).hexdigest()
    try:
        df=pd.read_csv(io.BytesIO(blob))
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        st.stop()
if df.empty:
    st.error("Your CSV has no records.")
    st.stop()
if len(df)>10000:
    st.error("For cloud reliability, please use 10,000 records or fewer.")
    st.stop()

# Visible workflow diagram, included in the actual Streamlit application.
st.markdown("""
<div style="background:#17243b;border:1px solid #43526e;border-radius:18px;padding:24px;margin:14px 0 24px;">
  <div style="font-size:1.25rem;font-weight:750;color:#f8fafc;margin-bottom:8px;">🧠 Training &amp; testing workflow</div>
  <p style="color:#cbd5e1;margin-top:0;">How the autoencoder learns normal patterns and is evaluated fairly</p>
  <div style="background:#334155;border-radius:12px;text-align:center;padding:16px;font-size:1.15rem;font-weight:700;color:white;">Labelled dataset — 100%</div>
  <div style="display:flex;gap:8px;margin-top:18px;">
    <div style="flex:3;background:#2563eb;border-radius:10px;padding:14px 4px;text-align:center;font-weight:700;color:white;">60% Train</div>
    <div style="flex:1;background:#0d9488;border-radius:10px;padding:14px 4px;text-align:center;font-weight:700;color:white;">20% Val</div>
    <div style="flex:1;background:#9333ea;border-radius:10px;padding:14px 4px;text-align:center;font-weight:700;color:white;">20% Test</div>
  </div>
  <div style="display:flex;gap:8px;margin-top:9px;font-size:0.84rem;text-align:center;color:#cbd5e1;">
    <div style="flex:3;">Train on normal rows</div>
    <div style="flex:1;">Set threshold</div>
    <div style="flex:1;">Evaluate unseen rows</div>
  </div>
  <p style="font-size:0.78rem;color:#aebdd0;margin-bottom:0;margin-top:13px;">The percentages are approximate for labelled datasets: known anomalies outside the test split are excluded from training and validation. The final test rows are never used to fit the model or calibrate the threshold.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("📋 Dataset overview")
a,b=st.columns(2)
a.metric("Rows",f"{len(df):,}")
b.metric("Columns",len(df.columns))
with st.expander("Preview first 10 rows",expanded=False):
    st.dataframe(df.head(10),use_container_width=True,hide_index=True)

numeric=df.select_dtypes(include="number").columns.tolist()
if not numeric:
    st.error("The dataset must have numeric measurement columns.")
    st.stop()
label_options=["No ground-truth labels"]+[c for c in numeric if pd.to_numeric(df[c],errors="coerce").dropna().isin([0,1]).all()]
default_label="actual_anomaly" if "actual_anomaly" in label_options else "No ground-truth labels"
label_choice=st.selectbox("Optional ground-truth anomaly label (0 = normal, 1 = anomaly)",label_options,index=label_options.index(default_label))
label_col=None if label_choice=="No ground-truth labels" else label_choice
features_options=[c for c in numeric if c!=label_col and c not in ("actual_anomaly", "is_anomaly")]
if not features_options:
    st.error("No numeric features remain after selecting the label column.")
    st.stop()
features=st.multiselect("Numeric model input features",features_options,default=features_options[:8])
left,right=st.columns(2)
with left:
    percentile=st.slider("Detection threshold percentile",85,99,97,help="Based on reconstruction errors for held-out normal reference rows; higher usually flags fewer rows.")
with right:
    epochs=st.slider("Maximum training epochs",10,80,40,step=5)
st.caption("The threshold is calibrated on held-out normal validation data only. Choose it before inspecting test metrics; repeatedly tuning it against the test set would invalidate the holdout evaluation.")

key=(sig,tuple(features),label_col,epochs)
if st.button("🧠 Train autoencoder",type="primary",disabled=not features):
    try:
        with st.spinner("Training network and calculating anomaly scores..."):
            fit=fit_and_score(df,features,label_col,epochs)
            st.session_state["anomalylab_fit"]={"key":key,"fit":fit}
    except Exception as exc:
        st.error(f"Training failed: {exc}")

saved=st.session_state.get("anomalylab_fit")
if not saved or saved["key"]!=key:
    st.info("Click **Train autoencoder** to generate results.")
    st.stop()

scored,reference,loss,details=saved["fit"]
threshold=float(np.percentile(reference,percentile))
scored=scored.copy()
scored["is_anomaly"]=scored["reconstruction_error"].gt(threshold)
flagged=scored.loc[scored["is_anomaly"]].sort_values("reconstruction_error",ascending=False)
flagged_count=len(flagged)

st.subheader("📊 Unseen test dashboard")
a,b,c,d=st.columns(4)
a.metric("Rows analysed",f"{details['test']:,}")
b.metric("Alerts",f"{flagged_count:,}")
c.metric("Alert rate",f"{100*flagged_count/max(1,details["test"]):.1f}%")
d.metric("Threshold (MSE)",f"{threshold:.4f}")
st.caption(f"Train: {details['train']:,} normal rows • Validation: {details['val']:,} normal rows • Unseen test: {details['test']:,} rows. Model and threshold never use the test data.")
if details["excluded_anomalies"]:
    st.caption(f"Excluded {details['excluded_anomalies']:,} known anomalous rows from the training/validation portion. Test anomalies remain untouched for evaluation.")

error_chart=scored[["reconstruction_error","is_anomaly"]].dropna().copy()
error_chart["Threshold"]=threshold
error_chart=error_chart.reset_index(drop=True)
ch1,ch2=st.columns([3,2])
with ch1:
    st.markdown("#### Reconstruction error over rows")
    st.line_chart(error_chart[["reconstruction_error","Threshold"]],height=310)
with ch2:
    st.markdown("#### Training history")
    st.line_chart(loss,height=310)

st.markdown("#### Alert distribution")
plot_df=error_chart.reset_index().rename(columns={"index":"row_number"})
plot_df["Detection"] = np.where(plot_df["is_anomaly"],"Flagged","Normal")
st.scatter_chart(plot_df,x="row_number",y="reconstruction_error",color="Detection",height=310)

if label_col:
    st.subheader("🎯 Evaluation on unseen test rows")
    available=scored["reconstruction_error"].notna()
    y_true=scored.loc[available,label_col].astype(int)
    y_pred=scored.loc[available,"is_anomaly"].astype(int)
    precision=precision_score(y_true,y_pred,zero_division=0)
    recall=recall_score(y_true,y_pred,zero_division=0)
    f1=f1_score(y_true,y_pred,zero_division=0)
    accuracy=(y_true.to_numpy()==y_pred.to_numpy()).mean()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Precision",f"{precision:.1%}")
    c2.metric("Recall",f"{recall:.1%}")
    c3.metric("F1 score",f"{f1:.1%}")
    c4.metric("Accuracy",f"{accuracy:.1%}")
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    matrix=pd.DataFrame({"Predicted normal":[int(tn),int(fn)],
                         "Predicted anomaly":[int(fp),int(tp)]},index=["Actually normal","Actually anomalous"])
    st.write("**Confusion matrix**")
    st.dataframe(matrix,use_container_width=True)
    st.caption("Scores use only the untouched 20% test partition. The random split assumes independent samples; for time series or repeated machines, use a chronological or group-wise split instead. Results on synthetic data do not establish real-world performance.")
else:
    st.info("No ground-truth labels selected: precision, recall and F1 cannot be calculated. Alerts are candidate anomalies, not confirmed faults.")

st.subheader("🚨 Flagged measurements")
st.caption("'largest_error_feature' is the input variable with the largest standardized reconstruction discrepancy for the row. It suggests what to inspect, but is not a causal explanation.")
if flagged.empty:
    st.success("No rows exceeded the selected threshold.")
else:
    st.dataframe(flagged.head(250),hide_index=True,use_container_width=True)
    top_features=flagged["largest_error_feature"].value_counts().rename_axis("Feature").reset_index(name="Flagged rows")
    st.markdown("#### Most frequent features contributing to alerts")
    st.bar_chart(top_features.set_index("Feature"))

x,y=st.columns(2)
with x:
    st.download_button("⬇️ Download scored test rows",scored.to_csv(index=False).encode("utf-8"),"anomalylab_test_scored.csv",mime="text/csv",use_container_width=True)
with y:
    st.download_button("⬇️ Download flagged rows",flagged.to_csv(index=False).encode("utf-8"),"anomalylab_flagged.csv",mime="text/csv",use_container_width=True)
st.divider()
st.caption("AnomalyLab Pro • Unsupervised anomaly scoring with an autoencoder • Review alerts before taking action")
