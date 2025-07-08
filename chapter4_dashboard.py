import streamlit as st

# ================== Streamlit Page Config ==================
st.set_page_config(
    page_title="Chapter Four Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import numpy as np

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    df = pd.read_csv("df_all.csv")
    return df

df_all = load_data()
X_test = df_all[['Gender', 'Age', 'Hospital']]
y_test = df_all['has_hypertension']


st.title("Chapter Four Results Dashboard")

with st.container():
    st.subheader("[Overview] Interactive Thesis Simulation")
    st.markdown("""
    Welcome to your live simulation dashboard for Chapter Four.  
    Use the sidebar to run real-time simulations for:
    - Centralized Learning  
    - Federated Learning  
    - Federated Learning + Differential Privacy
    """)

col1, col2 = st.columns(2)

with col1:
    st.info("Choose a model from the left and click ▶ Run Simulation.")
with col2:
    st.success("Results and performance metrics will appear below automatically.")

# ================== PHASE 1: Sidebar Simulation UI ==================
st.sidebar.title("[Settings] Simulate Chapter 4 Models")
selected_model = st.sidebar.selectbox(
    "Choose ML Model to Simulate",
    ["Centralized", "Federated", "Federated + DP"]
)
run_simulation = st.sidebar.button("[Run] Run Model Simulation")

# ================== PHASE 2: Simulation Execution ==================
if run_simulation:
    st.markdown("## [Analysis] Model Simulation Results")
    st.info(f"[Loading] Running {selected_model} simulation...")

    try:
        if selected_model == "Centralized":
            model = joblib.load("central_model.pkl")
            y_pred = model.predict(X_test)

        elif selected_model == "Federated":
            model = joblib.load("global_model.pkl")
            y_pred = model.predict(X_test)

        elif selected_model == "Federated + DP":
            model = joblib.load("dp_global_model.pkl")
            y_pred = model.predict(X_test)

        if isinstance(y_pred[0], (np.float32, np.float64)) or y_pred.max() <= 1:
            y_pred = (y_pred > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"[Success] Accuracy of {selected_model} Model: `{acc:.2f}`")

        st.markdown("**[Stats] Classification Report**")
        st.json(classification_report(y_test, y_pred, output_dict=True))

        st.markdown(f"#### [Confusion Matrix] Confusion Matrix – {selected_model} Model")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"[Error] Error running {selected_model} model: {e}")



# 🔁 Try to load pre-trained federated global model
try:
    global_model = joblib.load("global_model.pkl")
except:
    global_model = None


@st.cache_data
def load_data():
    df = pd.read_csv("df_all.csv")
    return df

df_all = load_data()

# Sidebar Navigation
section = st.sidebar.radio("Go to section:", [
    "[Pin] Overview",
    "[Trend] Descriptive Statistics",
    "[Model] Centralized Model Results",
    "[Federated] Federated Learning Simulation",
    "[Privacy] Federated Learning + Differential Privacy",
    "[Stats] Model Comparison Summary",
    "[Ethics] Ethical & Governance Notes",
])
# Page Title
st.title("Chapter Four Results Dashboard")

# Routing Logic
if section == "[Pin] Overview":
    st.subheader("Cross-Border Federated Learning Dashboard – Implementation Summary")

    st.markdown("""
    This dashboard presents implementation results for the thesis:  
    **"Enhancing Data Governance in Kenyan Cross-Border Telemedicine Using Machine Learning with Federated Learning."**

    The framework developed is referred to as **CB-PPFL** – *Cross-Border Privacy-Preserving Federated Learning*.  
    It simulates a privacy-compliant telemedicine collaboration between 7 hospitals in **Kenya, Uganda, and Tanzania**,  
    using synthetic EHR data to preserve jurisdictional data sovereignty.
    """)

    st.markdown("### [Settings] Key Architecture Components")
    st.markdown("""
    - [Hospital] **Hospital Nodes**: Each hospital locally trains a model without sharing raw data.
    - [Federated] **Federated Server**: Aggregates model weights using FedAvg algorithm.
    - [Privacy] **Differential Privacy Layer**: Gaussian noise is added to ensure (ε, δ)-DP compliance.
    - [Stats] **Compliance Dashboard**: This interface shows training results, privacy metrics, and participation logs.
    """)

    # Summary Metrics
    st.markdown("### [Stats] Summary of Current Implementation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("[Records] Total Records", "14,545", "Synthetic EHRs")
    with col2:
        st.metric("[Model] Models Compared", "3", "Centralized, FL, FL+DP")
    with col3:
        st.metric("[Hospital] Hospital Nodes", "7", "3 Countries")

    # Project Milestone Checklist
    st.markdown("### [Success] Project Milestones")
    st.checkbox("[Pin] Phase 1: Data Preprocessing Completed", value=True)
    st.checkbox("[Pin] Phase 2: Descriptive Statistics Generated", value=True)
    st.checkbox("[Pin] Phase 3: Centralized Models Trained (LogReg + RF)", value=True)
    st.checkbox("[Pin] Phase 4: Federated Learning Simulation (FedAvg)", value=True)
    st.checkbox("[Pin] Phase 5: Differential Privacy Layer Integrated", value=True)
    st.checkbox("[Pin] Phase 6: Model Comparison Summary", value=True)
    st.checkbox("[Ethics] Ethical & Governance Notes", value=False)

    # Legal Compliance Highlight
    with st.expander("[Privacy] Privacy & Policy Compliance"):
        st.markdown("""
        - [Success] Complies with **Kenya Data Protection Act (2019)** – no cross-border data transfer.
        - [Success] Aligned with **GDPR adequacy rules** (Corrales & Fenwick, 2024).
        - [Success] Preserves **data sovereignty** per Ayo-Farai et al. (2023).
        - [Success] Ensures patient anonymity via (ε, δ)-Differential Privacy mechanisms.
        """)

    # Footer
    st.markdown("""
    ---
    _Last updated: **July 7, 2025**_  
    _Author: **Michael Meyo** | Thesis Defense System Simulation Interface_
    """)
elif section == "[Trend] Descriptive Statistics":
    st.subheader("Section 4.2 – Descriptive Statistics")

    st.markdown("### [Hospital] Patient Count by Hospital")

    # Map hospital label-encoded values to actual names
    hospital_map = {
        0: "Busia County",
        1: "KNH",
        2: "MTRH",
        3: "Mulago (UG)",
        4: "Mwanza (TZ)",
        5: "Nairobi Hosp 1",
        6: "Nairobi Hosp 2"
    }
    df_all['Hospital_Name'] = df_all['Hospital'].map(hospital_map)
    hospital_counts_named = df_all['Hospital_Name'].value_counts()
    st.bar_chart(hospital_counts_named)

    st.markdown("### [Gender] Gender Distribution")
    gender_counts = df_all['Gender'].value_counts()

    # Assume 0 = Female, 1 = Male unless reversed in your LabelEncoder
    gender_labels = ['Female', 'Male']
    fig1, ax1 = plt.subplots()
    ax1.pie(gender_counts, labels=gender_labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.markdown("### [Stats] Age Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df_all['Age'].dropna(), bins=20, color='skyblue', edgecolor='black')
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

    st.markdown("### [Inspect] Sample EHR Data")
    st.dataframe(df_all.head(10))

    st.success("[Success] Descriptive statistics reflect the diversity and volume of EHR data across 7 cross-border hospital nodes.")
 
elif section == "[Model] Centralized Model Results":
    st.subheader("Section 4.3 – Traditional Centralized Model")

    st.markdown("### 🧪 Model Training on Full Dataset")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import seaborn as sns

    # Prepare Data
    X = df_all[['Gender', 'Age', 'Hospital']]
    y = df_all['has_hypertension']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Centralized Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{acc:.2%}")

    # Confusion Matrix
    st.markdown("### [Confusion Matrix] Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

    # Classification Report
    st.markdown("### [Report] Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

    # Model Notes
    st.markdown("""
    [Success] This Random Forest model acts as the **baseline centralized model**.  
    It simulates traditional hospital-level ML where all data is merged into one location.

    ⚠️ In real cross-border telemedicine, this approach would violate patient data sovereignty and regulatory boundaries.
    """)

elif section == "[Federated] Federated Learning Simulation":
    st.subheader("Section 4.4 – Federated Learning Simulation")

    st.markdown("### [Node] Local Node Training Summary")
    st.markdown(
        "Each hospital trained a local model on its own data silo without sharing EHRs across borders. "
        "These models were aggregated using the **FedAvg** algorithm to form a **global model**."
    )

    # Show hospital records used
    hospital_counts = df_all['Hospital'].value_counts().reset_index()
    hospital_counts.columns = ['Hospital', 'Records Used']
    st.dataframe(hospital_counts)

    st.markdown("### [Globe] Model Training on Full Dataset")

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        X_test = df_all[['Gender', 'Age', 'Hospital']]
        y_test = df_all['has_hypertension']

        if global_model is not None:
            # Predict with the global model
            y_pred = global_model.predict(X_test)

            # Ensure binary output if predictions are probabilistic
            if y_pred.dtype != int:
                y_pred = (y_pred > 0.5).astype(int)

            y_test = y_test.astype(int)  # Ensure y_test is binary

            acc = accuracy_score(y_test, y_pred)
            st.success(f"[Success] Accuracy of Global Model: **{acc:.2f}**")

            st.markdown("#### [Details] Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.markdown("#### [Error] Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        else:
            st.error("⚠️ global_model not found. Be sure to train it in Jupyter and export it as 'global_model.pkl'")

    except Exception as e:
        st.error(f"[Error] Error during evaluation: {e}")


elif section == "[Privacy] Federated Learning + Differential Privacy":
    st.subheader("Federated Learning with Differential Privacy")

    # Load model
    try:
        dp_model = joblib.load("dp_global_model.pkl")
        with open("dp_metrics.json") as f:
            dp_metrics = json.load(f)

        # Display metrics
        st.success(f"[Success] Accuracy of DP Global Model: **{dp_metrics['accuracy']:.2f}**")

        st.markdown("#### [Report] Classification Report")
        report = dp_metrics['classification_report']
        st.json(report)

        st.markdown("#### [Numbers] Confusion Matrix")
        cm = np.array(dp_metrics['confusion_matrix'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"[Error] Error loading DP model or metrics: {e}")

elif section == "[Stats] Model Comparison Summary":
    st.subheader("Model Comparison Summary")
    st.write("This section compares the evaluation metrics across three approaches:")

    st.markdown("""
    - [Model] Traditional Centralized Model  
    - [Federated] Federated Learning Model  
    - [Security] Federated + Differential Privacy Model
    """)

    # [Success] Load and define the test set
    X_test = df_all[['Gender', 'Age', 'Hospital']]
    y_test = df_all['has_hypertension']

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import json
    import joblib

    # ===============================
    # [Model] CENTRALIZED MODEL EVALUATION
    # ===============================
    try:
        central_model = joblib.load("central_model.pkl")
        st.success("[Success] Centralized model loaded successfully.")

        y_pred_central = central_model.predict(X_test)
        acc_central = accuracy_score(y_test, y_pred_central)
        report_central = classification_report(y_test, y_pred_central, output_dict=True)

        st.markdown(f"**[Model] Accuracy of Central Model: `{acc_central:.2f}`**")
        st.markdown("##### 🗂 Classification Report – Centralized")
        st.json(report_central)

        st.markdown("#### [Model] Confusion Matrix – Centralized Model")
        cm = confusion_matrix(y_test, y_pred_central)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"[Error] Error evaluating centralized model: {e}")

    # ===============================
    # [Federated] FEDERATED MODEL EVALUATION
    # ===============================
    try:
        y_pred_fl = global_model.predict(X_test)

        if isinstance(y_pred_fl[0], (np.float32, np.float64)) or (y_pred_fl.max() <= 1):
            y_pred_fl = (y_pred_fl > 0.5).astype(int)

        acc_fl = accuracy_score(y_test, y_pred_fl)
        report_fl = classification_report(y_test, y_pred_fl, output_dict=True)

        st.success("[Success] Federated model loaded successfully.")
        st.markdown(f"**[Federated] Accuracy of Federated Model: `{acc_fl:.2f}`**")
        st.markdown("##### 🗂 Classification Report – Federated")
        st.json(report_fl)

        st.markdown("#### [Federated] Confusion Matrix – Federated Model")
        cm = confusion_matrix(y_test, y_pred_fl)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"[Error] Error evaluating federated model: {e}")

    # ===============================
    # [Privacy] DIFFERENTIAL PRIVACY MODEL
    # ===============================
    try:
        dp_global_model = joblib.load("dp_global_model.pkl")
        y_pred_dp = dp_global_model.predict(X_test)

        if isinstance(y_pred_dp[0], (np.float32, np.float64)) or (y_pred_dp.max() <= 1):
            y_pred_dp = (y_pred_dp > 0.5).astype(int)

        acc_dp = accuracy_score(y_test, y_pred_dp)
        report_dp = classification_report(y_test, y_pred_dp, output_dict=True)

        st.success("[Success] DP model loaded successfully.")
        st.markdown(f"**[Privacy] Accuracy of DP Model: `{acc_dp:.2f}`**")
        st.markdown("##### 🗂 Classification Report – FL + Differential Privacy")
        st.json(report_dp)

        st.markdown("#### [Privacy] Confusion Matrix – DP Model")
        cm = confusion_matrix(y_test, y_pred_dp)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"[Error] Error evaluating DP model: {e}")
        # ===============================
    # [Stats] ACCURACY + F1-SCORE COMPARISON BAR CHART
    # ===============================
    st.markdown("### [Stats] Model Performance Comparison")

    try:
        model_names = []
        accuracies = []
        f1_scores = []

        # CENTRALIZED
        if 'report_central' in locals() and 'acc_central' in locals():
            model_names.append("Centralized")
            accuracies.append(acc_central)
            f1_scores.append(report_central["macro avg"]["f1-score"])

        # FEDERATED
        if 'report_fl' in locals() and 'acc_fl' in locals():
            model_names.append("Federated")
            accuracies.append(acc_fl)
            f1_scores.append(report_fl["macro avg"]["f1-score"])

        # DP
        if 'report_dp' in locals() and 'acc_dp' in locals():
            model_names.append("Federated + DP")
            accuracies.append(acc_dp)
            f1_scores.append(report_dp["macro avg"]["f1-score"])

        if model_names:
            fig, ax = plt.subplots(figsize=(8, 4))
            width = 0.35
            x = np.arange(len(model_names))

            ax.bar(x - width/2, accuracies, width, label='Accuracy')
            ax.bar(x + width/2, f1_scores, width, label='F1 Score')

            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.set_title("[Stats] Accuracy vs F1 Score by Model")
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            st.pyplot(fig)
        else:
            st.warning("⚠️ No model metrics available to generate comparison chart.")

    except Exception as e:
        st.error(f"[Error] Error generating comparison chart: {e}")

elif section == "[Ethics] Ethical & Governance Notes":
    st.subheader("[Ethics] Ethical and Governance Notes")

    st.markdown("""
    Federated learning in cross-border telemedicine introduces ethical, legal, and data governance complexities that must be addressed beyond algorithmic success. This section highlights key considerations drawn from the simulation and literature.

    ### [Map] Data Sovereignty and Jurisdictional Compliance
    Traditional cloud-based telemedicine systems often violate data sovereignty by transferring patient data across borders. Our federated approach retains data within local hospitals, ensuring compliance with Kenya’s Data Protection Act (2019) and GDPR’s data minimization principles (Corrales & Fenwick, 2024).

    > “Cross-border health data transfers require safeguards such as encryption and legal controls.”  
    — Corrales & Fenwick, 2024

    ### [AI] Ethical Use of AI in Healthcare
    While federated learning enhances privacy, differential privacy introduced fairness trade-offs by lowering performance, especially on minority class predictions. This supports Habehh & Gohel's (2021) view that ethical AI requires balanced transparency and impact analysis.

    > “Privacy risks and ethical considerations must be addressed early in model development.”  
    — Habehh & Gohel, 2021

    ### [Federated] Regulatory Fragmentation in East Africa
    Kenya’s DPA is progressive, but the region lacks harmonized telemedicine laws. Our model design ensures that health data stays within national borders, minimizing the risks of legal ambiguity raised by Ayo-Farai et al. (2023).

    > “Inconsistent cross-border regulations introduce challenges in enforcement and liability.”  
    — Ayo-Farai et al., 2023

    ### [PETs] Privacy-Enhancing Technologies (PETs)
    The project utilized differential privacy and federated averaging — two PETs that preserve anonymity and support real-time compliance. Mensah (2022) affirms that these techniques are essential for scalable, privacy-respecting AI in healthcare.

    > “Federated learning preserves jurisdictional control while enabling collaborative AI.”  
    — Mensah, 2022
    """)

    st.success("[Success] Ethical governance is embedded in the model through privacy-preserving design and regulatory alignment.")



