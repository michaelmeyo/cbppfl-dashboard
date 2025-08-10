import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io
import plotly.express as px



st.set_page_config(page_title="Privacy-Preserving Telemedicine Dashboard", layout="wide")

phase = st.sidebar.radio("Select Simulation Phase:", [
    "Upload Dataset",
    "Simulated Attack Preview",
    "Centralized Model",
    "Federated Learning",
    "Federated + Differential Privacy",
    "Compare Models",
    "Compliance & Legal Summary"
    
])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    return df

if phase == "Upload Dataset":
    st.title("Upload Hospital EHR Datasets")
    uploaded_files = st.file_uploader("Upload multiple hospital CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        dataframes = []
        for file in uploaded_files:
            df = load_data(file)
            df['hospital'] = file.name.replace(".csv", "")
            dataframes.append(df)
        full_data = pd.concat(dataframes, ignore_index=True)
        st.session_state["full_data"] = full_data
        st.write("Combined Preview of All Hospital Datasets:")
        st.dataframe(full_data.head())

elif phase == "Simulated Attack Preview":
    st.title("Simulated Model Inversion Attack")

    if "centralized_scores" in st.session_state and "full_data" in st.session_state:
        from sklearn.linear_model import LogisticRegression

        df = st.session_state["full_data"]
        X = df[['gender', 'age']]
        y_sensitive = df['gender']
        y_label = df['has_diabetes']

        X_train, X_test, y_train, y_test, y_sensitive_train, y_sensitive_test = train_test_split(
            X, y_label, y_sensitive, test_size=0.2, random_state=42
        )

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        reconstructed = (proba > 0.5).astype(int)
        attack_results = pd.DataFrame({
            "Model Output (Confidence)": proba,
            "Reconstructed Gender": reconstructed,
            "Actual Gender": y_sensitive_test.values
        })

        accuracy = (attack_results['Reconstructed Gender'] == attack_results['Actual Gender']).mean()
        st.session_state["attack_centralized"] = accuracy

        st.write("Attack results using Logistic Regression on centralized model")
        st.dataframe(attack_results.head(20))
        st.metric("Attack Reconstruction Accuracy", f"{accuracy*100:.2f}%")

        if accuracy > 0.7:
            st.warning("High reconstruction accuracy suggests privacy leakage.")
        else:
            st.success("Low attack accuracy implies model is better protected (likely with DP).")

    # Simulated comparison bar chart (if available)
    st.subheader("Attack Success Comparison Across Models")
    attack_data = {
        "Model": [],
        "Reconstruction Accuracy": []
    }
    if "attack_centralized" in st.session_state:
        attack_data["Model"].append("Centralized")
        attack_data["Reconstruction Accuracy"].append(st.session_state["attack_centralized"] * 100)
    if "attack_fl" in st.session_state:
        attack_data["Model"].append("Federated")
        attack_data["Reconstruction Accuracy"].append(st.session_state["attack_fl"] * 100)
    if "attack_dp" in st.session_state:
        attack_data["Model"].append("Federated + DP")
        attack_data["Reconstruction Accuracy"].append(st.session_state["attack_dp"] * 100)

    if attack_data["Model"]:
        fig = px.bar(
            pd.DataFrame(attack_data),
            x="Model", y="Reconstruction Accuracy",
            color="Model",
            title="Inversion Attack Success Rate by Model Type",
            range_y=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Simulate attack on at least one model to view comparison chart.")
elif phase == "Centralized Model":
    st.title("Centralized Model Simulation")
    if "full_data" in st.session_state:
        df = st.session_state["full_data"]
        X = df[['gender', 'age']]
        y = df['has_diabetes']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.session_state["centralized_scores"] = classification_report(y_test, y_pred, output_dict=True)
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.text("Confusion Matrix:")
        st.text(confusion_matrix(y_test, y_pred))
    else:
        st.warning("Please upload datasets in the first tab.")

elif phase == "Federated Learning":
    st.title("Federated Learning Simulation")
    if "full_data" in st.session_state:
        df = st.session_state["full_data"]
        hospitals = df['hospital'].unique()
        st.write(f"Simulating {len(hospitals)} hospital nodes...")
        preds = []
        for h in hospitals:
            node_df = df[df['hospital'] == h]
            if len(node_df['has_diabetes'].unique()) < 2:
                continue
            X = node_df[['gender', 'age']]
            y = node_df['has_diabetes']
            # One-hot encode gender if needed
            X = pd.get_dummies(X, drop_first=True)
            # Apply SMOTE only if both classes exist
            if len(np.unique(y)) > 1:
                sm = SMOTE(random_state=42)
                try:
                     X_res, y_res = sm.fit_resample(X, y)
                except ValueError as e:
                    st.warning(f"Skipping {h}: SMOTE failed due to data shape. {e}")
                    continue
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_res, y_res)
                prob = model.predict_proba(pd.get_dummies(df[['gender', 'age']], drop_first=True))[:, 1]
                preds.append(prob)
            if len(X) < 10 or len(np.unique(y)) < 2:
                st.warning(f"Skipping {h}: Insufficient data or only one class present.")
                continue
                sm = SMOTE(random_state=42, k_neighbors=1)

            prob = model.predict_proba(df[['gender', 'age']])[:, 1]
            preds.append(prob)
        if preds:
            avg_probs = np.mean(preds, axis=0)
            y_pred = (avg_probs > 0.5).astype(int)
            st.session_state["fl_scores"] = classification_report(df['has_diabetes'], y_pred, output_dict=True)
            st.text("Federated Learning Results:")
            st.text(classification_report(df['has_diabetes'], y_pred))
            st.text("Confusion Matrix:")
            st.text(confusion_matrix(df['has_diabetes'], y_pred))
        if st.button("Simulate Attack on Federated Model"):
            from sklearn.linear_model import LogisticRegression
            X_attack = df[['gender', 'age']]
            y_sensitive = df['gender']
            y_label = df['has_diabetes']
            X_train, X_test, y_train, y_test, y_sensitive_train, y_sensitive_test = train_test_split(
                X_attack, y_label, y_sensitive, test_size=0.2, random_state=42 )
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            reconstructed = (proba > 0.5).astype(int)
            attack_accuracy = (reconstructed == y_sensitive_test).mean()
            st.session_state["attack_fl"] = attack_accuracy
            st.success(f"Federated model inversion attack accuracy: {attack_accuracy*100:.2f}%")

    else:
            st.warning("Please upload datasets in the first tab.")
elif phase == "Federated + Differential Privacy":
    st.title("Federated Learning + Differential Privacy")
    epsilon = st.slider("Select Privacy Budget ε (lower = more noise)", 0.1, 1.0, 0.3, step=0.1)

    def add_dp_noise(probs, epsilon):
        noise = np.random.normal(0, 1/epsilon, size=probs.shape)
        return np.clip(probs + noise, 0, 1), noise

    if "full_data" in st.session_state:
        df = st.session_state["full_data"]
        hospitals = df['hospital'].unique()
        preds = []
        noise_visual = []
        for h in hospitals:
            node_df = df[df['hospital'] == h]
            if len(node_df['has_diabetes'].unique()) < 2:
                continue
            X = node_df[['gender', 'age']]
            y = node_df['has_diabetes']
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
            model.fit(X, y)
            prob = model.predict_proba(df[['gender', 'age']])[:, 1]
            dp_prob, noise = add_dp_noise(prob, epsilon)
            preds.append(dp_prob)
            if len(noise_visual) < 1:
                noise_visual.append(pd.DataFrame({
                    'Original Probability': prob[:20],
                    'Noise': noise[:20],
                    'Noisy Probability': dp_prob[:20]
                }))
        if preds:
            avg_probs = np.mean(preds, axis=0)
            y_pred = (avg_probs > 0.5).astype(int)
            st.session_state["dp_scores"] = classification_report(df['has_diabetes'], y_pred, output_dict=True)
            st.session_state["epsilon"] = epsilon
            st.text("Federated + DP Results:")
            st.text(classification_report(df['has_diabetes'], y_pred))
            st.text("Confusion Matrix:")
            st.text(confusion_matrix(df['has_diabetes'], y_pred))

            if noise_visual:
                st.subheader("Sample of Differential Privacy Noise Injection")
                st.dataframe(noise_visual[0], use_container_width=True)

        if st.button("Simulate Attack on FL + DP Model"):
            from sklearn.linear_model import LogisticRegression
            X_attack = df[['gender', 'age']]
            y_sensitive = df['gender']
            y_label = df['has_diabetes']

            X_train, X_test, y_train, y_test, y_sensitive_train, y_sensitive_test = train_test_split(
                X_attack, y_label, y_sensitive, test_size=0.2, random_state=42 )

            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            noisy_proba, _ = add_dp_noise(proba, epsilon)

            reconstructed = (noisy_proba > 0.5).astype(int)
            attack_accuracy = (reconstructed == y_sensitive_test).mean()

            st.session_state["attack_dp"] = attack_accuracy
            st.success(f"FL + DP model inversion attack accuracy: {attack_accuracy * 100:.2f}%")

            attack_results = pd.DataFrame({
                "Noisy Confidence": noisy_proba,
                "Reconstructed Gender": reconstructed,
                "Actual Gender": y_sensitive_test.values
            })
            st.dataframe(attack_results.head(20))

    else:
        st.warning("Please upload datasets in the first tab.")

elif phase == "Compare Models":
    st.title("Model Comparison View")
    if "centralized_scores" in st.session_state and "fl_scores" in st.session_state and "dp_scores" in st.session_state:
        central_f1 = st.session_state["centralized_scores"]["1"]["f1-score"]
        fl_f1 = st.session_state["fl_scores"]["1"]["f1-score"]
        dp_f1 = st.session_state["dp_scores"]["1"]["f1-score"]

        df_compare = pd.DataFrame({
            "Model": ["Centralized", "Federated", "Federated + DP"],
            "F1-score (Diabetic class)": [central_f1, fl_f1, dp_f1]
        })
        st.bar_chart(df_compare.set_index("Model"))
    else:
        st.warning("Run all simulations before comparing.")

elif phase == "Compliance & Legal Summary":
    st.title("Compliance & Governance Dashboard")

    centralized_score = st.session_state.get("centralized_scores", {}).get("1", {}).get("f1-score", 0)
    fl_score = st.session_state.get("fl_scores", {}).get("1", {}).get("f1-score", 0)
    dp_score = st.session_state.get("dp_scores", {}).get("1", {}).get("f1-score", 0)
    epsilon = st.session_state.get("epsilon", 1.0)

    def legal_score(base, f1):
        return round(min(1.0, base + f1 * 0.5), 2)

    law_scores = pd.DataFrame({
        "Law": ["HIPAA", "GDPR", "DPA 2019"],
        "Centralized": [legal_score(0.2, centralized_score), legal_score(0.3, centralized_score), legal_score(0.4, centralized_score)],
        "Federated": [legal_score(0.6, fl_score), legal_score(0.6, fl_score), legal_score(0.6, fl_score)],
        "Federated + DP": [legal_score(0.7, dp_score) + (1.0 - epsilon)/10]*3
    }).set_index("Law")

    st.subheader("Legal Compliance Scores Table")
    st.dataframe(law_scores, use_container_width=True)

    st.markdown("**Interpretation:**")
    st.markdown("- Scores closer to 1.0 indicate stronger alignment with respective data protection frameworks.")
    st.markdown("- Centralized model typically shows poor compliance unless accuracy is exceptionally high.")
    st.markdown("- Federated learning improves compliance by retaining data at source.")
    st.markdown("- Federated + DP scores incorporate both accuracy and the epsilon privacy budget (lower ε means stronger privacy).")
    st.markdown("These scores are dynamic, updated based on simulation results above.")
