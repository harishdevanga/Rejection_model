import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Set up the page layout and title
st.set_page_config(layout="wide")
st.title("🔍 Rejection Prediction Model")

# File uploader widget
uploaded_file = st.file_uploader("Upload the OBD_trial.xlsx file", type=["xlsx"])

# Process the uploaded file
if uploaded_file is not None:
    # Read the uploaded Excel file
    try:
        # Load the dataset
        df = pd.read_excel(uploaded_file, sheet_name='HW Returns Analysis')

        # Display the uploaded data
        st.write("### Uploaded Data")
        st.dataframe(df)
        st.write("-----------------------------------")

        # Preprocessing
        st.write("### Data Preprocessing")
        df['Shipment Date'] = pd.to_datetime(df['Shipment Date'], errors='coerce')
        df['Failure/PR Date'] = pd.to_datetime(df['Failure/PR Date'], errors='coerce')
        df['Aging from(Months)'] = pd.to_numeric(df['Aging from(Months)'], errors='coerce')
        df['Final Status'] = df['Final Status'].apply(lambda x: 1 if x == 'Closed' else 0)

        # Inform about preprocessing status
        st.write("Data preprocessing complete.")

        # Frequency Rejection Analysis
        st.write("### Rejection Frequency Analysis")
        freq_analysis = df.groupby(['Aging from(Months)', 'TPN']).size().reset_index(name='Frequency')

        # Plotting with Plotly Express
        st.write("#### Rejection Frequency by Aging and Product")
        fig = px.bar(
            freq_analysis,
            x='Aging from(Months)',
            y='Frequency',
            color='TPN',
            title='Rejection Frequency by Aging (Months) and Product',
            labels={'Aging from(Months)': 'Aging (Months)', 'Frequency': 'Rejection Frequency'},
            barmode='group'
        )
        st.plotly_chart(fig)
        st.write("-----------------------------------")

        # Prediction Model
        st.write("### Prediction Models")
        features = ['Aging from(Months)', 'TPN', 'Product Family', 'Location']
        X = pd.get_dummies(df[features], drop_first=True)  # One-hot encode categorical features
        y = df['Final Status']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        random_col1, random_col2, random_col3 = st.columns(3)
        # 1. Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
        with random_col1:
            st.write("#### Random Forest Classifier")
            st.write("##### Classification Report - Random Forest")
            st.dataframe(pd.DataFrame(rf_report).transpose())
        with random_col3:
            # Predict future rejections for Random Forest
            aging_input = st.number_input("Select Aging (Months)", min_value=0.0, max_value=30.0, step=0.1, key='rf_input')
            sample_data_rf = X_test.copy()
            sample_data_rf['Aging from(Months)'] = aging_input
            prediction_rf = rf_model.predict(sample_data_rf)
            st.write(f"Predicted Rejections for Aging {aging_input} Months (Random Forest): {sum(prediction_rf)}")
                # Feature importance for Random Forest
        
        rf_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
        rf_importances.sort_values(by='Importance', ascending=False, inplace=True)

        rf_fig = px.bar(
            rf_importances,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Random Forest Feature Importances",
            labels={'Importance': 'Feature Importance'}
        )
        with random_col2:
            st.plotly_chart(rf_fig)
        
        st.write("-----------------------------------")
        
        # 2. Gradient Boosting - XGBoost
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        
        xgb_col1, xgb_col2, xgb_col3 = st.columns(3)

        xgb_report = classification_report(y_test, y_pred_xgb, output_dict=True)
        with xgb_col1:
            st.write("#### Gradient Boosting - XGBoost")
            st.write("##### Classification Report - XGBoost")
            st.dataframe(pd.DataFrame(xgb_report).transpose())

        with xgb_col3:
            # Predict future rejections for XGBoost
            aging_input_xgb = st.number_input("Select Aging (Months)", min_value=0.0, max_value=30.0, step=0.1, key='xgb_input')
            sample_data_xgb = X_test.copy()
            sample_data_xgb['Aging from(Months)'] = aging_input_xgb
            prediction_xgb = xgb_model.predict(sample_data_xgb)
            st.write(f"Predicted Rejections for Aging {aging_input_xgb} Months (XGBoost): {sum(prediction_xgb)}")

        # Feature importance for XGBoost
        xgb_importances = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_model.feature_importances_})
        xgb_importances.sort_values(by='Importance', ascending=False, inplace=True)

        xgb_fig = px.bar(
            xgb_importances,
            x='Importance',
            y='Feature',
            orientation='h',
            title="XGBoost Feature Importances",
            labels={'Importance': 'Feature Importance'}
        )
        with xgb_col2:
            st.plotly_chart(xgb_fig)

        st.write("-----------------------------------")

        # 3. Gradient Boosting - LightGBM
        lgbm_model = LGBMClassifier(random_state=42)
        lgbm_model.fit(X_train, y_train)
        y_pred_lgbm = lgbm_model.predict(X_test)

        lgbm_col1, lgbm_col2, lgbm_col3 = st.columns(3)

        lgbm_report = classification_report(y_test, y_pred_lgbm, output_dict=True)
        with lgbm_col1:
            st.write("#### Gradient Boosting - LightGBM")
            st.write("##### Classification Report - LightGBM")
            st.dataframe(pd.DataFrame(lgbm_report).transpose())

        with lgbm_col3:
            # Predict future rejections for LightGBM
            aging_input_lgbm = st.number_input("Select Aging (Months)", min_value=0.0, max_value=30.0, step=0.1, key='lgbm_input')
            sample_data_lgbm = X_test.copy()
            sample_data_lgbm['Aging from(Months)'] = aging_input_lgbm
            prediction_lgbm = lgbm_model.predict(sample_data_lgbm)
            st.write(f"Predicted Rejections for Aging {aging_input_lgbm} Months (LightGBM): {sum(prediction_lgbm)}")

        # Feature importance for LightGBM
        lgbm_importances = pd.DataFrame({'Feature': X.columns, 'Importance': lgbm_model.feature_importances_})
        lgbm_importances.sort_values(by='Importance', ascending=False, inplace=True)

        lgbm_fig = px.bar(
            lgbm_importances,
            x='Importance',
            y='Feature',
            orientation='h',
            title="LightGBM Feature Importances",
            labels={'Importance': 'Feature Importance'}
        )
        with lgbm_col2:
            st.plotly_chart(lgbm_fig)

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.warning("Please upload a valid Excel file.")
st.write("-----------------------------------")

# revision history 23-11-2024
# code reference rp_trial4.py
