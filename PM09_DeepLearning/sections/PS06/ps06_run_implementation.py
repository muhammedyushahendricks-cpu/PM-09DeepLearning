# PM-09-PS06: Run AI Implementation 
import numpy as np 
import pandas as pd 
import joblib 
import tensorflow as tf  
import time 
 
print("=" * 60) 
print("PM-09-PS06: RUN AI IMPLEMENTATION") 
print("=" * 60) 
 
# 1. LOAD THE AI SOLUTION 
model = tf.keras.models.load_model("models/hr_model.keras") 
scaler = joblib.load("models/scaler.pkl") 
label_encoder = joblib.load("models/label_encoder.pkl") 
feature_cols = joblib.load("models/feature_cols.pkl") 
 
# 2. BATCH PREDICTION FUNCTION 
def batch_predict(df_input): 
    df = df_input.copy() 
    df = pd.get_dummies(df, columns=['Department'], prefix='Dept') 
    for col in feature_cols: 
        if col not in df.columns: 
            df[col] = 0 
    df = df[feature_cols] 
 
    X = scaler.transform(df.values) 
    probs = model.predict(X, verbose=0) 
    preds = np.argmax(probs, axis=1) 
 
    labels = label_encoder.classes_[preds] 
    confidences = probs[np.arange(len(preds)), preds] 
    return labels, confidences, probs 
 
# 3. RUN ON THE FULL DATASET 
print("\n--- RUNNING AI SOLUTION ON FULL DATASET ---") 
 
df_full = pd.read_csv("hr_data.csv") 
df_input = df_full.drop(columns=['EmployeeID', 'Salary', 'SalaryBand']) 
 
start = time.time() 
labels, confidences, probs = batch_predict(df_input) 
elapsed = time.time() - start 
 
print(f"Predicted {len(labels)} employees in {elapsed:.3f} seconds") 
print(f"Throughput: {len(labels)/elapsed:.0f} predictions/second") 
 
# 4. BUILD THE OUTPUT REPORT 
results = df_full[['EmployeeID', 'Department', 'YearsOfService', 
                   'IsManager', 'PerformanceScore', 'Age']].copy() 
results['PredictedBand'] = labels 
results['ActualBand'] = df_full['SalaryBand'] 
results['Confidence'] = confidences.round(3) 
results['Correct'] = (results['PredictedBand'] == results['ActualBand']) 
 
print("\n--- SAMPLE PREDICTIONS ---") 
print(results.head(10).to_string(index=False)) 
 
# 5. MONITORING DASHBOARD 
print("\n--- MONITORING DASHBOARD ---") 
 
overall_acc = results['Correct'].mean() 
print(f"Overall accuracy on full dataset: {overall_acc:.4f}") 
 
print("\nAccuracy by department:") 
print(results.groupby('Department')['Correct'].mean().round(3)) 
 
print("\nAccuracy by actual band:") 
print(results.groupby('ActualBand')['Correct'].mean().round(3)) 
 
print("\nConfidence distribution:") 
print(f"  Mean confidence:   {results['Confidence'].mean():.3f}") 
print(f"  Min confidence:    {results['Confidence'].min():.3f}") 
print(f"  Low-confidence (< 0.6) predictions: " 
      f"{(results['Confidence'] < 0.6).sum()}") 
 
# 6. FLAG LOW-CONFIDENCE CASES 
print("\n--- LOW-CONFIDENCE CASES (need human review) ---") 
low_conf = results[results['Confidence'] < 0.6].sort_values('Confidence') 
if len(low_conf) > 0: 
    print(low_conf.head(10).to_string(index=False)) 
else: 
    print("No low-confidence cases found.") 
 
# 7. SAVE THE REPORT 
results.to_csv("evidence/ps06_predictions.csv", index=False) 
print("\nFull predictions saved to evidence/ps06_predictions.csv") 
 
print("\n" + "=" * 60) 
print("AI implementation ran successfully.") 
print("Ready to productionise the solution (PS07).") 
print("=" * 60) 