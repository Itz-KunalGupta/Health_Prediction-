
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the models
with open('/content/random_forest_model.pkl', 'rb') as rf_file:
    random_forest_model = pickle.load(rf_file)

with open('/content/decision_tree_model.pkl', 'rb') as dt_file:
    decision_tree_model = pickle.load(dt_file)

# Define function to get health suggestions based on prediction
def get_health_suggestions(prediction):
    suggestions = {
        "Heart Attack Likely": {
            "message": "Please contact your doctor before it's too late.",
            "exercise": "Engage in cardiovascular exercises like walking, jogging, or cycling for at least 30 minutes a day.",
            "diet": "Follow a heart-healthy diet rich in fruits, vegetables, whole grains, lean proteins, and low in saturated fats.",
            "sleep": "Aim for 7-8 hours of sleep per night to improve heart health.",
            "stress": "Practice stress management techniques like meditation, deep breathing exercises, and yoga."
        },
        "Heart Attack Unlikely": {
            "message": "Keep maintaining a healthy lifestyle to stay heart-healthy.",
            "exercise": "Maintain an active lifestyle with moderate exercise such as walking, swimming, or biking for at least 30 minutes most days.",
            "diet": "Follow a balanced diet with appropriate portion sizes to maintain a healthy weight. Include fruits, vegetables, lean proteins, and whole grains.",
            "sleep": "Ensure adequate rest, aiming for 7-8 hours of sleep per night for optimal health.",
            "stress": "Manage stress through activities like reading, spending time with family, and engaging in hobbies."
        }
    }
    return suggestions[prediction]["message"], suggestions[prediction]["exercise"], suggestions[prediction]["diet"], suggestions[prediction]["sleep"], suggestions[prediction]["stress"]


# Streamlit UI
def app():
    st.set_page_config(page_title="Heart Attack Prediction", layout="wide")

    # Vertical Sidebar for navigation
    page = st.sidebar.selectbox("Navigation", ["Home", "Model Performance", "About"], index=2)  # Set default to "About"

    if page == "Home":
        st.title('Heart Attack Prediction and Health Tips')

        # Input fields with layout
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], index=0)
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, value=120)
        chol = st.number_input("Cholesterol Level (mg/dl)", min_value=0, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
        fbs_map = {"No": 0, "Yes": 1}
        restecg = st.selectbox("Resting ECG Results", options=["Normal", "Having ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        restecg_map = {"Normal": 0, "Having ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
        exang_map = {"No": 0, "Yes": 1}
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, value=1.0)

        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=["No vessels colored", "One vessel colored", "Two vessels colored", "Three vessels colored", "Four vessels colored"])
        ca_map = {"No vessels colored": 0, "One vessel colored": 1, "Two vessels colored": 2, "Three vessels colored": 3, "Four vessels colored": 4}

        thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])
        thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}

        # Convert inputs to numeric values
        sex = 1 if sex == "Male" else 0
        cp = cp_map[cp]
        fbs = fbs_map[fbs]
        restecg = restecg_map[restecg]
        exang = exang_map[exang]
        thal = thal_map[thal]
        ca = ca_map[ca]

        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, ca, thal]])

        if st.button('Predict Heart Attack'):
            dt_prediction = decision_tree_model.predict(features)
            dt_result = "Heart Attack Likely" if dt_prediction[0] == 1 else "Heart Attack Unlikely"

            st.subheader(f"Prediction: {dt_result}")

            message, exercise, diet, sleep, stress = get_health_suggestions(dt_result)
            with st.expander("Health Tips:"):
                st.write(f"Message: {message}")
                st.write(f"Exercise: {exercise}")
                st.write(f"Diet: {diet}")
                st.write(f"Sleep: {sleep}")
                st.write(f"Stress: {stress}")

            # Create a figure for the bar plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot variables (excluding Age and Gender)
            variables = ['Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol Level', 'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Induced Angina', 'ST Depression', 'Major Vessels', 'Thalassemia']
            values = [cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, ca, thal]

            # Define thresholds for classification
            thresholds = {
                'Chest Pain Type': [0, 1], 
                'Resting Blood Pressure': [90, 140], 
                'Cholesterol Level': [150, 200], 
                'Fasting Blood Sugar': [0, 1], 
                'Resting ECG': [0, 1, 2], 
                'Max Heart Rate': [100, 160], 
                'Exercise Induced Angina': [0, 1], 
                'ST Depression': [0, 1], 
                'Major Vessels': [0, 1, 2, 3], 
                'Thalassemia': [0, 1, 2]
            }

            # Function to classify values based on thresholds
            def classify_value(variable, value):
                if variable in thresholds:
                    threshold = thresholds[variable]
                    if value < threshold[0]:
                        return "Low"
                    elif value < threshold[1]:
                        return "Normal"
                    else:
                        return "High"
                return "Normal"

            # Create the bar plot
            sns.barplot(x=variables, y=values, ax=ax)

            # Add labels above each bar with appropriate classification
            for i, value in enumerate(values):
                label = classify_value(variables[i], value)
                ax.text(i, value + 1, label, ha='center', va='bottom', fontsize=10, color='black')

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45, ha="right")
            plt.title("Input Feature Values with Classification")

            # Display the plot
            st.pyplot(fig)

    elif page == "Model Performance":
        st.title("Model Performance Evaluation 📊")
        st.header("Performance of Different Machine Learning Models 🧑‍💻")
        st.write("""
    This section showcases the performance of various machine learning models trained to predict the likelihood of heart disease. The models are evaluated based on metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**. Below, we explain each evaluation parameter and how it relates to model performance. 📈

    ### Evaluating Parameters:
    To assess the quality of a model, we use the following metrics:

    1. **Accuracy**: The proportion of correct predictions made by the model out of all predictions. 
        - A higher accuracy means the model is making more correct predictions overall.
    2. **Precision**: The proportion of true positive predictions (correctly predicted heart attack cases) out of all positive predictions made by the model.
        - Higher precision means the model is good at predicting positive cases and not making too many false positives.
    3. **Recall**: The proportion of true positive predictions (correct heart attack cases) out of all actual positive cases in the dataset.
        - A higher recall means the model is good at identifying heart attack cases and not missing too many.
    4. **F1-Score**: The harmonic mean of precision and recall, offering a balance between the two.
        - A higher F1-Score indicates that both precision and recall are high, meaning the model is both accurate and reliable.

    ### Confusion Matrix for Random Forest and Decision Tree:
    We also evaluate models using the **Confusion Matrix**, which helps understand the breakdown of predictions into:
    - **True Positives (TP)**: Correct predictions of a heart attack.
    - **True Negatives (TN)**: Correct predictions of no heart attack.
    - **False Positives (FP)**: Incorrect predictions of a heart attack when there is none.
    - **False Negatives (FN)**: Incorrect predictions of no heart attack when there is one.
    
    The confusion matrix shows how well the model distinguishes between the two classes (heart attack vs. no heart attack).

    ### Performance Metrics of Random Forest:
    **Accuracy**: 83.70%  
    Random Forest's confusion matrix shows that:
    
    - For **Class 0 (No Heart Attack)**, the model has a precision of 0.79 and a recall of 0.79, indicating a reasonable balance between predicting no heart attack and avoiding false positives.
    - For **Class 1 (Heart Attack)**, the precision and recall are both higher at 0.87, meaning the model is quite effective at identifying actual heart attack cases.
    
    Overall, the Random Forest model performs well with a **macro average** of 0.83 for precision, recall, and F1-score, and a **weighted average** of 0.84, showing good overall performance across both classes.

    ### Performance Metrics of Decision Tree:
    **Accuracy**: 83.70%  
    Decision Tree’s confusion matrix shows:
    
    - For **Class 0 (No Heart Attack)**, the precision is 0.88, but recall drops to 0.67, meaning that the model is good at predicting no heart attack but misses some cases.
    - For **Class 1 (Heart Attack)**, the precision is 0.82 and recall is 0.94, indicating that the model is very effective at identifying true heart attack cases, but may sometimes misclassify a few no-heart-attack cases as heart attack cases.

    The Decision Tree model's **macro average** is 0.81 for recall and 0.82 for F1-score, with a **weighted average** of 0.84, suggesting it performs well but with a slight bias towards predicting heart attacks.

    ### Performance Comparison of All Models:
    Below is the comparison of the performance of all the models used in this study:

    | Model                        | Accuracy (%) | Precision | Recall | F1 Score |
    |------------------------------|--------------|-----------|--------|----------|
    | Logistic Regression           | 82.22        | 0.82      | 0.82   | 0.82     |
    | Naive Bayes                   | 78.52        | 0.79      | 0.79   | 0.79     |
    | **Random Forest**             | **83.70**    | **0.84**  | **0.84** | **0.84** |
    | Extreme Gradient Boost        | 61.48        | 0.38      | 0.61   | 0.47     |
    | K-Nearest Neighbour           | 82.22        | 0.82      | 0.82   | 0.82     |
    | **Decision Tree**             | **83.70**    | **0.84**  | **0.84** | **0.83** |
    | Support Vector Machine        | 83.70        | 0.84      | 0.84   | 0.83     |

    ### Insights:
    - The **Random Forest**, **Decision Tree**, and **Support Vector Machine** models achieved the highest **accuracy (83.70%)**. This shows their strong ability to predict heart attack risk.
    - **Random Forest** stands out with a balanced performance across precision, recall, and F1-Score, making it a reliable model for heart attack prediction.
    - While **Extreme Gradient Boost** has a much lower accuracy (61.48%) and precision (0.38), it struggles with predicting heart attack cases accurately and is not ideal for this task.
    - **Logistic Regression** and **K-Nearest Neighbour** perform similarly to Random Forest, but slightly less accurate and precise.

    ### Conclusion:
    The **Random Forest** and **Decision Tree** models both perform exceptionally well and are our top picks for predicting heart attack risk. They balance accuracy, precision, recall, and F1-Score, providing a reliable prediction system for heart disease risk assessment. 🔍💓

    We encourage users to use these predictions and visualizations to make informed decisions about their health. With further improvements in data collection and model training, we aim to increase the accuracy of these models for better, more personalized health predictions in the future.
    """)

    elif page == "About":
        st.title("About this Application")
        st.header("Predicting Heart Attack Risk Using Machine Learning 🧑‍💻❤️")
        st.write("""
    This application predicts the likelihood of a heart attack based on various health parameters using machine learning algorithms. The goal is to provide a tool that can help individuals understand their health better and take proactive measures to maintain a healthy lifestyle. 🏃‍♂️🍏

    **Algorithms Used:**
    
    We utilize two powerful machine learning algorithms — **Random Forest** and **Decision Tree** — to predict whether a person is at risk of having a heart attack. Here's why we chose these models:
    
    1. **Decision Tree**: 🌳
        - This algorithm is easy to interpret and understand, making it useful for healthcare practitioners to quickly assess risk factors. 
        - It works by splitting the data into smaller and smaller subsets, using criteria (such as cholesterol levels or age) to predict the outcome.
        - We use this model to make initial, fast predictions and explainability.

    2. **Random Forest**: 🌲
        - Random Forest is an ensemble method that uses multiple decision trees to make more accurate predictions.
        - By combining the results of several decision trees, Random Forest helps minimize the errors that a single tree might make, leading to a more robust and reliable prediction.
        - This model performs well even with complex and noisy data, offering higher accuracy compared to the decision tree alone.
        
    The prediction output of both models is then compared, and the final decision is based on the prediction that best matches the data provided.

    **How the Prediction Works**:
    - The application takes in several health-related input parameters such as age, sex, cholesterol level, exercise-induced angina, etc. These inputs are fed into the trained machine learning models.
    - The **Decision Tree** model classifies these inputs based on thresholds to quickly give an initial heart attack risk prediction.
    - The **Random Forest** model aggregates predictions from multiple decision trees, providing a more accurate and stable output.
    - The final result is a prediction of whether the individual is at **high risk** or **low risk** of a heart attack.
    - After the prediction, you will receive health tips and lifestyle recommendations tailored to the prediction result. 👨‍⚕️💡

    **Input Parameters Explained**: 📝
    Below are the parameters you provide for the prediction:

    1. **Age**: The age of the individual, which is a major factor in predicting heart disease risk. 🧓👶
    2. **Sex**: Gender, as heart disease risk varies based on sex. Men generally have a higher risk at younger ages.
    3. **Chest Pain Type**: A symptom that helps determine the severity of the heart condition. 
    4. **Resting Blood Pressure**: High blood pressure can lead to heart problems over time. 💉
    5. **Cholesterol Level**: Elevated cholesterol is a major risk factor for heart disease. 🧀
    6. **Fasting Blood Sugar**: Higher blood sugar levels indicate possible risk of heart disease, especially diabetes. 🍬
    7. **Resting ECG**: The results of an electrocardiogram can indicate issues like arrhythmia, which impacts heart health. 📉
    8. **Max Heart Rate**: Maximum heart rate achieved during exercise. A low rate can indicate poor heart function. ❤️‍🩹
    9. **Exercise-Induced Angina**: Whether chest pain occurs during physical activity, suggesting potential heart issues.
    10. **ST Depression**: This measures how much the ST segment in an ECG drops, signaling possible heart attack risk. ⚡
    11. **Major Vessels**: The number of major coronary vessels affected can indicate the severity of heart disease.
    12. **Thalassemia**: A blood disorder that can impact heart health, causing complications over time.

    **Visualization Insights** 📊:
    After the prediction, we provide a **visualization** of your health data, comparing your input values with the normal thresholds for each parameter. 
    The bar chart will show:
    
    - **Low, Normal, or High** classification for each health parameter based on predefined thresholds.
    - For example, if your cholesterol level is high, it will be highlighted as "High," prompting you to take action like modifying your diet. 🥦
    - This makes it easier for you to understand what areas of your health need improvement and where you're doing well. 🚴‍♀️
    
    The classification labels ("Low", "Normal", "High") are added above each bar in the visualization to help you interpret your results clearly. The chart acts as a **health tracker** that highlights both strengths and areas for improvement.

    **How You Benefit**:
    - You can get an early **heart attack risk assessment** based on your health data, enabling you to take necessary actions to improve your lifestyle.
    - You can visualize your health data and understand how your health metrics compare to recommended thresholds.
    - The health tips provided after the prediction give personalized recommendations to help you reduce your risk of heart disease. 🌱🏋️‍♂️

    We hope this tool helps you stay informed and take steps toward a healthier, longer life! 😃❤️
    """)



# Run the app
if __name__ == "__main__":
    app()
