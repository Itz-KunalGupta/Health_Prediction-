# Health_Prediction_Model
Can be made to  Predict Multiple Disease 

This project was an exciting journey where we combined cutting-edge machine learning techniques, effective UI design, and meaningful data collection to develop a system that predicts the likelihood of health issues such as heart attacks and diabetes. Let's dive into the details! 🚀

🌟 Overview

This project aims to create an intuitive and accurate health prediction system using machine learning algorithms. It leverages health data parameters like age, cholesterol levels, blood pressure, and more to predict the likelihood of heart attacks or diabetes. We developed this project with the following goals in mind:

Providing accurate predictions using machine learning models.

Ensuring a user-friendly interface with Streamlit.

Highlighting the technical complexity and robustness of the system to make it job-ready for employers.

Our prototype achieved 83.7% accuracy on both models, with a well-balanced recall, making it suitable for critical health predictions. 🤖📊

📋 Data Collection

Sources of Data

We collected the data through web scraping and referencing authenticated government health datasets. Here are the key sources:

Mendeley Search: A research repository providing valuable heart attack prediction datasets.

NHM Health Statistics Information Portal: Offers detailed health indicators, prevalence data, and outcomes.

Open Government Data Platform India: A central repository for government-published datasets.

National Health Portal: Provides authenticated health data and risk factor insights.

Health Management Information System (HMIS): Includes detailed health statistics and state-level data.

Web Scraping Methodology

We used Python libraries such as BeautifulSoup and Requests to extract and preprocess data from the above sources. Steps included:

Identifying relevant data pages.

Fetching HTML content using Requests.

Parsing data with BeautifulSoup to extract key statistics and tables.

Cleaning and structuring the data into CSV format for further analysis.

This process ensured we had a robust dataset for training our models. 🛠️

🤖 Machine Learning Models

Why Two Algorithms?

We implemented two algorithms, **Decision tree **and Random Forest, to ensure reliability and provide a comparative perspective. This dual-model approach allowed us to:

Validate predictions through agreement between models.

Handle diverse types of health data with different complexities.

Highlight discrepancies when predictions differed by showing percentage probabilities.

Model Training and Accuracy

Data Preprocessing: Cleaned, normalized, and split the data into training and testing sets.

Feature Selection: Identified critical health parameters like cholesterol, age, and blood pressure.

Algorithm Implementation:

Decision tree : Simple yet effective for binary classification.

Random Forest: Robust for handling non-linear relationships and providing feature importance.

Results:

Achieved 83.7% accuracy on both models.

Maintained balanced recall, critical for minimizing false negatives in health predictions.

🛠️ System Logic

Here’s a breakdown of how the system works:

Input Health Parameters: Users enter parameters like age, cholesterol, and blood pressure.

Model Predictions:

Both models independently predict the likelihood of health risks.

If predictions align, a single result is displayed.

If predictions differ, we display percentage probabilities from each model.

Result Interpretation: Users receive clear outputs with confidence scores, ensuring they understand their risk levels.

🖥️ Development Process

Google Colab Environment

The entire project was developed in Google Colab, leveraging its cloud computing power. Key steps included:

Data Preprocessing:

Loaded raw datasets into Colab.

Cleaned and normalized data for consistency.

Split data into training (70%) and testing (30%) sets.

Model Training:

Implemented Decision tree and Random Forest using sklearn.

Saved trained models as .pkl** files** for deployment.

Streamlit UI Development:

Designed an interactive UI with Streamlit for user input and output display.

Integrated model predictions seamlessly into the UI.

🎨 Features

Dual-Model Prediction: Ensures accuracy and reliability with two algorithms.

Probability Display: Highlights prediction confidence levels for transparency.

User-Friendly UI: Built with Streamlit, offering an intuitive experience.

Portable Models: .pkl files make deployment easy and efficient.

Scalable Design: The system can be expanded to include additional health parameters or diseases.

The User Get some health / lifestyle related  generalized suggestion based on the output .

The users input data is visualized for to get insights from the data which can be helpful in focusing around the parameters  which needs  improvements .



🚀 Technical Complexity

Data Collection: Web scraping from multiple sources required advanced techniques to handle diverse data formats.

Model Integration: Combining predictions from two algorithms with a unified logic demanded careful design.

Cloud-Based Development: Utilizing Google Colab for preprocessing, training, and UI development optimized resources.

Balanced Metrics: Achieving high accuracy with balanced recall ensured the system was practical for real-world health predictions.

Streamlit Integration: Developing an interactive UI that communicates technical results in a user-friendly way added depth to the project.

🔮 Conclusion

This project demonstrates the power of machine learning in addressing critical health issues. With robust data collection, dual-algorithm implementation, and an intuitive UI, we’ve built a system that not only predicts health risks but also highlights the technical complexity and innovation involved.

We’re excited to continue improving and scaling this project. Feel free to contribute or reach out with suggestions! 🌟

Made with ❤️, Python, and Streamlit.

Feel free to contribute to this project by providing suggestions or improvements!

---




