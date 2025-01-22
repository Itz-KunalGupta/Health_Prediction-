# Health_Prediction_Model  🩺🤖
Can be made to  Predict Multiple Disease 


Overview 🌟

This project is a cutting-edge health prediction system leveraging machine learning to assess the likelihood of health issues like heart attacks etc  based on various health parameters. 💡

We used two machine learning algorithms to ensure robust predictions. The system showcases not only the likelihood of a condition but also provides insights when predictions differ, including confidence percentages. 🧠✨ With an accuracy of 83.7% and well-balanced recall of 97% in one model & 87% in other model , this project highlights the importance of precision in health analytics. 🏆

Data Collection 📊

The dataset was meticulously collected using web scraping from reputable government sources. Here's a summary of the sources we tapped into:

Mendeley: Academic reports and research papers on health predictions. 📚

NHM Health Statistics Information Portal: Key health indicators in India, focusing on disease prevalence and outcomes. 🌐

Open Government Data Platform India: Comprehensive datasets from various government ministries. 🏛️

National Health Portal: Authenticated health information and disease risk factors. ✅

Health Management Information System (HMIS): Detailed state-level health management statistics. 📈

Web Scraping Methodology 🕸️

To ensure high-quality data:

Specific pages containing the data were identified.

Requests library was used to fetch HTML content.

BeautifulSoup parsed the HTML to extract tables and statistics. 🛠️

Data was cleaned and formatted for analysis. 🧹

Data Preprocessing 🧪

Handled missing values to ensure clean datasets.

Normalized numerical features for better model performance.

Encoded categorical variables using one-hot encoding. 🔢

The processed data was saved as .csv files and loaded during training and testing.

Model Building ⚙️

We implemented two machine learning algorithms:

Logistic Regression:

Simple and efficient for binary classification problems.

Provided a baseline to evaluate the complexity of the problem. 📊

Random Forest:

Used for its ability to handle feature interactions and provide high accuracy.

Gave probabilistic outputs, which we utilized to show confidence levels in predictions. 🌲

The logic was to use these models in parallel and:

Display the average prediction when both agree. ✅

Show percentage-based confidence when outputs differ. 📉📈

The models were saved as .pkl files for easy deployment. 🗂️

User Interface 🚀

The UI was built using Streamlit, providing a clean, modern, and interactive experience:

Input Form: Users input their health parameters via a simple form. 🖋️

Prediction Display: Results are shown with clear probabilities and health insights. 📺

Dark/Light Mode: Enhanced usability with theme support. 🌓

Development Environment 🌍

All development and testing were carried out in Google Colab, ensuring seamless integration and powerful computational resources:

Python Libraries: Pandas, NumPy, Scikit-learn, Streamlit, BeautifulSoup, and Requests. 🐍

Version Control: GitHub for code collaboration and version tracking. 🛡️

Key Features ✨

Dual-Model Logic: Combines predictions from two algorithms for accuracy and reliability.

Balanced Recall: Ensures fairness across predictions, crucial for health-related insights.

Interactive UI: Easy-to-use interface built on Streamlit.

Scalability: Models can be retrained with new data, enhancing performance over time. 📈

Results 🎯

Achieved:

83.7% accuracy in both models.

High recall to reduce false negatives.

Transparent results with confidence levels for better decision-making. 🔍

Why Two Algorithms? 🤔

Ensures reliability when models agree.

Provides interpretability and confidence percentages when they differ.

Combines the simplicity of logistic regression with the complexity-handling capability of random forests. 🏅

Contributions 🤝

Feel free to:

Suggest improvements 🌱

Contribute additional datasets 📂

Enhance UI/UX design 🎨

Conclusion 🏁

This health prediction system is a testament to the power of machine learning in addressing real-world health challenges. By combining robust algorithms, thoughtful design, and detailed data analysis, we've created a solution that is not only technically sophisticated but also impactful. 🌟

Let’s make health predictions smarter together! 🚀



Feel free to contribute to this project by providing suggestions or improvements!

---




