# Health_Prediction-Model
Can Predict Multiple Disease 

For now supports heart attack & Diabetes Prediction
# Health Prediction System

 Overview
This project aims to develop a health prediction system that utilizes machine learning algorithms to predict the likelihood of health issues such as heart attacks and diabetes based on various health parameters. The data used for this project was collected through web scraping and refering reports given in this websites from several reputable government health data sources.

 Data Collection
The dataset for this project was gathered using web scraping techniques from the following government websites:

0. https://www.mendeley.com/search/?page=1&query=Heart%20attack%20prediction%20&sortBy=relevance

1. NHM Health Statistics Information Portal: This portal provides various health indicators in India, which were essential for understanding the prevalence of diseases and health outcomes. The data includes statistics related to heart disease and diabetes. [Visit NHM Portal](https://nhm.gov.in)

2. Open Government Data Platform India: This platform serves as a repository for datasets published by various government ministries, including health-related data that is crucial for our predictive modeling. [Visit Open Government Data](https://data.gov.in)

3. National Health Portal: This site offers authenticated health information, including disease prevalence and risk factors for conditions like heart disease and diabetes. [Visit National Health Portal](https://www.nhp.gov.in)

4. Health Management Information System (HMIS): The HMIS portal provided detailed data derived from state-level health management systems, which were instrumental in our analysis. [Visit HMIS Portal](https://hmis.nhp.gov.in)

Web Scraping Methodology
To collect the data, I employed Python libraries such as BeautifulSoup and Requests to scrape relevant information from the above websites. The steps involved included:

- Identifying the specific pages containing the desired data.
- Using Requests to fetch the HTML content of those pages.
- Parsing the HTML using BeautifulSoup to extract relevant tables and statistics.
- Cleaning and formatting the data into a structured format suitable for analysis.

Conclusion
The collected data serves as a foundation for building predictive models using machine learning algorithms. By leveraging this information, we aim to provide insights into potential health risks associated with heart attacks and diabetes.

Feel free to contribute to this project by providing suggestions or improvements!

---




