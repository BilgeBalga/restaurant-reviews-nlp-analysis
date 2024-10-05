# Restaurant Reviews NLP Analysis

## Project Overview
The **Restaurant Reviews NLP Analysis** project aims to leverage Natural Language Processing (NLP) techniques to analyze customer reviews of restaurants. In today's digital age, online reviews are crucial for businesses, influencing potential customers' choices. By categorizing these reviews as positive or negative, restaurant owners can gain valuable insights into customer sentiment and improve their services accordingly.

This project uses a dataset containing of restaurant reviews. The goal is to develop a predictive model that can classify reviews based on sentiment, helping restaurant managers understand customer satisfaction levels, identify strengths and weaknesses, and ultimately enhance the dining experience.

## Technologies Used
- **Python**: The primary programming language utilized for data manipulation and model building.
- **Pandas**: A powerful data analysis library used for data manipulation and analysis.
- **Scikit-learn**: A machine learning library used for creating and evaluating predictive models.
- **NLTK (Natural Language Toolkit)**: A library used for various NLP tasks, including text preprocessing and tokenization.
- **Matplotlib/Seaborn**: Libraries used for data visualization to better understand the data and model performance.

## Project Steps

1. **Data Loading**: The dataset containing restaurant reviews is loaded into the environment for analysis.
2. **Data Exploration**: The dataset is explored to understand its structure, distribution of sentiments, and any missing values that need addressing.
3. **Text Cleaning**: The reviews are cleaned to remove punctuation, numbers, and stop words. Additionally, stemming and lemmatization techniques are employed to normalize words to their base forms.
4. **Tokenization**: The cleaned text is tokenized into words to prepare for the next steps in analysis.
5. **Bag of Words Model Creation**: A Bag of Words model is created to convert the text data into a numerical format, allowing it to be fed into machine learning algorithms.
6. **Train-Test Split**: The dataset is divided into training and testing sets, ensuring that the model can be evaluated on unseen data.
7. **Model Training**: Various machine learning models, such as Logistic Regression, Random Forest, and Support Vector Machines, are trained on the training dataset to find the best-performing model.
8. **Prediction of Results**: The trained model is then used to predict sentiments on the test dataset.
9. **Evaluation of Results**: The model's performance is assessed using accuracy scores, confusion matrices, and classification reports to gauge its effectiveness.

## Results
At the conclusion of the project, the model achieved an accuracy score of **78.80%** on the test dataset. This score indicates a reasonably good performance in classifying reviews based on sentiment. 

The confusion matrix revealed that the model accurately classified a majority of the positive reviews but struggled slightly with negative reviews. The precision, recall, and F1-score metrics were also calculated to provide a more comprehensive understanding of the model's performance. 

### Insights:
- **Customer Satisfaction**: The analysis provides a clearer picture of overall customer satisfaction, allowing restaurant owners to focus on areas needing improvement.
- **Decision-Making**: Insights from the analysis can aid in strategic decision-making, such as menu changes or service enhancements based on customer feedback.
- **Future Work**: There is potential for further improvement by implementing more advanced models, such as deep learning approaches, and by expanding the dataset to include more diverse reviews.

This project not only demonstrates the application of NLP techniques but also emphasizes the significance of data-driven decision-making in the restaurant industry.
