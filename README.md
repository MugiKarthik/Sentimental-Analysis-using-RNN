# Sentimental Analysis using RNN 
This project focuses on sentiment analysis of tweets, specifically classifying them into
positive and negative categories. The dataset consists of 1,600,000 records, each representing a tweet with attributes such as sentiment, ID, date, user, and text. The primary
objective of this analysis is to process and clean the text data, analyze sentiment distribution, and use machine learning and deep learning techniques to classify sentiments
effectively.

# Problem Statement
With the rapid rise of social media, platforms like Twitter generate massive amounts
of user-generated text data every day. Extracting meaningful insights from this data,
especially sentiments, is crucial for organizations seeking to understand public opinion on
various topics, monitor brand perception, and respond to consumer needs. Traditional
sentiment analysis approaches may struggle to capture the nuanced context of language
in real-time due to the informal and varied structure of social media text.
This project aims to develop an efficient sentiment analysis system to classify tweets
as either positive or negative. The core challenges involve preprocessing large amounts of
noisy text data, handling vocabulary variety, and accurately capturing sentiment context
within a limited character count. The project will explore traditional machine learning
methods, such as Logistic Regression, as well as advanced deep learning models, specifically Recurrent Neural Networks (RNNs), to improve sentiment classification accuracy.
By achieving high accuracy, the system can provide actionable insights to help organizations make informed decisions based on public sentiment trends

#  Data Description
The dataset contains the following columns:
• Sentiment: A binary classification where 0 represents negative sentiment, and 4
represents positive sentiment. After replacing the ’4’ with ’1’, the dataset contains
two classes: negative (0) and positive (1).
1
• ID: Unique identifier for each tweet.
• Date: The timestamp of when the tweet was posted.
• Query: The query used to retrieve the tweet (in this case, it’s always ’NO QUERY’).
• User: The username of the individual who posted the tweet.
• Text: The content of the tweet.
After reshaping the data, we focus on a subset containing 200,000 records with 6
columns: Sentiment, ID, Date, Query, User, and Text.
3.1 Data Summary
• Unique values in ’Sentiment’: [0, 1]
• Unique values in ’ID’: IDs of tweets
• Unique values in ’Date’: Timestamps of tweets
• Unique values in ’Query’: [’NO QUERY’]
• Unique values in ’User’: Various usernames
• Unique values in ’Text’: The text content of the tweets

# model
Logistic Regression: The first approach utilized a classical machine learning method,
Logistic Regression, which is known for its simplicity and efficiency. The model performed
reasonably well, achieving a training accuracy of 83.36% and a testing accuracy of 77.19%.
The confusion matrix and classification report revealed that the model was able to classify positive and negative sentiments with a decent level of precision and recall. While
Logistic Regression provides an interpretable model with relatively fast training times, it
might not capture the complexities inherent in sequential data such as text.
Recurrent Neural Network (RNN): To address the limitations of traditional machine learning models, a Recurrent Neural Network (RNN), specifically an LSTM-based
architecture, was employed. The RNN model showed an improvement in the model’s
ability to handle sequential data, achieving a test accuracy of 77.00%. While the accuracy of the RNN was slightly lower than the Logistic Regression model, it is important
to note that the RNN’s strength lies in its ability to capture long-term dependencies and
patterns in text data, which is critical for tasks involving language and sentiment analysis.
Additionally, the RNN model can be further improved by fine-tuning hyperparameters
and exploring more advanced architectures like GRUs or bidirectional LSTMs.
Data Preprocessing and Model Deployment: The development and deployment
of these models highlighted the importance of proper data preprocessing, including text
tokenization, padding, and vectorization, which were crucial for making the models capable of processing textual data. The application of Streamlit to build a simple, interactive
web interface demonstrated how machine learning models can be effectively deployed in
real-world scenarios for end-users.

# Deploying the Web App
Once the app is built, the next step is to deploy it. Streamlit apps can be easily deployed
on platforms like Streamlit Cloud or Heroku, where you can share the app with others.
Streamlit Cloud is a free service where you can host your app with minimal setup, while
Heroku requires more configuration but offers additional features.
To deploy the app:
Streamlit Cloud: Push your project to a GitHub repository, then link it to
Streamlit Cloud. Once connected, Streamlit Cloud will automatically build and
deploy the app.
