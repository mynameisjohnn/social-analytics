
# Analysis 
* Observed Trend 1: The sentiments from CBS on average are the most positive compared to the other four news organizations.
* Observed Trend 2: CNN is the only news organization with an avregae negative sentiment.
* Observed Trend 3: In general for all organizations combined, the occurence of a tweet with positive or neutral sentiment is more frequent than one with a negative sentiment.


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
```


```python
# Import and initialize sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# A list to hold sentiments
sentiments = []

# Target the users
target_users = ("BBC", "CBS", "CNN", "FoxNews", "NYTimes")

# Loop through target users
for user in target_users:
    counter = 0
    
    #Loop through and get last 100 tweets for each target
    public_tweets = api.user_timeline(user, count = 100)
    for tweet in public_tweets:

        # VADER analysis
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        tweets_ago = counter
        tweet_text = tweet["text"]

        # Add sentiments to list
        sentiments.append({"User" : user,
                           "Date": tweet["created_at"],
                           "Compound" : compound,
                           "Positive" : pos,
                           "Negative" : neg,
                           "Neutral" : neu,
                           "Tweets Ago" : counter,
                           "Tweet Text" : tweet_text})
        counter = counter + 1
```


```python
# Create data frame from all sentiments
all_sentiments = pd.DataFrame.from_dict(sentiments)
all_sentiments.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>Sun Mar 11 16:00:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>By 2050, 30-50% of all species could be headin...</td>
      <td>0</td>
      <td>BBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>Sun Mar 11 15:01:05 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>🗑🚀 Tomorrow’s space scientists will have to de...</td>
      <td>1</td>
      <td>BBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Sun Mar 11 14:43:09 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @5liveSport: For the final time...\n\nThe l...</td>
      <td>2</td>
      <td>BBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.0772</td>
      <td>Sun Mar 11 14:00:04 +0000 2018</td>
      <td>0.284</td>
      <td>0.459</td>
      <td>0.257</td>
      <td>Making fake videos just got easier. https://t....</td>
      <td>3</td>
      <td>BBC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.5267</td>
      <td>Sun Mar 11 12:33:04 +0000 2018</td>
      <td>0.145</td>
      <td>0.855</td>
      <td>0.000</td>
      <td>Captain America teams up with Black Widow to g...</td>
      <td>4</td>
      <td>BBC</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Export data frame to csv
all_sentiments.to_csv("all_sentiments.csv", index=False)
```

# Overall Media Sentiment


```python
# Set size for chart
fig = plt.figure(dpi=100)

# Plot scatterplot
for user in target_users:
    dataframe = all_sentiments.loc[all_sentiments["User"] == user]
    plt.scatter(dataframe["Tweets Ago"],dataframe["Compound"], edgecolor="black", label = user)

# Create scatter plot settings and save as png file
plt.xlim(100, -1)
plt.legend(bbox_to_anchor = (1,1))
plt.title("Overall Media Sentiment based on Twitter (03/11/2018)")
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.grid(b=True, which='major', color='1', alpha =1, linestyle='-')
plt.style.use('ggplot')
plt.savefig("Overall_Media_Sentiment_Plot")
plt.show()
```


![png](output_7_0.png)



```python
# Group Users and calculate average
avg_sentiment = all_sentiments.groupby("User")["Compound"].mean()
avg_sentiment
```




    User
    BBC        0.140103
    CBS        0.329151
    CNN       -0.043815
    FoxNews    0.034763
    NYTimes    0.048006
    Name: Compound, dtype: float64



# Sentiment Analysis of Media Tweets


```python
# Set size for chart
fig = plt.figure(dpi=100)

# Plot bar chart
count = 0
for sentiment in avg_sentiment:
    plt.text(count, sentiment+.01, str(round(sentiment,2)))
    count = count + 1
    
# Create bar plot settings and save as png file
x_axis = np.arange(len(avg_sentiment))
xlabels = avg_sentiment.index
plt.bar(x_axis, avg_sentiment, tick_label = xlabels, color = ['red', 'blue', 'yellow', 'green', 'orange'])
plt.title("Sentiment Analysis of Media Tweets (03/11/2018)")
plt.xlabel("New Organizations")
plt.ylabel("Tweet Polarity")
plt.savefig("Sentiment_Analysis_Plot")
plt.show()
```


![png](output_10_0.png)

