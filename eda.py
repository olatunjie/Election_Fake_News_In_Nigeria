#This is an Exploratory Data Analysis (EDA) script for analyzing election-related fake news data.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 1. Load the data
# Using the structure from the provided CSV 
data = pd.read_csv('election_fake_news_data.csv')

# 2. Preprocessing: Define Likert Mappings
# 1=Strongly Disagree/Very Infrequently, 5=Strongly Agree/Very Frequently 
likert_map = {
    'Strongly agree': 5, 'Agree': 4, 'Neutral': 3, 'Disagree': 2, 'Strongly disagree': 1,
    'Very Frequently': 5, 'Somewhat Frequently': 4, 'Occasionally': 3, 
    'Somewhat Infrequently': 2, 'Very Infrequently': 1
}

# Apply mapping to Likert columns
likert_cols = [col for col in data.columns if 'Awar_' in col or 'Neg_Influ' in col or 'Inoculatoin' in col]
for col in likert_cols:
    data[col] = data[col].map(likert_map)

# 3. Handle Missing Values
# Row 30 noted to have null values in Influence metrics 
data = data.fillna(data.mean(numeric_only=True))

# 4. Descriptive Statistics
# Calculate Mean and Standard Deviation for constructs
# Formula: SD = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}}
summary_stats = data[likert_cols].describe().T[['mean', 'std']]

# 5. Exploratory Data Analysis (EDA)
# Demographic Frequencies 
age_dist = data['Age'].value_counts(normalize=True) * 100
gender_dist = data['Gender'].value_counts()

# 6. Correlation Analysis (Spearman's Rho)
# Formula: \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
awareness_total = data[[c for c in likert_cols if 'Awar' in c]].mean(axis=1)
influence_total = data[[c for c in likert_cols if 'Neg' in c]].mean(axis=1)
corr, p_value = spearmanr(awareness_total, influence_total)

# 7. Visualization Logic
plt.figure(figsize=(10, 6))
sns.barplot(x=gender_dist.index, y=gender_dist.values)
plt.title('Gender Distribution of Respondents')
plt.show()

print(f"Spearman Correlation (Awareness vs Influence): {corr:.3f}")