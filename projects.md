This section documents my data science projects, research questions, and data stories I create throughout the semesters.
---
## Project 2


## 1. Problem Definition
## What prediction problem are you addressing?

This project looks at whether NFL team statistics can be used to predict team success. The main goal is to predict how many wins an NFL team will have during a regular season. 
I also use logistic regression to predict whether a team will have a winning season.

## What is the target variable?
For the multiple linear regression model, the target variable is Wins, which represents the total number of regular-season wins for each team.
For the logistic regression model, I created a new target called Winning_Season. A team with 9 or more wins is classified as 1, meaning it had a winning season. 
A team with 8 or fewer wins is classified as 0.
## Is this a classification or regression problem?
The project uses both types of problems. Predicting the exact number of wins is a regression problem because wins are a numerical value. Predicting whether a team has a winning season is a classification problem because the outcome is either 1 or 0.
## Who might benefit from this model or its predictions?
NFL analysts, coaches, sports organizations, and fans could benefit from understanding which team statistics are most closely related to winning. The model could also help demonstrate how statistical and machine-learning methods can be applied to sports data.
## Why is this problem meaningful or worth investigating?
Team success in the NFL depends on many different factors. Looking at statistics such as points scored, points allowed, turnovers, passing yards, and rushing yards can help 
identify which areas of team performance are most closely associated with winning.

## 2. Background and Context
 
NFL teams collect a large amount of statistical information during a season. Statistics such as points scored, points allowed, turnovers, passing yards, and rushing yards can 
be used to evaluate how well a team performs. Previous research supports using these types of statistics to study NFL success. Onwuegbuzie (1999) found that points allowed and 
turnover differential were important factors related to NFL team success. Research by Pelechrinis and Papalexakis (2016) also showed that relatively simple NFL 
game statistics, including turnovers and offensive statistics, can be used to model the probability of winning. Their model achieved strong predictive performance using NFL 
game data. More recent research has also applied machine learning to NFL win prediction. A 2025 study used NFL team statistics from 2003–2023 and compared traditional 
methods with machine-learning models. The study found that statistics such as points scored, points allowed, turnovers, rushing, and passing efficiency can be useful for predicting wins.
These studies support my decision to use points scored, points allowed, turnover differential, passing yards, and rushing yards as features in my models.
## APA References:
Onwuegbuzie, A. J. (1999). Defense or offense? Which is the better predictor of success for professional football teams? Perceptual and Motor Skills, 89(1), 151–158. https://doi.org/10.2466/pms.1999.89.1.151

Pelechrinis, K., & Papalexakis, V. (2016). The anatomy of American football: Evidence from 7 years of NFL game data. PLOS ONE, 11(12), e0168716. https://doi.org/10.1371/journal.pone.0168716

Advancing NFL win prediction: From Pythagorean formulas to machine learning algorithms. (2025). Frontiers in Sports and Active Living.

## 3. Data Description
## Where did the data come from?
The dataset contains NFL team statistics from the 2024 and 2025 regular seasons. The statistics include team wins, points scored, points allowed, turnover differential, passing 
yards, and rushing yards. NFL team-season statistics can be found through Pro Football Reference and other public NFL statistics sources. I used AI to gather data from these 
public NFL statistics sources and cleaned the data later in the project.
## What does each observation or row represent?
Each row represents one NFL team during one regular season.
## How large is the dataset?
The dataset contains 64 observations, representing 32 teams from the 2024 season and 32 teams from the 2025 season.
## What is the target variable?
The main target variable is Wins.
For the classification model, the target is Winning_Season.
## What potential features are available?
The features used in the models are:
- Points Scored
- Points Allowed
- Turnover Differential
- Passing Yards
- Rushing Yards
The dataset also contains Team and Season, but these were not used as model features because they identify the team or season rather than directly measuring team performance.
## What assumptions, restrictions, or limitations affected data collection?
The biggest restriction was the amount of data available for this project. Only two seasons were used, giving the project 64 observations. This is a relatively small dataset for machine learning. The project also uses end-of-season statistics, meaning the model is better at examining the relationship between team performance and wins than making a true preseason prediction.
## 4. Data Understanding and Exploration
I first explored the dataset before training the models. The dataset contained 64 rows and no missing values. The summary statistics showed differences between NFL teams in 
points scored, points allowed, turnover differential, passing yards, rushing yards, and wins. These differences make the variables useful for comparing team performance. The 
correlations with wins showed that turnover differential had the strongest positive relationship with wins. Points scored also had a strong positive relationship, while points allowed had a negative relationship with wins. This makes sense because teams that score more and allow fewer points generally have better records.


The project also created scatterplots comparing each feature with wins. These visualizations helped show the relationships between the statistics and team success.
The features with the strongest relationships were especially useful when deciding which variables to include in the models. I selected points scored, points allowed, turnover differential, passing yards, and rushing yards because they represent different areas of team performance and were supported by previous research.
For the classification target, teams were divided into two groups:
- 1 = 9 or more wins
- 0 = 8 or fewer wins

This allowed logistic regression to predict whether a team had a winning season.

## 5. Data Preparation and Feature Selection
Before modeling, I checked the dataset for missing values. The dataset did not contain missing values, so no missing-value replacement was necessary.
I did not use Team or Season as features because they are identifiers rather than performance measurements. I also did not use winning percentage because it is directly calculated from wins and would create data leakage when predicting wins.
The five features selected were:
- Points Scored
- Points Allowed
- Turnover Differential
- Passing Yards
- Rushing Yards

All five features were already numerical, so categorical encoding was not necessary. I also did not apply scaling because the linear regression model does not require scaled features for this project, and the logistic regression model was still able to train with these numerical variables.
I created the Winning_Season variable using the following rule:
df["Winning_Season"] = (df["Wins"] >= 9).astype(int)


The data was separated by season. 2024 was used for training and 2025 was used for testing. This chronological split prevents the 2025 test data from being used to train the models and provides a more realistic evaluation.
## 6. Baseline and Model Development
## What baseline did you establish?
For the linear regression model, I created a baseline that predicts the average number of 2024 wins for every 2025 team.
The average number of 2024 wins was 8.5, and the baseline MSE was 11.625.
This is an appropriate baseline because it provides a simple prediction to compare against. A useful machine-learning model should perform better than simply predicting the same average number of wins for every team.
## Which models did you train?
I trained two models:
1. Multiple Linear Regression
   - Predicts the exact number of team wins.
2. Logistic Regression
   - Predicts whether a team has a winning season.
## Why were these models appropriate?
Multiple linear regression is appropriate because the main target, Wins, is a numerical value. Logistic regression is appropriate for the second target because Winning_Season has two possible outcomes: winning season or non-winning season.
## Did you tune any model settings?
I did not perform extensive hyperparameter tuning. For logistic regression, I used max_iter=1000 to give the model enough iterations to reach a solution.
How did you ensure the models were compared fairly?
Both models used the same five performance features and the same 2024 training and 2025 testing split. However, the models predict different targets, so their metrics should not be directly treated as a competition between the two models.
## 7. Model Evaluation and Selection
For the linear regression model, I used R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).
The results were:
Metric	Result
Baseline MSE	11.625
Model R²	0.920
Model MSE	0.927
Model MAE	0.845

The model's MSE of 0.927 was much lower than the baseline MSE of 11.625. This shows that the linear regression model performed substantially better than simply predicting the average number of wins for every team.
The R² value of 0.920 means that the model explained about 92% of the variation in 2025 team wins in this test set.
The MAE of 0.845 means that the model's predictions were off by about 0.85 wins on average.
For logistic regression, I used accuracy and a confusion matrix. Accuracy shows the percentage of teams that were correctly classified as either having a winning or non-winning season. The confusion matrix shows the number of correct and incorrect predictions in each category.
Because linear regression predicts the number of wins and logistic regression predicts a category, I did not select one as an overall winner. The linear regression model is the main model for answering my original question about predicting the number of wins, while logistic regression provides an additional way to examine team success.
## 8. Model Interpretation and Insights
The linear regression coefficients show how each feature was associated with the predicted number of wins while the other features were held constant.
The coefficients from the model were:

- Points Scored	0.0100
- Points Allowed	-0.0219
- Turnover Differential	0.1385
- Passing Yards	0.0018
- Rushing Yards	0.0014

Turnover differential had the largest positive coefficient among the features. This suggests that teams with better turnover differentials were associated with higher predicted win totals.
Points scored also had a positive relationship with wins. Points allowed had a negative coefficient, meaning that allowing more points was associated with fewer predicted wins.
Passing yards and rushing yards had smaller coefficients in this model.
The model's example predictions can also be compared with the actual number of wins for each 2025 team. This helps show where the model performed well and where its predictions were less accurate.
The logistic regression confusion matrix provides another way to understand the model's performance. It shows how many teams were correctly classified and where the model made false positive or false negative predictions.
These results show relationships between statistics and wins, but they do not prove that one statistic directly causes a team to win. Other factors that were not included in the dataset could also affect team success.
## 9. Limitations, Ethics, and Reflection
One major limitation of this project is the small dataset. Only two seasons were included, resulting in 64 team-season observations. A larger dataset covering more seasons could provide more reliable results.
Another limitation is that the features are based on end-of-season statistics. Because of this, the model should not be considered a true preseason prediction model. The statistics already reflect how teams performed during the season.
The dataset also does not include factors such as injuries, strength of schedule, coaching changes, player performance, or individual game circumstances. Including these variables could potentially improve the model.
There could also be differences between NFL eras that are not represented when using only two seasons.
Incorrect predictions could cause people to have an inaccurate view of a team's expected performance. For example, a model could predict that a team will have a successful season when it actually performs poorly. Because of this, the model should be used as a statistical tool rather than a guarantee of future performance.
I would not recommend using this model by itself for important real-world decisions. It would be better to combine it with additional statistics and information.
For future work, I would use more NFL seasons and consider adding variables such as strength of schedule, point differential, offensive efficiency, defensive efficiency, injuries, and player-level statistics. Recent research also suggests that machine-learning models can benefit from using a larger set of NFL performance variables.
## 10. Code and Transparency
The project was created using Python in a Jupyter Notebook. The main libraries used were:
- Pandas for organizing and analyzing the data
- Matplotlib and Seaborn for visualizations
- Scikit-learn for machine-learning models and evaluation metrics

My Jupyter Notebook: file:///Users/samwagstaff/Desktop/ds%20studio%202/project_2.html

The NFL team statistics were collected from publicly available NFL statistical information. Pro Football Reference is one source of NFL team-season statistics used in research involving NFL prediction.

I used ChatGPT (GPT-5.6 Luna) as a generative AI tool during the project for a few things. I used it to help explain python and machine-learning concepts to myself, troubleshoot 
coding errors, and understand how different parts of the code worked. I reviewed, edited, and tested the final code myself.
