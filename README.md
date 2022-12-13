# Ethical-Machine-Learning
This week, there are three tasks, two of which are closely related synthesis tasks and one of which is a short exploration task. As usual, you'll be turning in both code and a PDF write-up.

Task 1 and Task 2 encompass building machine learning models, trying to explain them, understanding some of the ways in which they might be unfair or discriminatory, and analyzing one way of attempting to correct these issues. We'll be building models of twodifferent datasets.

We encourage that you write your code in Python 3 since we'll be able to provide the most detailed help if you do so. If you use Python, you can use a Jupyter Notebook (encouraged), or you can just write your code in standard Python (.py) files (also perfectly fine). As usual, you'll be turning in both code and a PDF write-up.

For Task 1, we have a special exception to our usual policy about code reuse. Other than the section where you are creating a model card, you should feel free to use code taken from tutorials (with proper attribution!) even if you are using more than 5 lines of code. These tasks mostly involve very standard model building process, and there's no need to "reinvent the wheel." For the model card section and Task 2, the course's usual code reuse rules (per the syllabus) still apply, however.

Task 1 (Synthesis): Building an ML Model That Requires More Careful Thought (35 points)
In this task, you'll be building a machine learning model using the listings.csv dataset, which is data about Chicago Airbnb rentals uploaded to Kaggle Links to an external site.. In particular, your task is to build a model to predict the price of the rental based on the other features available in the dataset. Note that this problem setting's prediction of a continuous value (the price) contrasts with tasks in previous assignments, in which you made models that tried to predict a binary categorical outcome variable. This tutorial's example of predicting a continuous value Links to an external site.is a good example of a fairly simple linear model for a "regression" task (predicting a continuous variable, as opposed to the "classification" task of predicting a categorical variable). However, many other supervised learning techniques can also be used.

Note that this modeling task is a fair bit more complex than the previous one because you probably can't just use some of the features without thinking carefully about their data type and potential pre-processing steps that would make them more useful for your model.

Using whatever ML technique(s) you want, build a predictive model for the price of Chicago Airbnb rentals. If this course has been your first time ever building machine learning models, please note that in your answer to Question 1 (below) and feel free to build a linear model or other fairly simple model. If it's not your first time, we'll expect a bit more sophistication in your approach.

Write-up Question 1 (5 points): Describe the type of model you built, note your prior experience (or lack thereof) with machine learning, and also provide the the root mean squared error (RMSE) of your model. We'll have an informal competition (with neither prizes nor points awarded) for the most accurate model.

Write-up Question 2 (5 points): Please describe which features (columns of the data) you chose to include in, or exclude from, your model. Why did you choose to include/exclude these particular features? Consider both the overall performance of the model and the ethical issues we've discussed in this class in making your decision.

Write-up Question 3 (5 points): What data pre-processing steps, if any, did you follow, and why? What data type (numerical data vs. dummy-coded categorical data) did you use for the different columns you included in your model? Why?

Write-up Question 4 (10 points): In class, we briefly discussed the approach of model cards. Look more closely at some demo model cards Links to an external site.and also skim the academic paper Links to an external site.describing them. Create your own one-page model card for the model you created for this task, making your own judgments about what you think is important to include. Submit your model card as part of your write-up. No prose is required in answering this question.

Write-up Question 5 (5 points): Now, describe in prose your process for making your model card, focusing on what you chose to include in (and what you chose to exclude from) your model card.

Now, try to understand what features were most important in your prediction. The exact method for doing so is going to depend on the type of model you built. This tutorial Links to an external site.is a good starting point, but feel encouraged to search for other techniques that are specific to the type of model you used.

Write-up Question 6 (5 points): Briefly discuss what input features were important in your model's predictions. What does this exercise reveal about Airbnb pricing in Chicago?

Task 2 (Synthesis): Fair ML (40 points)
For our second task, we will be using the UCI income data set. This data was originally collected from the US census in 1994. It contains demographic data paired with information about whether the person in the entry has annual income over $50,000. While this data is of questionable use for solving data science problems, it is commonly used as a benchmark for machine learning tasks. We've uploaded a variant of that dataset with some extra columns specific to this task as income.csv.

As in previous tasks, please first read the data into a Pandas dataframe.

import pandas as pd

income_data = pd.read_csv("income.csv")

You can read about what each of the variables in income_names means here Links to an external site.. Before we use this data, we'll need to do some preprocessing. predictions1 and predictions2 are not a part of the original dataset, and you should ignore them for now. They will only be used in Question 11.

As in the previous tasks, perform any data cleaning you need and convert the categorical data into dummy coded data. The income column is what you will attempt to predict, and all other columns (except predictions1 and predictions2) will be your initial predictor variables.

As in Task 1, split the data into a train and test set and then use a decision tree to model this data. As your task, perform binary classification to predict whether or not the person has an income above $50,000. That is, encode this column as either a 1/0 or True/False. Use all other variables except predictions1 and predictions2 as your predictive features. Use the default parameters for the decision tree. We'll ask you to vary the variables you use, and to think more carefully about fairness, in subsequent problems.

Write-up Question 7 (5 points): Report your decision tree model's accuracy, precision, recall, and F1 score based on your test set. You should see a pretty high accuracy. Does this mean that this is a good model?

Write-up Question 8 (5 points): Visualize your decision tree. Include the decision tree in your writeup. Examine the facets that contribute to these predictions. What issues do you notice?

Now that you've built an initial model, let's consider fairness. As implied when we left out a few variables in Section 1, certain variables may be considered "sensitive variables," meaning that they are generally regarded as inappropriate bases on which to make decisions. Create a new model where you think more carefully about what variables you use in your model. You are welcome to continue to use a decision tree, or to switch to another model architecture. We will consider this model your proposed model in the rest of this task.

Write-up Question 9 (10 points): For your proposed model, plot the false positive rate (FPR) and false negative rate (FNR) by gender and (separately) also by race. In your write-up, include your graphs and describe in prose any patterns you see. Does your model seem fair through this lens?

Write-up Question 10 (7.5 points): One proposed fairness definition is disparate impact in outcomes Links to an external site.. Disparate impact is defined as the ratio Pr[prediction = 1 | member of a particular group]/Pr[prediction = 1 | not member of a particular group]. Note that you can also formaulate this ratio based on privileged and unprivileged groups; for simplicity, calculate this ratio for each group where the denominator represents all other groups. Consider prediction = 1 to represent a predicted income above $50,000. If this ratio is less than some threshold tau, it is considered to have disparate impact. Evaluate your proposed model using the disparate impact metric looking first at gender, then at race. Explain what you conclude from this measurements. Should we be concerned by these results? Why or why not?

Write-up Question 11 (7.5 points): The column predictions1 and predictions2 in the dataframe contains predictions from two black-box models we've constructed. Examine the predictions in light of gender and race. Do they seem fair by the metrics you have considered so far?

Write-up Question 12 (5 points): "Intersectionality" is a term coined by Kimberlé Crenshaw used to refer to the phenomena in anti-discrimination law where plaintiffs who brought cases alleging discrimination on two bases (e.g., sex and race) would lose cases because the sex-based discrimination claims and the race-based discrimination claims would be evaluated separately, rather than considering the intersection between different demographic categories. For example, Black women alleging discrimination in a seniority system were denied relief because the seniority system did not disadvantage Black employees (compared to all non-Black employees) when ignoring gender. It also did not disadvantage women (compared to all non-women) when ignoring race. However, the seniority system did discriminate against the intersectional category of Black women (compared to all data subjects who were not Black women). Analyze the performance of your proposed model through this intersectionality lens. Since the number of comparisons to make grows exponentially with the number of intersectional groupings, please choose two intersectional groups you consider to be of particular importance and evaluate the models you've developed with respect to those two groups. What do you conclude?

Task 3 (Exploration): Screen Readers (25 points)
This last task is an exploration intended to take at most one hour. When people who are blind or low vision use computers, they typically use screen readers, software that verbalizes and sonifies what is on their screen.

In this task, download a screen reader for your browser. While some of the most fully featured options are expensive commercial tools, we want you to pick one of the free tools. For instance, you could pick a screen reader for Firefox Links to an external site.or use Google's screen reader Links to an external site.or any other alternatives for Chrome.

Once you have done so, visit a handful of webpages and answer the questions below.

Write-up Question 13 (5 points): In a few sentences, state which screen reader you've downloaded (and for which browser), and then briefly describe the key features of the screen reader in a few sentences based on your experience using it.

Write-up Question 14 (10 points): What aspects of web browsing do you feel are captured well by this screen reader?

Write-up Question 15 (10 points): What aspects of web browsing do you feel are captured poorly, or missed entirely, by this screen reader? These are, of course, aspects of the experience of browsing the web that are typically inaccessible to people who are blind or low vision.
