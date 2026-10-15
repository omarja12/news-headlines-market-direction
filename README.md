# News Headlines and Market Direction

Can the day's top 25 world-news headlines predict whether the Dow Jones closes up or down? An NLP classification study on 1,690 trading days, built to answer the question honestly: time-ordered validation, all fitting inside the folds, and a baseline to beat.

The full analysis, with outputs, is in [`news_market_direction.ipynb`](news_market_direction.ipynb).

## Result

| Model | CV accuracy | CV ROC AUC | Test accuracy | Test ROC AUC |
|---|---|---|---|---|
| Majority class | 0.547 | 0.500 | 0.509 | 0.500 |
| Logistic regression | 0.515 | 0.488 | 0.497 | 0.526 |
| Naive Bayes | 0.523 | 0.489 | 0.512 | 0.540 |
| Linear SVM | 0.507 | 0.490 | 0.500 | 0.534 |
| MLP | 0.504 | 0.470 | 0.533 | 0.534 |

No model beats the baseline. The headlines carry no usable signal for next-close direction in this data.

**Leakage check.** Fitting TF-IDF and feature selection on all labelled days and then cross-validating with shuffled folds lifts the same logistic regression from 0.515 to **0.611** accuracy. That ten-point gain is pure leakage: the selector has seen the validation labels, and shuffled folds train on the future.

## Method

1. **Text cleaning:** lower-casing, contraction expansion, removal of links, mentions, hashtags, digits and punctuation, stop words and one-letter tokens, then WordNet lemmatisation.
2. **Features:** TF-IDF on 1- and 2-grams (min_df 10, max_df 0.8), then the 300 terms with the highest ANOVA F-score.
3. **Models:** logistic regression, multinomial Naive Bayes, linear SVM and an MLP (80-60-40), each as one scikit-learn pipeline with the vectoriser and selector, against a majority-class baseline.
4. **Validation:** days sorted by date; 5-fold `TimeSeriesSplit` on the first 80% of days, then a single test on the last 20%.

## Run it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute news_market_direction.ipynb --output executed.ipynb
```

The GitHub Actions workflow executes the notebook on every push.

## Data

Top 25 daily headlines from Reddit's r/worldnews paired with the Dow Jones Industrial Average's daily direction, as published in the Kaggle dataset *Daily News for Stock Market Prediction*.
