# Restaurant Market Analysis (Zomato)

> **Project type:** Client project

A data analysis and machine learning project on **9,551 Zomato restaurants in 15 countries**.
It finds out what goes together with a high restaurant rating, shows where the best markets are, and trains models that predict a rating.

## Business questions

- Which countries, cities, and areas have the most and the best-rated restaurants?
- Which cuisines are popular, and which ones are rated highest?
- Do price, votes, table booking, online delivery, or being a chain change the rating?
- Can a restaurant's rating be predicted from simple features?

## Dataset

- Source: [Zomato Market Analysis on Kaggle](https://www.kaggle.com/datasets/srisyra02/zomato-market-analysis) (public data)
- Size: 9,551 rows and 21 columns
- The notebook downloads the data by itself, so you do not need to download a CSV file.

## What the notebook does

1. **Load and clean the data:** simple column names, country names, Yes/No columns turned into True/False, and new columns such as rated flag, cuisine list, cost rank inside each country, and chain flag.
2. **Explore:** distributions of ratings, votes, cost, and price range.
3. **Locations:** countries, cities, localities in India, and a map of rated restaurants.
4. **Cuisines:** most common cuisines, best-rated cuisines, and cuisine combinations.
5. **What affects ratings:** price range, cost, votes, table booking, online delivery, and chains, with statistical tests (Spearman, Kruskal-Wallis, Mann-Whitney).
6. **Correlation heatmap** of the main features.
7. **Models:** a Gradient Boosting regressor to predict the rating, an improved version with more features, and a classifier for "poor" vs "not poor" restaurants.
8. **Key findings** printed from the data.

## Key results

- 22.5% of restaurants have no rating. The average rating of the rest is 3.44.
- Only 3.2% of restaurants are rated "Excellent" (4.5 or higher).
- India has 90.6% of the restaurants in the dataset.
- The most common cuisine is North Indian. The best-rated cuisine (at least 30 rated restaurants) is Indian, with 4.16.
- Higher price range goes with higher ratings: 3.24, 3.38, 3.78, and 3.89 for price ranges 1 to 4.
- Votes and ratings go together (Spearman 0.68).
- Chains are rated lower on average than independent restaurants (3.24 vs 3.48).

### Model results (rating prediction)

| Model | R² | MAE |
|---|---|---|
| Baseline (always predicts the average) | 0.00 | 0.455 |
| Gradient Boosting | 0.62 | 0.258 |
| Optimized Gradient Boosting | 0.65 | 0.250 |

The first Gradient Boosting model has a 5-fold cross-validation R² of 0.53 (± 0.05).
The classifier for "poor" vs "not poor" restaurants (rating below 2.5 or not) has 97.3% accuracy. Poor restaurants are rare, so this number looks better than it really is.

## How to run

**On Kaggle or Google Colab:** upload the notebook and run all cells.

**On your own computer:**

```bash
pip install -r requirements.txt
jupyter notebook Restaurant_Market_Analyzer_model.ipynb
```

Run all cells from top to bottom. You need an internet connection, because the notebook downloads the dataset from Kaggle. If Kaggle asks you to log in, use your free Kaggle account.

## Project files

```
.
├── Restaurant_Market_Analyzer_model.ipynb   # full analysis and models
├── requirements.txt                         # Python packages
└── README.md
```

## Tools used

Python, pandas, NumPy, SciPy, Matplotlib, Seaborn, scikit-learn, kagglehub
