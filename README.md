# Background 
You have just joined an off-price retailer that specialises in selling branded products at 
discounted prices. The business wants to understand which product types, brands, and  
suppliers move fastest, how pricing affects inventory turnover, and how customer shopping 
habits vary by region and time.

# Task

## Data Cleaning and Preparation 
●  Load the data from offprice_transactions.csv. 
●  Check for and address missing, inconsistent, or outlier values. 
●  Convert columns to appropriate data types, especially for dates and prices. 
●  Summarise the main steps you took and any challenges encountered. 
 
## Exploratory Data Analysis (EDA) 
Address the following business questions: 
●  Which product categories, brands, and suppliers sell the most units, and which 
generate the most revenue? 
●  How does discount percentage (OriginalPrice vs. DiscountedPrice) relate to sales 
volume for top brands and suppliers? 
●  What is the average inventory turnover per store (units sold per store per week)? 
●  How does clearance pricing impact sales volume and returns? 
●  Compare customer purchase behavior across different regions. 


## redictive Task 
Predict whether a given product will be sold on clearance or not. 
●  Create a target variable: ClearanceFlag (Yes/No). 
●  Select appropriate features (category, brand, supplier, region, discount, etc.). 
●  Split the data into training and test sets. 
●  Train a simple classification model (e.g., logistic regression, decision tree). 
●  Report on key metrics (accuracy, precision, recall) and explain your process