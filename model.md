# 4. Predictive task

The goal is to predict whether a given product will be sold on clearance or not.

```{python}
#| code-fold: true
#| label: current-data

df.info()
```

## 4.1. Target encoding

```{python}
#| code-fold: true
#| label: target-encoding

# Convert Target feature into Integer

df["ClearanceFlag"] = (df["ClearanceFlag"] == 'Yes').astype(int)

```

## 4.2. Feature Selection

Feature selection is a very important step in machine learning, just like the saying `garbage in, garbage out`, feeding the wrong features may negatively impact the model performance.

There are numerous way to select what features, recursive selection, mutual information, to more advanced methods making use of evolutionary algorithms such as particle swarm optimization or grey wolf.

In this project, we will follow a simple `mutual information(MI)`, a concept from information theory telling us how much we can learn from a feature if we know the value of of another. 

**Data Splitting**

```{python}
#| code-fold: true
#| label: data-splitting


# Split
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

len(df_train),len(df_val),len(df_test)
```

**Remove target feature**

```{python}
#| code-fold: true
#| label: remove-target


## ---- Extract and create "ClearanceFlag" for the different splits
y_train = df_train["ClearanceFlag"].values
y_val = df_val["ClearanceFlag"].values
y_test = df_test["ClearanceFlag"].values

### ------ Delete "ClearanceFlag"from the splits
del df_train['ClearanceFlag']
del df_val['ClearanceFlag']
del df_test['ClearanceFlag']
```

**Numerical and Categorical features**

We drop out some features (TransactionID, ProductID, customerID) as they will not add value to our model.

```{python}
#| code-fold: true
#| label: df-head

df.Year.head(5)

df.Year.describe()
```
We remove `Year` as it does not add any value.

**Numerical**

```{python}
#| code-fold: true
#| label: numerical-corr


numerical_cols = ['OriginalPrice', 'QuantityAbs','RevenueAbs', 'RevenueFinal', 'Month', 'WeekOfYear', 'DayOfWeek', 'DiscountPercentage','DiscountedPriceBoxCox']


df[numerical_cols].corrwith(df.ClearanceFlag).to_frame('correlation')
```

**Note:**

-   `DiscountPercentage` has a 0.74 correlation with `ClearanceFlag` indicating that higher discount percentage likely to sell the product at clearance which is obvious.
-   `RevenueAbs` (-0.47) and `RevenueFinal` (-0.25) indicate that when revenue increases, the item is unlikely to have been sold at clearance.
-   Naturally increasing the `DiscountedPrice` will result in the unlikelihood of the item being sold at clearance.


**Categorical**

```{python}
#| code-fold: true
#| label: categorical-MI



# Convert `IsReturn` into a categorical variable
df["IsReturn"] = df["IsReturn"].astype("category")

categorical_cols = ['StoreID', 'Brand', 'Supplier', 'Category', 'Subcategory', 'Type', 'Region','IsReturn','TransactionType']



### ----- Define the function to calculate the M.I on the training set ONLY
def calculate_mi(series):
    return mutual_info_score(series, y_train)

#### ---- Calculate MI between 'y' and the categorical variables of the training set ONLY 
df_mi = df_train[categorical_cols].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')

print('Below are the variable with highest M.I score:')
display(df_mi.head(15))
```

**Note:**

-   Knowing the `Type` of the product provides more information on whether or not it will be sold at clearance.
-   `TransactionType` and `IsReturn` values are not very informative.
-   `Brand`, `StoreID`, and `Subcategory` values also provide valuable information on the target feature status.
  

**Selected Fearures**

```{python}
#| code-fold: true
#| label: selected-features


# Based on correlation
selected_num = ['OriginalPrice', 'QuantityAbs','RevenueAbs', 'RevenueFinal', 'DiscountPercentage','DiscountedPriceBoxCox']

# Based on MI
selected_cat = ['StoreID', 'Brand', 'Supplier', 'Category', 'Subcategory', 'Type', 'Region']

df_train[selected_num + selected_cat]
df_val[selected_num + selected_cat]
df_test[selected_num + selected_cat]

```


## 4.3. Cross validation

```{python}
#| code-fold: true
#| label: cross-validation


# splits = 5 as we have a lot of data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```


## 4.4. Train

We first build a base model then we try to improve upon it. For a better understanding, I will not make use of third-party package like `optuna` or `hyperopt`.

### 4.4.1. Base Logistic regression

```{python}
#| code-fold: true
#| label: base-LR

## ----- Initialize the encoder: ------
dv = DictVectorizer(sparse=False)

## ---- Apply the transformation on the training set

train_dict = df_train[selected_num + selected_cat].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

## ---- Apply the transformation on the validation set (for evaluation)
val_dict = df_val[selected_num + selected_cat].to_dict(orient='records')
X_val = dv.transform(val_dict)

#### Training on Logistic regression

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state = 42)
model.fit(X_train, y_train)


# Predict on validation set
y_pred = model.predict(X_val)

# Compute metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')  
recall = recall_score(y_val, y_pred, average='weighted')

# Print results
print("Base logistic regression")
print(f"Validation Accuracy : {accuracy:.4f}")
print("Classification report on baseline LR model:")
print(classification_report(y_val, y_pred))
```

:::{.callout-warning title="Overfitting"}

-   We noticed earlier a very high correlation between certain features to the clearance status. 
This results in a model learning for example that whenever `DiscountPercentage` is high, predict `ClearanceFlag`=Yes.
-   A feature like `Revenue` is a target-leaking feature as it was engineered based on the `ClearanceFlag` status.

**Solution**: Drop any feature that is engineered from the target feature and observe the result. 
:::


```{python}
#| code-fold: true
#| label: base-LR-retraining

filtered_numerical_cols = ['OriginalPrice', 'QuantityAbs', 'Month', 'WeekOfYear', 'DayOfWeek']
selected_cat = ['StoreID', 'Brand', 'Supplier', 'Category', 'Subcategory', 'Type', 'Region', 'IsReturn', 'TransactionType']

# DictVectorizer encoding
dv = DictVectorizer(sparse=False)
train_dict = df_train[filtered_numerical_cols + selected_cat].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# DictVectorizer encoding (val)
val_dict = df_val[filtered_numerical_cols + selected_cat].to_dict(orient='records')
X_val = dv.transform(val_dict)



# Train logistic regression
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on validation
y_pred = model.predict(X_val)

# Evaluate performance
accuracy = accuracy_score(y_val, y_pred)


print("Logistic Regression (Leakage-Free Features)")
print(f"  Accuracy : {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
```



:::{.callout-warning title="Logistic Regression Base model"}

-   No overfitting, that is a good sign.
-   The base model never predicts class 1 (Clearance) as the accuracy is 80% which is also the percentage of observations that were sold at normal price.
-   Recall for class 1 = 0 indicating all actual clearance items are missed.
-   Precision for class 1 = 0 indicating when it predicts class 1 (which it doesn't), it's always wrong.
-   Macro avg: Treats both classes equally which exposes failure on class 1.
-   Weighted avg: Dominated by class 0 which looks better but hides the issue.
  

**Possible solutions:**
    -   class_weight='balanced' in Logistic Regression to penalize misclassification of rare class more.
    -   Try a sampling technique such oversampling or undersampling or SMOTE.
    -   Use a tree-based models like decision tree.
:::


### 4.4.2. Logistic regression with Class Weight


```{python}
#| code-fold: true
#| label: LR-class-weight

filtered_numerical_cols = ['OriginalPrice', 'QuantityAbs', 'Month', 'WeekOfYear', 'DayOfWeek']
selected_cat = ['StoreID', 'Brand', 'Supplier', 'Category', 'Subcategory', 'Type', 'Region', 'IsReturn', 'TransactionType']

# --- Encoding with DictVectorizer ---
dv = DictVectorizer(sparse=False)

# Train encoding
train_dict = df_train[filtered_numerical_cols + selected_cat].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# Validation encoding
val_dict = df_val[filtered_numerical_cols + selected_cat].to_dict(orient='records')
X_val = dv.transform(val_dict)


# --- Train Logistic Regression with class_weight ---
model = LogisticRegression(solver='liblinear',
                           C=1.0,
                           max_iter=1000,
                           random_state=42,
                           class_weight='balanced')

model.fit(X_train, y_train)

# --- Predict & Evaluate ---
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')

print(" Logistic Regression with class_weight='balanced'")
print(f"  Accuracy : {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
```


:::{.callout-warning title="LR Underfitting"}

-   Imbalance remains an issue: Even though class weighting was used, class 1 (the minority class) is still poorly predicted.
-   High precision on class 0: Model is very good at avoiding false positives for class 0.
-   Low precision on class 1: Only 20% of predicted positives are truly class 1 → lots of false positives for minority class.
-   Recall for class 1 is 0.49: It recovers ~49% of true class 1 cases → improved sensitivity, but at a precision cost.
-   Overall accuracy is misleading: 51% accuracy is barely better than random guessing because the class distribution is imbalanced.
:::

### 4.4.3. Decision tree

```{python}
#| code-fold: true
#| label: decision-tree

# Define hyperparameters to tune for Decision Tree
max_depth_values = [3, 5, 10, None]
min_samples_leaf_values = [1, 5, 10]

# Prepare dictionary to store results
results = {}

# Sampling methods dict
sampling_methods = {
    'SMOTE': SMOTE(random_state=42),
    'Undersampling': RandomUnderSampler(random_state=42),
    'Oversampling': RandomOverSampler(random_state=42)
}

for sampling_name, sampler in sampling_methods.items():
    print(f"\n--- Sampling: {sampling_name} ---")

    # Apply sampling on training data
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    best_f1 = 0
    best_params = None
    best_model = None

    # Hyperparameter tuning loop
    for max_depth in max_depth_values:
        for min_samples_leaf in min_samples_leaf_values:

            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42
            )

            dt.fit(X_resampled, y_resampled)
            y_pred = dt.predict(X_val)

            # Evaluate with macro F1 
            from sklearn.metrics import f1_score
            f1 = f1_score(y_val, y_pred, average='macro')

            if f1 > best_f1:
                best_f1 = f1
                best_params = (max_depth, min_samples_leaf)
                best_model = dt

    # Final evaluation with best model
    y_pred_best = best_model.predict(X_val)

    print(f"Best params: max_depth={best_params[0]}, min_samples_leaf={best_params[1]}")
    print(f"Best macro F1-score: {best_f1:.4f}")
    print(f"Accuracy: {accuracy_score(y_val, y_pred_best):.4f}")
    # print(f"Precision (macro): {precision_score(y_val, y_pred_best, average='macro'):.4f}")
    # print(f"Recall (macro): {recall_score(y_val, y_pred_best, average='macro'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_best))
```


:::{.callout-important title="Decision Tree Underfitting"}

-   Macro F1 is flat (~0.50) across all three — the model handles imbalance slightly better than random but struggles with minority class.
-   Class 1 Recall ~0.22–0.27: Some improvement over default behavior, but still poor.
-   Accuracy is misleading, driven mainly by the dominant class (class 0 with 80%+ support).
-   Precision for class 1 is low (~0.20) indicating many false positives.

  

**Improvement alternatives:**

-   Tune the weight in a cost-sensitive way.
-   Use `Grid search` or `Random search` or even `simulated annealing`
-   Perform more feature engineering.
:::