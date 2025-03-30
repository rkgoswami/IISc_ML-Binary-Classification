import pandas as pd
from sklearn.model_selection import train_test_split


def handle_missing_data(df, train_df):
    """Handle missing values using metrics from training data"""
    df = df.copy()

    # Numerical features: Populate missing data with median
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    for col in num_cols:
        col_median = train_df[col].median()
        df[col] = df[col].fillna(col_median)

    # Categorical features: Populate missing data with mode
    cat_cols = ['Sex', 'Embarked', 'Pclass', 'Title']
    for col in cat_cols:
        col_mode = train_df[col].mode()[0] # get first mode if multiple
        df[col] = df[col].fillna(col_mode)

    return df

def onehot_encoding(df, train_cols=None):
    """One-hot encoding of categorical features"""
    cat_cols = ['Sex', 'Embarked', 'Pclass', 'Title']
    encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols, prefix_sep='_', columns=cat_cols)

    if train_cols is None:
        train_cols = encoded.columns

    return encoded.reindex(columns=train_cols, fill_value=0)

def standardize_numerical_features(df, train_df=None):
    """Standardize numerical features using train metrics"""
    num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
    df = df[num_cols].copy()

    if train_df is not None:
        means = train_df[num_cols].mean()
        stds = df[num_cols].std()
    else:
        means = df.mean()
        stds = df.std()

    # Handle zero std to avoid division by zero
    stds = stds.replace(0, 1)

    return (df - means) / stds, means, stds

def load_and_preprocess_data():
    # Load data from data.csv
    data = pd.read_csv('data.csv')

    print(f"{chr(0x2714)} Loaded data from file.")

    """
        Feature engineering from redundant feature like 'Name' (unique per passenger which not useful for modeling)
            - Extract title from name like 'Mr', 'Mrs', 'Master', 'Miss' etc
            - It provide more insight into age, marital status and social status that may not be fully captured by standalone feature like 'Sex' or 'Age'.
            - Based on given use-case of Titanic disaster, societal norms prioritized women, children and higher status individual for lifeboat access.
            - Title might improve the model performance, say "Master" (young boys) may have higher survival rate.
            - Similarly "Mrs" (married women) and "Miss" (unmarried women) might behave differently towards survival rate.
    """
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # print("\nTitles extracted:", data['Title'].unique())

    """
     Handling spare title or redundant title
      - Title Counts:
            Main Title: Mr 517, Miss 182, Mrs 125, Master 40, 
            Rare Title: Dr 7, Rev 6, Mlle 2, Major 2, Col 2, Countess 1, Capt 1, Ms 1, Sir 1, Lady 1, Mme 1, Don 1, Jonkheer 1
      - Since, Title like Dr, Major etc might have more social status but categorising them separately will introduce sparsity leading to overfitting
      - Also grouping similar title like "Lady", "Sir" etc as Nobility would require more domain knowledge and would not add any value to the modelling
      - Thus,simply grouping the rare title (value count < 10) to one category as 'Others'
    """
    title_count = data['Title'].value_counts()
    rare_titles = title_count[title_count < 10].index.tolist()
    data['Title'] = data['Title'].replace(rare_titles, 'Rare')

    # print("\nUpdated Title distribution:")
    # print(data['Title'].value_counts())

    print(f"{chr(0x2714)} Completed feature engineering (i.e. Extracted 'Title' from 'Name').")

    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    print(f"{chr(0x2714)} Dropped irrelevant features (like 'PassengerId', 'Name', 'Ticket', 'Cabin').")

    # Split the data
    X = data.drop('Survived', axis=1)   # drop the target column to be used in Y
    y = data['Survived']

    print(f"{chr(0x2714)} Split the data in features and target variable.")


    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% for test
        stratify=y,         # Preserve the class balance (need to check with TA, if we are allow to use this or not)
        random_state=42     # Seed value for make the same split always
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.125,                # 12.5% of 80% = 10% of original data
        stratify=y_train_val,           # Preserve class balance
        random_state=42                 # Reproducibility
    )

    print(f"{chr(0x2714)} Split data into 70% training , 10% validation and 20% test sets.")


    # Handle missing values with mean, mode data
    X_train_clean = handle_missing_data(X_train, X_train)
    X_val_clean = handle_missing_data(X_val, X_train)
    X_test_clean = handle_missing_data(X_test, X_train)

    print(f"{chr(0x2714)} Handle missing values (i.e. Populate numeric type with median and category type with mode).")

    # Encode (one-hot) the training data and get columns
    X_train_encoded = onehot_encoding(X_train_clean)
    encoded_columns = X_train_encoded.columns

    # Encode validation and test using training columns
    X_val_encoded = onehot_encoding(X_val_clean, encoded_columns)
    X_test_encoded = onehot_encoding(X_test_clean, encoded_columns)

    print(f"{chr(0x2714)} Encode categorical features (with One-Hot Encoding).")

    # Standardize using training data
    X_train_scaled, train_means, train_stds = standardize_numerical_features(X_train_clean)
    X_val_scaled, _, _ = standardize_numerical_features(X_val_clean, X_train_clean)
    X_test_scaled, _, _ = standardize_numerical_features(X_test_clean, X_train_clean)

    print(f"{chr(0x2714)} Standardize numerical features based on train data.")

    # Combine features
    X_train_processed = pd.concat([X_train_encoded, X_train_scaled], axis=1)
    X_val_processed = pd.concat([X_val_encoded, X_val_scaled], axis=1)
    X_test_processed = pd.concat([X_test_encoded, X_test_scaled], axis=1)

    # Validation checks
    # result = {
    #     "Dataset": ["Train", "Validation", "Test"],
    #     "Missing Values": [X_train_processed.isnull().sum().sum(), X_val_processed.isnull().sum().sum(), X_test_processed.isnull().sum().sum()],
    #     "Data Shape": [X_train_processed.shape, X_val_processed.shape, X_test_processed.shape]
    # }
    #
    # result_df = pd.DataFrame(result)
    #
    # print("\n-----------------------------------------")
    # print("             Data Validation             ")
    # print("-----------------------------------------")
    # print(result_df)
    # print("-----------------------------------------")

    print(f"{chr(0x2714)} Final processed train split with {X_train_processed.isnull().sum().sum()} missing value with shape {X_train_processed.shape}")
    print(f"{chr(0x2714)} Final processed validation split with {X_val_processed.isnull().sum().sum()} missing value with shape {X_val_processed.shape}")
    print(f"{chr(0x2714)} Final processed test split with {X_test_processed.isnull().sum().sum()} missing value with shape {X_test_processed.shape}")

    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test


if __name__ == "__main__":
    load_and_preprocess_data()







