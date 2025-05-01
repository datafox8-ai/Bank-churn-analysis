
"""
                                  BankChurners   
"""
# BankChurners.csv     
# Churn_Modelling.csv  
  

# 1. BankChurners EDA 
# Importing necessary libraries  
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Enhanced data loading with error handling # BankChurners.csv
def load_data(file_path):      
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print("Data loaded successfully.")
            print(df.head())
            return df
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Clean & preprocess data # Datenbereinigung und Vorverarbeitung
def preprocess_data(df):
    df = df.drop_duplicates()
    df = df.ffill()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

# Perform extended data analysis # Erweiterte Datenanalyse 
def analyze_data(df):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()
    print("Data analysis completed.")

# Generate Excel report from dataset # Reporting-Funktionen 
def generate_report(df):
    with pd.ExcelWriter('data_analysis_report.xlsx') as writer:
        df.to_excel(writer, sheet_name='Raw Data')
        df.describe().to_excel(writer, sheet_name='Descriptive Statistics')
        print("Report generated successfully!")
        
# Pie charts  # Vizualisation
def plotting_pie(df):
    exclude_columns = ['Card_Category']  # Optional
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for column in tqdm(categorical_columns):
        if column in exclude_columns:
            continue

        counts = df[column].value_counts().iloc[:10]

        plt.figure(figsize=(8, 6))
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10, 'color': 'black'}
        )

        plt.title(f'Pie chart of {column}', fontsize=14)
        plt.legend(wedges, counts.index, title=column, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis('equal')  # Force circular shape # Kreisform erzwingen
        plt.tight_layout()
        plt.savefig(f'pie_{column}.png', bbox_inches='tight')
        plt.show()

# Bar Plot and Histogram Function
def plotting_bar(df):
    exclude_columns = ['Gender', 'Education_Level', 'Marital_Status']  #Exclude these columns 
    
    for column in tqdm(df.columns):
        if column in exclude_columns:
            continue  # Skip excluded columns # Überspringe ausgeschlossene Spalten
        
        plt.figure(figsize=(10, 5))
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.histplot(df[column], kde=True, bins=30) # Reduce bin count for better perfomance
        else:
            counts = df[column].value_counts().nlargest(10)  # Top 10 only
            sns.barplot(x=counts.index, y=counts.values)
            plt.xticks(rotation=45)
        plt.title(f'Bar chart of {column}')
        plt.savefig(f'bar_{column}.png', bbox_inches='tight')
        plt.show()
      
# Swarm Plot Function
def plotting_swarm(df, x_col, y_col):
    sample_size = min(1000, len(df))  # Limit to 1000 samples for speed
    df_sample = df.sample(sample_size, random_state=42)  # Downsampling
    
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x=df_sample[x_col], y=df_sample[y_col], size=2)  # Using smaller subset
    plt.title(f'Swarm plot of {y_col} by {x_col}')
    plt.savefig(f'swarm_{y_col}_by_{x_col}.png', bbox_inches='tight')
    plt.show()

# Interaction main interface 
def main():
    file_path = input("Please enter the path to the dataset: ")  # Updated input prompt for clarity
    df = load_data(file_path)
    if df is not None:
        df = preprocess_data(df)
        analyze_data(df)

        while True:
            plot_choice = input("Enter 'pie', 'bar', 'swarm', or 'exit': ").lower()
            if plot_choice in ['pie', 'bar', 'swarm']:
                if plot_choice == 'swarm':
                    x_col = input("Enter X column name for swarm plot: ") #Income_Category
                    y_col = input("Enter Y column name for swarm plot: ") #Credit_Limit
                    if x_col in df.columns and y_col in df.columns:
                        plotting_swarm(df, x_col, y_col)
                    else:
                        print("Invalid column names. Try again.")
                elif plot_choice == 'pie':
                    plotting_pie(df)
                elif plot_choice == 'bar':
                    plotting_bar(df)
            elif plot_choice == 'exit':
                print("Exiting program.")
                break
           
# Main program start 
if __name__ == "__main__":
    main()

    
# 2. Churn Modelling EDA  
# Enhanced data loading with error handling # Churn_Modelling.csv
def load_data(file_path):   
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print("Data loaded successfully.")
            print(df.head())
            return df
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Clean & preprocess data # Daten bereinigen und vorbereiten
def preprocess_data(df):
    df = df.drop_duplicates()
    df = df.ffill()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    if 'Exited' in df.columns:
        df['Exited'] = df['Exited'].astype(str)
    return df

# Churn by Geography
def plot_churn_by_country(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Geography', hue='Exited', data=df)
    plt.title('Customer Churn by Country')
    plt.ylabel('Count')
    plt.savefig("churn_by_country.png", bbox_inches='tight')
    plt.show()

# Stripplot: Age vs. Churn
def plot_stripplot_age(df):
    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Exited', y='Age', data=df, jitter=True, size=3, alpha=0.5)
    plt.title("Age Distribution by Churn")
    plt.xticks([0, 1], ['Stayed', 'Exited'])
    plt.ylabel("Age")
    plt.savefig("stripplot_age_churn.png")
    plt.show()

# Boxplot: Credit Score by Churn
def plot_boxplot_credit(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Exited', y='CreditScore', data=df)
    plt.title("Credit Score by Churn")
    plt.xticks([0, 1], ['Stayed', 'Exited'])
    plt.savefig("boxplot_credit_churn.png")
    plt.show()

# Correlations Heatmap (enhanced)
def plot_enhanced_heatmap(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="BrBG", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Correlation Matrix (Enhanced)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig("enhanced_heatmap.png", bbox_inches='tight')
    plt.show()

# Distributions as subplots
def plot_feature_distributions(df):
    features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.kdeplot(df[feature], fill=True, color='teal')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.tight_layout()
    plt.savefig("feature_distributions.png")
    plt.show()

# Dashboard Summary
def dashboard_summary(df):
    print("\n--- DASHBOARD OVERVIEW ---")
    print(f"Total records: {len(df)}")
    print("\nExited value counts:")
    print(df['Exited'].value_counts())
    print("\nGeography distribution:")
    print(df['Geography'].value_counts())
    print("\nGender distribution:")
    print(df['Gender'].value_counts())

# Main interface
def main():
    file_path = input("Please enter the path to Churn_Modelling.csv: ")
    df = load_data(file_path)
    if df is not None:
        df = preprocess_data(df)
        dashboard_summary(df)

        # Visualisierungen
        plot_churn_by_country(df)
        plot_stripplot_age(df)
        plot_boxplot_credit(df)
        plot_feature_distributions(df)
        plot_enhanced_heatmap(df)

        print("\nTeil 2 Visualisierung abgeschlossen.")

# Start
if __name__ == "__main__":
    main()

# 2.2. Churn_Modelling. Geographycal/ Country-Based Analysis
# Bar chart: Geography vs. Income Category
def plot_geography_vs_income(df):
    if 'Geography' in df.columns and 'Income_Category' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Income_Category', hue='Geography')
        plt.title("BankChurners: Geography vs Income Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("BankChurners_Geography_vs_Income.png")
        plt.show()
    else:
        print("Spalten 'Geography' oder 'Income_Category' fehlen.")

# Donut chart function
def plot_donut_chart(title, values, labels, filename):
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        wedgeprops=dict(width=0.4),
        startangle=90,
        autopct='%1.1f%%'
    )
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# Average values by geography (for Credit Score, Salary, Card Category)
def plot_avg_by_geography(df, column, title, filename):
    if column not in df.columns:
        print(f"Spalte {column} nicht vorhanden.")
        return
    if not np.issubdtype(df[column].dtype, np.number):
        print(f"Spalte {column} ist nicht numerisch.")
        return
    mean_vals = df.groupby('Geography')[column].mean().sort_values()
    plot_donut_chart(title, mean_vals.values, mean_vals.index, filename)

# Churn distribution by geography
def plot_churn_by_geography(df):
    churn_counts = df[df['Exited'] == '1']['Geography'].value_counts()
    plot_donut_chart("Churn Rate by Country", churn_counts.values, churn_counts.index, "donut_churn_by_country.png")

# Number of card categories per country
def plot_card_category_by_geography(df):
    if 'Card_Category' not in df.columns:
        print("Spalte 'Card_Category' fehlt.")
        return
    card_counts = df.groupby('Geography')['Card_Category'].value_counts().unstack().fillna(0)
    card_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
    plt.title("Card Category Distribution by Country")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("stacked_card_category_by_geo.png")
    plt.show()

# Main interface
def main():
    file_path = input("Please enter the path to Churn_Modelling.csv: ")
    df = load_data(file_path)
    if df is not None:
        df = preprocess_data(df)

        # Visualisierungen
        plot_geography_vs_income(df)
        plot_avg_by_geography(df, 'CreditScore', "Average Credit Score by Country", "donut_creditscore_by_geo.png")
        plot_avg_by_geography(df, 'EstimatedSalary', "Average Salary by Country", "donut_salary_by_geo.png")
        plot_card_category_by_geography(df)
        plot_churn_by_geography(df)

        print("\nTeil 3 Visualisierungen abgeschlossen.")

if __name__ == "__main__":
    main()

         
# 3. Dataset Comparison: Churn_Modelling vs. BankChurners *** Datensatzvergleich CM vs. BCh           
# 3.1 Churn-Modelling vs BankChurners – Churn by Income & Country-Based Analysis

# Load files
df1 = pd.read_csv('BankChurners.csv')
df2 = pd.read_csv('Churn_Modelling.csv')

# prepare 'Exited' – in df1, 'Attrition_Flag' appears to be a boolean
# 'Exited' vorbereiten – bei df1 ist 'Attrition_Flag' offenbar ein bool
df1['Churn'] = df1['Attrition_Flag'].replace({True: 'Exited', False: 'Stayed'}).astype('category')
df2['Churn'] = df2['Exited'].replace({1: 'Exited', 0: 'Stayed'}).astype('category')

plt.tight_layout()
plt.show()

# Plot 2 – Comparison of churn by Income/Geography
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=df1, x='Income_Category', hue='Churn', ax=axes[0])
axes[0].set_title("BankChurners – Churn by Income")
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(data=df2, x='Geography', hue='Churn', ax=axes[1])
axes[1].set_title("Churn_Modelling – Churn by Geography")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Plot 3 – Comparison of Credit Limit/Balance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df1, x='Churn', y='Credit_Limit', ax=axes[0])
axes[0].set_title("BankChurners – Credit Limit by Churn")

sns.boxplot(data=df2, x='Churn', y='Balance', ax=axes[1])
axes[1].set_title("Churn_Modelling – Balance by Churn")

plt.tight_layout()
plt.show()

# 3.2 Churn Rate Comparison between BCh & CM 
# Read CSV files
df1 = pd.read_csv('BankChurners.csv')
df2 = pd.read_csv('Churn_Modelling.csv')

# Standardize churn labels
df1['Churn'] = df1['Attrition_Flag'].apply(lambda x: 'Exited' if x == 'Attrited Customer' else 'Stayed')
df2['Churn'] = df2['Exited'].apply(lambda x: 'Exited' if x == 1 else 'Stayed')

# Label data sources
df1['Source'] = 'BankChurners'
df2['Source'] = 'Churn_Modelling'

# Extract relevant columns
df1_subset = df1[['Churn', 'Source']]
df2_subset = df2[['Churn', 'Source']]

# Merge datasets # Zusammenführen
combined_df = pd.concat([df1_subset, df2_subset], ignore_index=True)

# Generate comparison plot
plt.figure(figsize=(8, 5))
sns.countplot(data=combined_df, x='Source', hue='Churn', palette='Set2')
plt.title("Churn Rate Comparison between BankChurners and Churn_Modelling")
plt.ylabel("Number of Customers")
plt.xlabel("Dataset")
plt.tight_layout()
plt.savefig("churn_rate_comparison.png")
plt.show()

 
#4. Predicted Churn Comparison: BankChurners_GH Vs. ChurnModelling *** Prognostizierter Churn-Vergleich: CM & BCh 
# Load data (simulated as before)
bankchurners_df = pd.read_csv('BankChurners.csv')
churn_model_df = pd.read_csv('Churn_Modelling.csv')

# Preparation: simulated 'Exited' labeling for BankChurners
bankchurners_df["Inactivity"] = bankchurners_df["Months_Inactive_12_mon"] / bankchurners_df["Months_Inactive_12_mon"].max()
bankchurners_df["FewContacts"] = 1 - (bankchurners_df["Contacts_Count_12_mon"] / bankchurners_df["Contacts_Count_12_mon"].max())
bankchurners_df["Utilization"] = bankchurners_df["Avg_Utilization_Ratio"]

bankchurners_df["ChurnScore"] = (
    0.4 * bankchurners_df["Inactivity"] +
    0.3 * bankchurners_df["FewContacts"] +
    0.3 * bankchurners_df["Utilization"]
)

# Threshold for predicted 'Exited'
bankchurners_df["Exited"] = (bankchurners_df["ChurnScore"] > 0.6).astype(int)

# Prepare comparison data
bank_counts = bankchurners_df["Exited"].value_counts().rename({0: "Stayed", 1: "Exited"})
churn_counts = churn_model_df["Exited"].value_counts().rename({0: "Stayed", 1: "Exited"})

compare_df = pd.DataFrame({
    "BankChurners": bank_counts,
    "Churn_Modelling": churn_counts
}).T.reset_index().rename(columns={"index": "Dataset"})

compare_melted = compare_df.melt(id_vars="Dataset", var_name="Churn", value_name="Number of Customers")

# Visualization
plt.figure(figsize=(8, 5))
sns.barplot(data=compare_melted, x="Dataset", y="Number of Customers", hue="Churn", palette=["#74c69d", "#f28e2c"])
plt.title("Predicted Churn Comparison: BankChurners vs. Churn_Modelling")
plt.tight_layout()
plt.show()

# Pie charts. 2x Dataset Churn Rate Comparison: Prediction vs Actual 
# Data
labels = ['Retained', 'Churned']
predicted_churn = [0.83, 0.17]
actual_churn = [0.80, 0.20]

colors = ['steelblue', 'darkorange']

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Churn Rate Comparison: Prediction vs. Actual', fontsize=14, weight='bold')

axes[0].pie(predicted_churn, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
axes[0].set_title('Predicted Churn Rate (BankChurners)')

axes[1].pie(actual_churn, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
axes[1].set_title('Actual Churn Rate (Churn_Modelling)')

plt.tight_layout()
plt.show()

#%%

