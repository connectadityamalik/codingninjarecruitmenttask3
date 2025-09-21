# AIML Recruitment Task 1: Student Well-being and Academic Performance Analysis

## Student Information
- **Name**: [Your Name Here]
- **Student ID**: [Your Student ID]
- **Task**: EDTA Dataset Analysis
- **Date**: September 2025

## 1. Import Required Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


## 2. Data Loading and Initial Exploration


# Load the dataset
df = pd.read_csv('EDTA_dataset.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())


## 3. Data Quality Assessment


print("Missing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")
print("\nBasic Statistics:")
print(df.describe())


## 4. Data Preprocessing

### 4.1 Handle Missing Values


# Check missing value patterns
print("Missing value percentages:")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])

# Fill missing values with median for numerical columns
df['Average_Sleep_Hours'].fillna(df['Average_Sleep_Hours'].median(), inplace=True)
df['Daily_Screen_Time'].fillna(df['Daily_Screen_Time'].median(), inplace=True)
df['Attendance_Percentage'].fillna(df['Attendance_Percentage'].median(), inplace=True)

print("\nMissing values after treatment:")
print(df.isnull().sum())


### 4.2 Handle Duplicates


# Remove duplicate rows
print(f"Before removing duplicates: {len(df)} rows")
df_clean = df.drop_duplicates()
print(f"After removing duplicates: {len(df_clean)} rows")
print(f"Removed {len(df) - len(df_clean)} duplicate rows")


### 4.3 Handle Outliers


# Detect outliers using IQR method
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers

numerical_columns = ['Hours_Study_per_day', 'Average_Sleep_Hours', 'Daily_Screen_Time', 
                    'Attendance_Percentage', 'CGPA']

for col in numerical_columns:
    outliers = detect_outliers(df_clean[col])
    print(f"\n{col}: {len(outliers)} outliers detected")
    if len(outliers) > 0:
        print(f"Outlier range: {outliers.min():.2f} to {outliers.max():.2f}")


### 4.4 Categorical Data Encoding


# Create encoded versions of categorical variables
from sklearn.preprocessing import LabelEncoder

# Encode Extracurricular Activities
df_clean['Extracurricular_Encoded'] = df_clean['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})

# Encode Stress Level
stress_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df_clean['Stress_Level_Encoded'] = df_clean['Stress_Level'].map(stress_mapping)

print("Categorical encoding completed!")
print("\nExtracurricular Activities distribution:")
print(df_clean['Extracurricular_Activities'].value_counts())
print("\nStress Level distribution:")
print(df_clean['Stress_Level'].value_counts())


## 5. Exploratory Data Analysis (EDA)

### 5.1 Study Hours vs CGPA Relationship


plt.figure(figsize=(10, 6))
plt.scatter(df_clean['Hours_Study_per_day'], df_clean['CGPA'], alpha=0.6, color='blue')
plt.xlabel('Hours of Study per Day')
plt.ylabel('CGPA')
plt.title('Relationship between Study Hours and CGPA')
plt.grid(True, alpha=0.3)

# Add correlation coefficient
correlation = df_clean['Hours_Study_per_day'].corr(df_clean['CGPA'])
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
plt.show()

print(f"Correlation between Study Hours and CGPA: {correlation:.3f}")


### 5.2 Sleep Hours vs CGPA Relationship


plt.figure(figsize=(10, 6))
plt.scatter(df_clean['Average_Sleep_Hours'], df_clean['CGPA'], alpha=0.6, color='green')
plt.xlabel('Average Sleep Hours')
plt.ylabel('CGPA')
plt.title('Relationship between Sleep Hours and CGPA')
plt.grid(True, alpha=0.3)

correlation_sleep = df_clean['Average_Sleep_Hours'].corr(df_clean['CGPA'])
plt.text(0.05, 0.95, f'Correlation: {correlation_sleep:.3f}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))
plt.show()

print(f"Correlation between Sleep Hours and CGPA: {correlation_sleep:.3f}")


### 5.3 Screen Time vs CGPA Relationship


plt.figure(figsize=(10, 6))
plt.scatter(df_clean['Daily_Screen_Time'], df_clean['CGPA'], alpha=0.6, color='red')
plt.xlabel('Daily Screen Time (hours)')
plt.ylabel('CGPA')
plt.title('Relationship between Screen Time and CGPA')
plt.grid(True, alpha=0.3)

correlation_screen = df_clean['Daily_Screen_Time'].corr(df_clean['CGPA'])
plt.text(0.05, 0.95, f'Correlation: {correlation_screen:.3f}', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral'))
plt.show()

print(f"Correlation between Screen Time and CGPA: {correlation_screen:.3f}")


### 5.4 Academic Performance across Stress Levels


plt.figure(figsize=(10, 6))
sns.boxplot(x='Stress_Level', y='CGPA', data=df_clean, order=['Low', 'Medium', 'High'])
plt.title('CGPA Distribution across Different Stress Levels')
plt.xlabel('Stress Level')
plt.ylabel('CGPA')
plt.show()

# Statistical comparison
stress_groups = df_clean.groupby('Stress_Level')['CGPA'].agg(['mean', 'std', 'count'])
print("\nCGPA Statistics by Stress Level:")
print(stress_groups)


### 5.5 Extracurricular Activities vs Academic Performance


plt.figure(figsize=(10, 6))
sns.boxplot(x='Extracurricular_Activities', y='CGPA', data=df_clean)
plt.title('CGPA Comparison: Students with vs without Extracurricular Activities')
plt.xlabel('Extracurricular Activities')
plt.ylabel('CGPA')
plt.show()

# Statistical comparison
extra_stats = df_clean.groupby('Extracurricular_Activities')['CGPA'].agg(['mean', 'std', 'count'])
print("\nCGPA Statistics by Extracurricular Participation:")
print(extra_stats)


### 5.6 Correlation Heatmap


plt.figure(figsize=(12, 8))
correlation_matrix = df_clean[numerical_columns + ['Extracurricular_Encoded', 'Stress_Level_Encoded']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix of All Variables')
plt.tight_layout()
plt.show()


### 5.7 Distribution Analysis


fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Variables', fontsize=16)

# Hours of Study
axes[0,0].hist(df_clean['Hours_Study_per_day'], bins=20, alpha=0.7, color='blue')
axes[0,0].set_title('Hours of Study per Day')
axes[0,0].set_xlabel('Hours')

# Sleep Hours
axes[0,1].hist(df_clean['Average_Sleep_Hours'], bins=20, alpha=0.7, color='green')
axes[0,1].set_title('Average Sleep Hours')
axes[0,1].set_xlabel('Hours')

# Screen Time
axes[0,2].hist(df_clean['Daily_Screen_Time'], bins=20, alpha=0.7, color='red')
axes[0,2].set_title('Daily Screen Time')
axes[0,2].set_xlabel('Hours')

# Attendance
axes[1,0].hist(df_clean['Attendance_Percentage'], bins=20, alpha=0.7, color='orange')
axes[1,0].set_title('Attendance Percentage')
axes[1,0].set_xlabel('Percentage')

# CGPA
axes[1,1].hist(df_clean['CGPA'], bins=20, alpha=0.7, color='purple')
axes[1,1].set_title('CGPA Distribution')
axes[1,1].set_xlabel('CGPA')

# Stress Level
stress_counts = df_clean['Stress_Level'].value_counts()
axes[1,2].bar(stress_counts.index, stress_counts.values, alpha=0.7, color='brown')
axes[1,2].set_title('Stress Level Distribution')
axes[1,2].set_xlabel('Stress Level')

plt.tight_layout()
plt.show()


## 6. Key Insights and Findings

### Insight 1: Positive Study-Performance Relationship

print("INSIGHT 1: STUDY HOURS AND ACADEMIC PERFORMANCE")
print("=" * 60)
high_study = df_clean[df_clean['Hours_Study_per_day'] >= 8]['CGPA'].mean()
low_study = df_clean[df_clean['Hours_Study_per_day'] < 6]['CGPA'].mean()
print(f"Average CGPA for students studying ≥8 hours/day: {high_study:.2f}")
print(f"Average CGPA for students studying <6 hours/day: {low_study:.2f}")
print(f"Difference: {high_study - low_study:.2f} points")


### Insight 2: Sleep Impact on Academic Performance

print("\nINSIGHT 2: SLEEP AND ACADEMIC PERFORMANCE")
print("=" * 60)
good_sleep = df_clean[df_clean['Average_Sleep_Hours'] >= 8]['CGPA'].mean()
poor_sleep = df_clean[df_clean['Average_Sleep_Hours'] < 6]['CGPA'].mean()
print(f"Average CGPA for students with ≥8 hours sleep: {good_sleep:.2f}")
print(f"Average CGPA for students with <6 hours sleep: {poor_sleep:.2f}")
print(f"Difference: {good_sleep - poor_sleep:.2f} points")


### Insight 3: Screen Time Impact

print("\nINSIGHT 3: SCREEN TIME AND ACADEMIC PERFORMANCE")
print("=" * 60)
high_screen = df_clean[df_clean['Daily_Screen_Time'] > 10]['CGPA'].mean()
low_screen = df_clean[df_clean['Daily_Screen_Time'] <= 6]['CGPA'].mean()
print(f"Average CGPA for students with >10 hours screen time: {high_screen:.2f}")
print(f"Average CGPA for students with ≤6 hours screen time: {low_screen:.2f}")
print(f"Difference: {low_screen - high_screen:.2f} points (lower screen time performs better)")


### Insight 4: Stress Level Impact

print("\nINSIGHT 4: STRESS LEVEL AND ACADEMIC PERFORMANCE")
print("=" * 60)
for stress in ['Low', 'Medium', 'High']:
    avg_cgpa = df_clean[df_clean['Stress_Level'] == stress]['CGPA'].mean()
    count = len(df_clean[df_clean['Stress_Level'] == stress])
    print(f"{stress} Stress: Average CGPA = {avg_cgpa:.2f} (n={count})")


### Insight 5: Extracurricular Activities Impact

print("\nINSIGHT 5: EXTRACURRICULAR ACTIVITIES AND ACADEMIC PERFORMANCE")
print("=" * 60)
with_extra = df_clean[df_clean['Extracurricular_Activities'] == 'Yes']['CGPA'].mean()
without_extra = df_clean[df_clean['Extracurricular_Activities'] == 'No']['CGPA'].mean()
print(f"Average CGPA with extracurricular activities: {with_extra:.2f}")
print(f"Average CGPA without extracurricular activities: {without_extra:.2f}")
print(f"Difference: {with_extra - without_extra:.2f} points")


## 7. Export Cleaned Dataset


# Export the cleaned dataset
df_clean.to_csv('EDTA_dataset_cleaned.csv', index=False)
print("Cleaned dataset exported successfully as 'EDTA_dataset_cleaned.csv'")
print(f"\nFinal dataset shape: {df_clean.shape}")
print("\nCleaned dataset summary:")
print(df_clean.info())


## 8. Conclusion

Based on the comprehensive analysis of the student well-being and academic performance dataset, we can conclude:

1. **Study Hours Matter**: Students who study more hours per day tend to achieve higher CGPA scores
2. **Sleep is Crucial**: Adequate sleep (≥8 hours) is associated with better academic performance
3. **Screen Time Impact**: Excessive screen time (>10 hours/day) appears to negatively impact CGPA
4. **Stress Management**: Lower stress levels are associated with better academic performance
5. **Balanced Lifestyle**: Students participating in extracurricular activities show slightly better academic performance, suggesting a balanced approach to student life

These insights can help educational institutions and students make informed decisions about study habits, lifestyle choices, and well-being practices to optimize academic outcomes.
