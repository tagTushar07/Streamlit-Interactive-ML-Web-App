# Exploratory Data Analysis (EDA) on Titanic Dataset
### Internship Practical Task | Data Science & Artificial Intelligence

## Overview
This repository contains the complete Exploratory Data Analysis (EDA) and data preprocessing pipeline for the classic Titanic passenger dataset using Python and Pandas.

## Key Features
- **DataFrame Operations:** Inspected structure, schema, and five-number summaries using `head()`, `info()`, and `describe()`.
- **Data Cleaning & Handling Nulls:** 
  - Checked and validated duplicate records.
  - Handled missing values (`Cabin` dropped due to >77% sparsity; `Age` imputed with median 28.0; `Embarked` imputed with mode 'S').
- **Grouping and Statistical Aggregation:**
  - Evaluated survival rates by passenger class (`Pclass`) and gender (`Sex`).
  - Cross-tabulated survival distributions to observe multi-level interactions.

## Project Structure
- `titanic_eda.ipynb`: Interactive Jupyter Notebook with complete analysis and outputs.
- `main.py`: Standalone, self-contained Python script to execute the analysis.
- `README.md`: Project documentation and summary of findings.

## How to Run
```bash
# Clone the repository
git clone https://github.com/TusharGaming7/Titanic-EDA-Pandas-Analysis.git

# Install requirements
pip install pandas numpy

# Run the Python script
python main.py
```
