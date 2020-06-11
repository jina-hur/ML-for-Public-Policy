# ML-for-Public-Policy: Final Project
## Project Title
Policy recommendation for preventing obesity, a potential predictor for Coronavirus disease

## Team members
Tetsuo Fujino (tfujino), Jinyoung Hur (jinahur), Takayuki Kitamura (kitamura), Sarah Woo (sarahwoo)

## Running the Code
To run the code, all files listed below should be downloded in the same folder:
 - **ML Project.ipynb**: this is our jupyter notebook to run all relevant codes (code chunk outputs are visible)
 - **data (folder)**: please download all data files (.csv and .xls) listed below
   - consumption_behavior.xls
   - covid.csv
   - education.csv
   - obesity.csv
   - income.csv
   - total_population.csv
 - **pipeline_.py**: machine learning pipeline (helper function) that is used for ML Final Project.ipynb
   - Read data (read_data)
   - Explore data (find_min_max, find_mean_std, plot_distribution)
   - Create training and testing sets (create_train_test)
   - Pre-process data (convert_to_numeric, impute_missing_values, normalize_train, normalize_test)
   - Generate features (one_hot_encoding, discretize_cont_var, categorize_col)
   - Build pipeline for training and testing machine learning models (build_apply_model)
   - Evaluate classifiers (evaluate_classifiers, get_precision, get_recall)
   - Summarize model results (summarize_best_model_result, obtain_match_rate)
   
## Report Submission
- **ML Project Final Report_Fujino, Hur, Kitamura, Woo.pdf**: 10 single-spaced pages final report

## Dependency and required libraries
- python 3.7.6
- pandas 1.0.1
- numpy 1.18.4
- seaborn 0.10.0
- matplotlib 3.1.3
- scikit-learn 0.22.1
- datetime (python's built-in module)

## Reference
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12
