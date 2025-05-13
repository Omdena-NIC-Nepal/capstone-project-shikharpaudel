# Climate Change Impact Assessment and Prediction System for Nepal By Shikhar Paudel Omdena Batch II

## Project Overview

This project aims to assess and predict the impact of climate change on various climatic features in Nepal, including precipitation, temperature, and wind speed. The goal is to provide predictions for future climate conditions based on historical data and environmental factors.

## Key Features

* **Climate Data Analysis**: Data exploration and visualization of climate patterns in Nepal, including temperature trends, precipitation, humidity, and wind speeds across districts and provinces.
* **Feature Engineering**: Data preprocessing to extract relevant features for predictive modeling.
* **Model Training**: Training of machine learning models (Random Forest, Gradient Boosting, and Linear Regression) to predict climate variables based on selected inputs.
* **Prediction**: Allows users to input the year, month, and province, and get predictions for various climate variables.

## Technologies Used

* **Python**: The primary programming language for data processing, modeling, and app development.
* **Streamlit**: A web framework to build interactive data science applications.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical computations.
* **Scikit-Learn**: For machine learning model training and evaluation.
* **Statsmodels**: For statistical models and smoothing (required for certain features).
* **Matplotlib**: For creating data visualizations.
## 📂 Project Structure

```text
Climate-Change-Prediction/
├── dashboard/
│   ├── app.py                       # Main entry point for Streamlit app
│   └── modules/
│       ├── eda.py                   # EDA visualizations
│       ├── featured_view.py        # Feature engineered data viewer
│       ├── footer.py               # Footer and credits
│       ├── Home.py                 # Home page content
│       ├── NLP_sentiment_view.py   # Sentiment analysis display
│       ├── sentiment_utils.py      # Text preprocessing & sentiment scoring
│       └── train_and_predict.py    # Model training and prediction logic
│
├── data/
│   ├── raw/                         # Raw climate and glacier datasets
│   ├── processed/                   # Cleaned and transformed data
│   └── featured/                    # Feature engineered datasets
│       └── feature_climate.csv
│
├── notebooks/
│   ├── eda.ipynb                    # EDA notebook
│   ├── preprocessing.ipynb         # Data preprocessing steps
│   └── save_featured_data.ipynb    # Save feature engineered datasets
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview and instructions


## Installation

### Prerequisites

Ensure you have Python 3.7+ installed. You will also need pip (Python's package installer) to install the dependencies.

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/climate-change-impact-assessment.git
cd climate-change-impact-assessment
```

### 2. Install dependencies:

It is recommended to create a virtual environment before installing the dependencies. You can do this with the following commands:

```bash
python -m venv venv
source venv/bin/activate  # For Linux or macOS
venv\Scripts\activate     # For Windows
```

Then, install the required libraries:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can install the necessary libraries manually:

```bash
pip install pandas numpy scikit-learn streamlit statsmodels matplotlib
```

### 3. Run the application:

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py
```

This will open a web application in your browser where you can interact with the model and make predictions.


## How to Use

1. **Training Models**: Go to the "Model Training and Prediction" section, and click on the "Train Models" button to train the machine learning models using the climate data. Once training is complete, you can proceed to make predictions.

2. **Making Predictions**: After training the models, use the input fields to select the year, month, and province. The app will return predicted values for various climate features like precipitation, temperature, and wind speed.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request. Please make sure to follow the contribution guidelines for a smooth collaboration process.

## Acknowledgements

* Data sources for climate information.
* Libraries and frameworks like Pandas, NumPy, Scikit-learn, and Streamlit that make data science and app development easier.
