# Churn Prediction 

The aim of this project is to perform a supervised classification of customers as to their probability of churning. To make our predictions we use historical transactional data from the past two years. 

## Structure

Most of the preprocessing steps can be accessed from the classes defined in `src`. The user should run the preprocessing steps and then the `extract_labels.ipynb` notebook. All the modelling is done in the `model.ipynb` notebook. 

Once the model has been trained and that all files have been saved in the `data` folder, the user can run a minimal streamlit application by running from the root directory:

```[bash]
streamlit run src/app.py
```

## Instalation 

The user can install all the required packages by running from the root directory:

```[bash]
pip install -r requirements
```

The python version used for this project is `3.8.12`