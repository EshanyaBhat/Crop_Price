# Importing the required libraries
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models

# Setting the page title and layout
st.set_page_config(page_title="ANN Regression Model")

# Defining the function to load the data
def load_data():
    data = pd.read_csv("F:\Crop_Price\downloaded_file1.csv")
    return data


# Defining the function to preprocess the data
def preprocess_data(data):
    X = data.drop(['Yield'], axis=1)
    y = data['Yield'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    return X_train, X_test, y_train, y_test, sc_X, sc_y

# Defining the function to create the ANN model
def create_model(X_train, sc_X, sc_y):
    model = models.Sequential()
    model.add(layers.Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Defining the Streamlit app
def main():

    data = load_data()

    # Preprocessing the data
    X_train, X_test, y_train, y_test,sc_X,sc_y = preprocess_data(data)

    # Creating the ANN model
    model = create_model(X_train, sc_X, sc_y)

    st.title("ANN Regression Model")
    st.image("ANN.jpg")

    # Adding a sidebar to upload the data file
    with st.sidebar:
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            X_train, X_test, y_train, y_test,sc_X,sc_y = preprocess_data(df)
            model = create_model(X_train, sc_X, sc_y)
    
    if uploaded_file is not None:
        st.success('File successfully uploaded!')

        # Display raw data
        if st.checkbox('Display Raw Data'):
            st.write(df.head())
    
        # Shape of The Data
        if st.checkbox('Shape of Data'):
            st.write(df.shape)
    
        # Data Types
        if st.checkbox('Data Types'):
            st.write(df.dtypes)
    
    # Allow users to select columns to drop
        st.write("###### Drop Columns which are not Required")
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        df = df.drop(columns_to_drop, axis=1)

    #Check missing values
        if st.checkbox('Check missing values'):
            st.write(df.isnull().sum()==0)
            st.success('No Missing Values!')
        else:
        # Remove missing values
            if st.checkbox('Remove Missing Values'):
                df.dropna(inplace=True)
                st.success('Missing Values successfully Removed!')

        # Identify numerical and categorical columns
        num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Display results
        if st.checkbox('Identify Numerical Columns'):
            st.write("Numerical Columns:", num_cols)

        if st.checkbox('Identify Categorical Columns'):
            st.write("Categorical Columns:", cat_cols)

    # Show data statistics
        if st.checkbox('Show Data Statistics'):
            st.write(df.describe())

        st.write('#### Data Visualization')
        Plots = ["Histogram", "Correlation Matrix", "Scatter plot","Bar plot"]
        selected_Plot = st.selectbox("Choose a Plot for Visualization", Plots)

        if selected_Plot == "Histogram":
            selected_column = st.selectbox('Select Column', df.columns)
            fig, ax = plt.subplots()
            ax.hist(df[selected_column])
            st.pyplot(fig)

        elif selected_Plot == "Correlation Matrix":
            corr_matrix = df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, ax=ax)
            st.pyplot(fig)

        elif selected_Plot == "Scatter plot":
            x = st.selectbox('Select the x-axis for scatter plot', df.columns)
            y = st.selectbox('Select the y-axis for scatter plot', df.columns)
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[y])
            st.pyplot(fig)

        elif selected_Plot == "Bar plot":
            col = st.selectbox('Select the column for bar chart', df.columns)
            fig, ax = plt.subplots()
            ax.bar(df[col].value_counts().index, df[col].value_counts())
            st.pyplot(fig)

    

    # Adding a form to allow users to input the model parameters
    with st.form(key='my_form'):
        st.write('## Model Parameters')
        num_epochs = st.number_input('Number of epochs:', value=100)
        batch_size = st.number_input('Batch size:', value=32)
        st.form_submit_button(label='Train Model')

    # Training the model with the selected parameters
    if 'my_form' in st.session_state:
        for i in range(num_epochs):
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
            loss = model.evaluate(X_test, y_test, verbose=0)
            st.write('Epoch:', i+1, 'Loss:', loss)

    # Adding a form to allow users to input the predictor variables
    with st.form(key='predict_form'):
        st.write('## Predict')
        input_data = []
        for i in range(X_train.shape[1]):
            value = st.number_input(f'Enter value for {i+1} variable:', format='%.6f')
            input_data.append(value)
        if st.form_submit_button(label='Predict'):
            input_data = np.array(input_data).reshape(1, -1)
            input_data = sc_X.transform(input_data)
            prediction = model.predict(input_data)
            st.write('Prediction:', prediction[0][0])

# Running the Streamlit app
if __name__ == '__main__':
    main()


