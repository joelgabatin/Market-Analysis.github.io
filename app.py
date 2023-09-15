from flask import Flask, render_template, Response, request
import numpy as np
import pandas as pd
import csv
import os
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import matplotlib
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
from pmdarima import auto_arima
from apyori import apriori
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.io as pio
from kmodes.kprototypes import KPrototypes
import plotly.graph_objects as go

def create_figure1():
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot the gender structure with years
    df_gender_year = df.groupby(['gender', 'year'])['customer_id'].nunique().unstack()
    df_gender_year.plot(kind='barh', ax=ax, figsize=(10, 5), color=['yellow', 'blue', 'green'])
    ax.set_xlabel("Number of buyers")
    ax.set_title("Gender Structure by Year")
    ax.legend(title='Year', labels=['2021', '2022', '2023'])

    # Add percentage values to each bar
    for i, v in enumerate(df['gender'].value_counts(normalize=True)):
        ax.text(v, i, f'{v:.1%}', color='black', va='center')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image and textual data as a Flask response
    return Response(output.getvalue(), mimetype='image/png')



def create_figure2_1():
    fig = plt.figure(figsize=(20, 10))

    # Let's see the most item categories that have been purchased
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df, x='category', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure2_2():
    fig = plt.figure(figsize=(20, 10))

    # Let's see the purchases by gender and category
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df, x='category', hue='gender', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category and gender')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure3():
    fig = plt.figure(figsize=(7, 7))
    # Let's see the most item categories that has been purchased
    ax1 = fig.add_subplot()
    sns.countplot(data=df, x='payment_method', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by Payment Method')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure4():
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Let's see which the most sales shopping mall
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'c', 'pink', 'k']
    df_mall = df.groupby("shopping_mall")["sales"].sum()
    df_mall.plot(kind="bar", ax=ax, color=colors)

    ax.set_title('Shopping Mall Revenue Distribution')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Mall')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure5():
    fig = plt.figure(figsize=(10, 10))
    age_bins = [0, 18, 30, 40, 50, 60, 70, 120]
    age_labels = ['<18', '18-30', '30-40', '40-50', '50-60', '60-70', '>70']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    # group customers by age group
    grouped = df.groupby('age_group')

    # calculate mean spending for each age group
    mean_spending = grouped['sales'].mean()

    # plot mean spending by age group

    ax = fig.add_subplot(1, 1, 1)
    mean_spending.plot(kind='bar', ax=ax, color=['orange', 'blue', 'green', 'red', 'purple', 'yellow', 'brown'])
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Mean Spending')
    ax.set_title('Mean Spending by Age Group')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure6():
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Convert invoice_date to datetime and set as index
    df_copy['invoice_date'] = pd.to_datetime(df_copy['invoice_date'])
    df_copy.set_index('invoice_date', inplace=True)

    # Create revenue column
    df_copy['revenue'] = df_copy['quantity'] * df_copy['price']

    # Group by shopping_mall and month
    mall_revenue = df_copy.groupby(['shopping_mall', pd.Grouper(freq='M')])['revenue'].sum()

    # Reset index to convert month back to a column
    mall_revenue = mall_revenue.reset_index()

    # Pivot the data to create columns for each mall
    mall_revenue = mall_revenue.pivot(index='invoice_date', columns='shopping_mall', values='revenue')

    # Create a list of colors for the pie chart (excluding black)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink']

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    mall_revenue.sum().plot(kind='pie', labels=mall_revenue.columns, autopct='%1.1f%%', colors=colors, ax=ax)
    ax.set_title('Total revenue by Mall')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure7():
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.countplot(data=df, x='payment_method', hue='gender', palette='deep', ax=ax)

    ax.set_title('Distribution of purchases by Payment Method and Gender')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Count')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure8():
    dw_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }

    df_sorted_weekday = df.sort_values(['weekday'], ascending=True, axis=0, inplace=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted_weekday['dayofweek'].value_counts() \
        [df_sorted_weekday['dayofweek'].unique()] \
        .plot(kind='line', ax=ax, alpha=0.35)

    ax.set_title('Most Sales of the Week')
    ax.set_ylabel('Number of Transactions')
    ax.set_xlabel('Day')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


# 2021 DATA BELOW
def create_figure9():  # Gender Structure For Number of Buyers
    fig = Figure()
    df_2021 = df[df['year'] == 2021]
    df_gender_year = df_2021.groupby(['gender', 'year'])['customer_id'].nunique().unstack()

    # Plot the gender structure with years
    ax = fig.add_subplot(1, 1, 1)
    df_gender_year.plot(kind='barh', ax=ax, figsize=(10, 5))
    ax.set_xlabel("Number of buyers")
    ax.set_title("Gender Structure for 2021")

    # Add percentage values to each bar
    for i, v in enumerate(df_2021['gender'].value_counts(normalize=True)):
        ax.text(v, i, f'{v:.1%}', color='black', va='center')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure10():
    fig = plt.figure(figsize=(20, 10))

    # Filter the data for 2021
    df_2021 = df[df['year'] == 2021]

    # Let's see the most item categories that have been purchased in 2021
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df_2021, x='category', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category in 2021')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure11():
    fig = plt.figure(figsize=(20, 10))

    # Filter the data for 2021
    df_2021 = df[df['year'] == 2021]

    # Let's see the purchases by gender and category in 2021
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df_2021, x='category', hue='gender', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category and gender in 2021')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure12():
    fig = plt.figure(figsize=(7, 7))

    # Filter the data for 2021
    df_2021 = df[df['year'] == 2021]

    # Let's see the most item categories that have been purchased in 2021
    ax1 = fig.add_subplot()
    sns.countplot(data=df_2021, x='payment_method', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by Payment Method in 2021')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure13():
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Filter the data for 2021
    df_2021 = df[df['year'] == 2021]

    # Let's see which the most sales shopping mall in 2021
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'c', 'pink', 'k']
    df_mall = df_2021.groupby("shopping_mall")["sales"].sum()
    df_mall.plot(kind="bar", ax=ax, color=colors)

    ax.set_title('Shopping Mall Revenue Distribution in 2021')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Mall')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure14():
    fig = Figure(figsize=(10, 10))
    age_bins = [0, 18, 30, 40, 50, 60, 70, 120]
    age_labels = ['<18', '18-30', '30-40', '40-50', '50-60', '60-70', '>70']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    # Filter the data for 2021
    df_2021 = df[df['year'] == 2021]

    # Group customers by age group
    grouped = df_2021.groupby('age_group')

    # Calculate mean spending for each age group
    mean_spending = grouped['sales'].mean()

    # Plot mean spending by age group
    ax = fig.add_subplot(1, 1, 1)
    mean_spending.plot(kind='bar', ax=ax, color=['orange', 'blue', 'green', 'red', 'purple', 'yellow', 'brown'])
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Mean Spending')
    ax.set_title('Mean Spending by Age Group in 2021')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure15():
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Convert invoice_date to datetime and set as index
    df_copy['invoice_date'] = pd.to_datetime(df_copy['invoice_date'])
    df_copy.set_index('invoice_date', inplace=True)

    # Create revenue column
    df_copy['revenue'] = df_copy['quantity'] * df_copy['price']

    # Filter the data for 2021
    df_2021 = df_copy[df_copy['year'] == 2021]

    # Group by shopping_mall and month
    mall_revenue = df_2021.groupby(['shopping_mall', pd.Grouper(freq='M')])['revenue'].sum()

    # Reset index to convert month back to a column
    mall_revenue = mall_revenue.reset_index()

    # Pivot the data to create columns for each mall
    mall_revenue = mall_revenue.pivot(index='invoice_date', columns='shopping_mall', values='revenue')

    # Create a list of colors for the pie chart (excluding black)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink']

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    mall_revenue.sum().plot(kind='pie', labels=mall_revenue.columns, autopct='%1.1f%%', colors=colors, ax=ax)
    ax.set_title('Total revenue by Mall in 2021')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure16():
    fig, ax = plt.subplots(figsize=(7, 7))
    df_2021 = df[df['year'] == 2021]  # Filter the data for 2021
    sns.countplot(data=df_2021, x='payment_method', hue='gender', palette='deep', ax=ax)

    ax.set_title('Distribution of purchases by Payment Method and Gender in 2021')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Count')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure17():
    dw_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }

    df_2021 = df[df['year'] == 2021]  # Filter the data for 2021

    df_sorted_weekday = df_2021.sort_values(['weekday'], ascending=True, axis=0, inplace=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted_weekday['dayofweek'].value_counts() \
        [df_sorted_weekday['dayofweek'].unique()] \
        .plot(kind='line', ax=ax, alpha=0.35)

    ax.set_title('Most Sales of the Week in 2021')
    ax.set_ylabel('Number of Transactions')
    ax.set_xlabel('Day')

    # Customize the x-axis tick labels to display the actual weekdays
    ax.set_xticks(range(7))
    ax.set_xticklabels([dw_mapping[i] for i in range(7)])

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


# 2022 DATA BELOW

def create_figure18():  # Gender Structure For Number of Buyers
    fig = Figure()
    df_2022 = df[df['year'] == 2022]
    df_gender_year = df_2022.groupby(['gender', 'year'])['customer_id'].nunique().unstack()

    # Plot the gender structure with years
    ax = fig.add_subplot(1, 1, 1)
    df_gender_year.plot(kind='barh', ax=ax, figsize=(10, 5))
    ax.set_xlabel("Number of buyers")
    ax.set_title("Gender Structure for 2022")

    # Add percentage values to each bar
    for i, v in enumerate(df_2022['gender'].value_counts(normalize=True)):
        ax.text(v, i, f'{v:.1%}', color='black', va='center')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure19():
    fig = plt.figure(figsize=(20, 10))

    # Filter the data for 2021
    df_2022 = df[df['year'] == 2022]

    # Let's see the most item categories that have been purchased in 2022
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df_2022, x='category', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category in 2022')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure20():
    fig = plt.figure(figsize=(20, 10))

    # Filter the data for 2021
    df_2022 = df[df['year'] == 2022]

    # Let's see the purchases by gender and category in 2021
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df_2022, x='category', hue='gender', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category and gender in 2022')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure21():
    fig = plt.figure(figsize=(7, 7))

    # Filter the data for 2021
    df_2022 = df[df['year'] == 2022]

    # Let's see the most item categories that have been purchased in 2021
    ax1 = fig.add_subplot()
    sns.countplot(data=df_2022, x='payment_method', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by Payment Method in 2022')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure22():
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Filter the data for 2022
    df_2022 = df[df['year'] == 2022]

    # Let's see which the most sales shopping mall in 2021
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'c', 'pink', 'k']
    df_mall = df_2022.groupby("shopping_mall")["sales"].sum()
    df_mall.plot(kind="bar", ax=ax, color=colors)

    ax.set_title('Shopping Mall Revenue Distribution in 2022')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Mall')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure23():
    fig = Figure(figsize=(10, 10))
    age_bins = [0, 18, 30, 40, 50, 60, 70, 120]
    age_labels = ['<18', '18-30', '30-40', '40-50', '50-60', '60-70', '>70']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    # Filter the data for 2021
    df_2022 = df[df['year'] == 2022]

    # Group customers by age group
    grouped = df_2022.groupby('age_group')

    # Calculate mean spending for each age group
    mean_spending = grouped['sales'].mean()

    # Plot mean spending by age group
    ax = fig.add_subplot(1, 1, 1)
    mean_spending.plot(kind='bar', ax=ax, color=['orange', 'blue', 'green', 'red', 'purple', 'yellow', 'brown'])
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Mean Spending')
    ax.set_title('Mean Spending by Age Group in 2022')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure24():
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Convert invoice_date to datetime and set as index
    df_copy['invoice_date'] = pd.to_datetime(df_copy['invoice_date'])
    df_copy.set_index('invoice_date', inplace=True)

    # Create revenue column
    df_copy['revenue'] = df_copy['quantity'] * df_copy['price']

    # Filter the data for 2021
    df_2022 = df_copy[df_copy['year'] == 2022]

    # Group by shopping_mall and month
    mall_revenue = df_2022.groupby(['shopping_mall', pd.Grouper(freq='M')])['revenue'].sum()

    # Reset index to convert month back to a column
    mall_revenue = mall_revenue.reset_index()

    # Pivot the data to create columns for each mall
    mall_revenue = mall_revenue.pivot(index='invoice_date', columns='shopping_mall', values='revenue')

    # Create a list of colors for the pie chart (excluding black)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink']

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    mall_revenue.sum().plot(kind='pie', labels=mall_revenue.columns, autopct='%1.1f%%', colors=colors, ax=ax)
    ax.set_title('Total revenue by Mall in 2022')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure25():
    fig, ax = plt.subplots(figsize=(7, 7))
    df_2022 = df[df['year'] == 2022]  # Filter the data for 2021
    sns.countplot(data=df_2022, x='payment_method', hue='gender', palette='deep', ax=ax)

    ax.set_title('Distribution of purchases by Payment Method and Gender in 2022')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Count')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure26():
    dw_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }

    df_2022 = df[df['year'] == 2022]  # Filter the data for 2021

    df_sorted_weekday = df_2022.sort_values(['weekday'], ascending=True, axis=0, inplace=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted_weekday['dayofweek'].value_counts() \
        [df_sorted_weekday['dayofweek'].unique()] \
        .plot(kind='line', ax=ax, alpha=0.35)

    ax.set_title('Most Sales of the Week in 2022')
    ax.set_ylabel('Number of Transactions')
    ax.set_xlabel('Day')

    # Customize the x-axis tick labels to display the actual weekdays
    ax.set_xticks(range(7))
    ax.set_xticklabels([dw_mapping[i] for i in range(7)])

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


# 2023 DATA BELOW
def create_figure27():  # Gender Structure For Number of Buyers
    fig = Figure()
    df_2023 = df[df['year'] == 2023]
    df_gender_year = df_2023.groupby(['gender', 'year'])['customer_id'].nunique().unstack()

    # Plot the gender structure with years
    ax = fig.add_subplot(1, 1, 1)
    df_gender_year.plot(kind='barh', ax=ax, figsize=(10, 5))
    ax.set_xlabel("Number of buyers")
    ax.set_title("Gender Structure for 2023")

    # Add percentage values to each bar
    for i, v in enumerate(df_2023['gender'].value_counts(normalize=True)):
        ax.text(v, i, f'{v:.1%}', color='black', va='center')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure28():
    fig = plt.figure(figsize=(20, 10))

    # Filter the data for 2021
    df_2023 = df[df['year'] == 2023]

    # Let's see the most item categories that have been purchased in 2022
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df_2023, x='category', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category in 2023')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure29():
    fig = plt.figure(figsize=(20, 10))

    # Filter the data for 2021
    df_2023 = df[df['year'] == 2023]

    # Let's see the purchases by gender and category in 2021
    ax1 = fig.add_subplot(111)
    sns.countplot(data=df_2023, x='category', hue='gender', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by category and gender in 2023')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure30():
    fig = plt.figure(figsize=(7, 7))

    # Filter the data for 2021
    df_2023 = df[df['year'] == 2023]

    # Let's see the most item categories that have been purchased in 2021
    ax1 = fig.add_subplot()
    sns.countplot(data=df_2023, x='payment_method', palette='deep', ax=ax1)
    ax1.set_title('Distribution of purchases by Payment Method in 2023')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure31():
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Filter the data for 2022
    df_2023 = df[df['year'] == 2023]

    # Let's see which the most sales shopping mall in 2021
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'c', 'pink', 'k']
    df_mall = df_2023.groupby("shopping_mall")["sales"].sum()
    df_mall.plot(kind="bar", ax=ax, color=colors)

    ax.set_title('Shopping Mall Revenue Distribution in 2023')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Mall')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure32():
    fig = Figure(figsize=(10, 10))
    age_bins = [0, 18, 30, 40, 50, 60, 70, 120]
    age_labels = ['<18', '18-30', '30-40', '40-50', '50-60', '60-70', '>70']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    # Filter the data for 2021
    df_2023 = df[df['year'] == 2023]

    # Group customers by age group
    grouped = df_2023.groupby('age_group')

    # Calculate mean spending for each age group
    mean_spending = grouped['sales'].mean()

    # Plot mean spending by age group
    ax = fig.add_subplot(1, 1, 1)
    mean_spending.plot(kind='bar', ax=ax, color=['orange', 'blue', 'green', 'red', 'purple', 'yellow', 'brown'])
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Mean Spending')
    ax.set_title('Mean Spending by Age Group in 2023')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure33():
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Convert invoice_date to datetime and set as index
    df_copy['invoice_date'] = pd.to_datetime(df_copy['invoice_date'])
    df_copy.set_index('invoice_date', inplace=True)

    # Create revenue column
    df_copy['revenue'] = df_copy['quantity'] * df_copy['price']

    # Filter the data for 2021
    df_2023 = df_copy[df_copy['year'] == 2023]

    # Group by shopping_mall and month
    mall_revenue = df_2023.groupby(['shopping_mall', pd.Grouper(freq='M')])['revenue'].sum()

    # Reset index to convert month back to a column
    mall_revenue = mall_revenue.reset_index()

    # Pivot the data to create columns for each mall
    mall_revenue = mall_revenue.pivot(index='invoice_date', columns='shopping_mall', values='revenue')

    # Create a list of colors for the pie chart (excluding black)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink']

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    mall_revenue.sum().plot(kind='pie', labels=mall_revenue.columns, autopct='%1.1f%%', colors=colors, ax=ax)
    ax.set_title('Total revenue by Mall in 2023')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure34():
    fig, ax = plt.subplots(figsize=(7, 7))
    df_2023 = df[df['year'] == 2023]  # Filter the data for 2021
    sns.countplot(data=df_2023, x='payment_method', hue='gender', palette='deep', ax=ax)

    ax.set_title('Distribution of purchases by Payment Method and Gender in 2023')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Count')

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_figure35():
    dw_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }

    df_2023 = df[df['year'] == 2023]  # Filter the data for 2021

    df_sorted_weekday = df_2023.sort_values(['weekday'], ascending=True, axis=0, inplace=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted_weekday['dayofweek'].value_counts() \
        [df_sorted_weekday['dayofweek'].unique()] \
        .plot(kind='line', ax=ax, alpha=0.35)

    ax.set_title('Most Sales of the Week in 2023')
    ax.set_ylabel('Number of Transactions')
    ax.set_xlabel('Day')

    # Customize the x-axis tick labels to display the actual weekdays
    ax.set_xticks(range(7))
    ax.set_xticklabels([dw_mapping[i] for i in range(7)])

    # Convert the figure to a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    # Return the PNG image as a Flask response
    return Response(output.getvalue(), mimetype='image/png')


def create_pivot_table():
    # Pivot Table
    # Read the dataset into a DataFrame
    df = pd.read_csv('customer_shopping_data.csv')
    # Create a pivot table overall
    pivot_table = df.pivot_table(index='gender', columns='category', values='quantity', aggfunc='sum')
    pivot_table_reset = pivot_table.reset_index()
    rows = pivot_table_reset.to_dict('records')

    return rows;

def create_pivot_table_2021():

    # Read the dataset into a DataFrame
    df = pd.read_csv('customer_shopping_data.csv')

    # Filter data for the year 2021
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df_2023 = df[df['invoice_date'].dt.year == 2021]

    # Create a pivot table
    pivot_table = df_2023.pivot_table(index='gender', columns='category', values='quantity', aggfunc='sum')
    pivot_table_reset = pivot_table.reset_index()
    rows = pivot_table_reset.to_dict('records')
    return rows;

def create_pivot_table_2022():

    # Read the dataset into a DataFrame
    df = pd.read_csv('customer_shopping_data.csv')

    # Filter data for the year 2022
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df_2023 = df[df['invoice_date'].dt.year == 2022]

    # Create a pivot table
    pivot_table = df_2023.pivot_table(index='gender', columns='category', values='quantity', aggfunc='sum')
    pivot_table_reset = pivot_table.reset_index()
    rows = pivot_table_reset.to_dict('records')
    return rows;

def create_pivot_table_2023():

    # Read the dataset into a DataFrame
    df = pd.read_csv('customer_shopping_data.csv')

    # Filter data for the year 2023
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df_2023 = df[df['invoice_date'].dt.year == 2023]

    # Create a pivot table
    pivot_table = df_2023.pivot_table(index='gender', columns='category', values='quantity', aggfunc='sum')
    pivot_table_reset = pivot_table.reset_index()
    rows = pivot_table_reset.to_dict('records')
    return rows;

app = Flask(__name__)


@app.route('/')
def main():

    menu_active = 'descriptive'

    selected_year = 'Overall'
    selected_graph = '1'
    image_url = '/plot2_1'  # image_url2

    # Read the CSV file
    df = pd.read_csv('customer_shopping_data.csv')
    filtered_df = df
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['sales'] = df['quantity'] * df['price']
    sales_by_year = df.groupby(df['invoice_date'].dt.year)['sales'].sum()
    sorted_sales_by_year = sales_by_year.sort_values(ascending=False)
    top_year_or_month = sorted_sales_by_year.index[0]

    # Compute statistics`
    total_sales = filtered_df['sales'].sum()
    total_sales = '{:,.2f}'.format(total_sales)
    top_category = filtered_df.groupby('category')['sales'].sum().idxmax()
    top_mall = filtered_df.groupby('shopping_mall')['sales'].sum().idxmax()
    selected_value = "Overall"

    # get rows for pivot table
    rows = create_pivot_table()

    return render_template('index.html', image_url=image_url,selected_year=selected_year,selected_graph=selected_graph,
                           menu_active=menu_active,total_sales=total_sales,top_category=top_category,top_mall=top_mall,
                           top_year_or_month=top_year_or_month,selected_value=selected_value, rows=rows )


@app.route('/index_show', methods=['POST'])
def fetch_selected_graph():

    selected_graph = request.form.get('select_graph')
    selected_year = request.form.get('select_year')
    menu_active = 'descriptive'
    image_url = ''
    rows = ''
    heatmap_html = ''

    if(selected_year == '2021'):
        # 2021 DATA

        if(selected_graph == '1'):
            image_url = '/plot10' # image_url2
        if(selected_graph == '2'):
            image_url = '/plot11' # image_url3
        if(selected_graph == '3'):
            image_url = '/plot12' # image_url4
        if(selected_graph == '4'):
            image_url = '/plot16' # image_url8
        if(selected_graph == '5'):
            image_url = '/plot13' # image_url5
        if(selected_graph == '6'):
            image_url = '/plot14' # image_url6
        if(selected_graph == '7'):
            image_url = '/plot9' # image_url1
        if(selected_graph == '8'):
            image_url = '/plot15' # image_url7
        if(selected_graph == '9'):
            image_url = '/plot17' # image_url9
        if(selected_graph == '10'):
            # get rows for pivot table
            rows = create_pivot_table_2021()
        if(selected_graph == '11'):
            # Create a heatmap using plotly
            heatmap = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='Viridis'))

            # Configure the heatmap layout
            heatmap.update_layout(
                title='Correlation Matrix',
                width=800,
                height=800,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                xaxis_tickangle=-45,
                yaxis_autorange='reversed')

            # Convert the heatmap to an HTML string
            heatmap_html = pio.to_html(heatmap, full_html=False)


        df = pd.read_csv('customer_shopping_data.csv')
        filtered_df = df
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        df['sales'] = df['quantity'] * df['price']
        filtered_df = df[df['invoice_date'].dt.year == 2021]
        monthly_sales = filtered_df.groupby(filtered_df['invoice_date'].dt.month)['quantity', 'sales'].sum()
        top_month = monthly_sales['sales'].idxmax()
        top_year_or_month = pd.to_datetime(str(top_month), format='%m').strftime('%B')


    elif(selected_year == '2022'):
        # 2022 DATA

        if(selected_graph == '1'):
            image_url = '/plot19' # image_url2
        if(selected_graph == '2'):
            image_url = '/plot20' # image_url3
        if(selected_graph == '3'):
            image_url = '/plot21' # image_url4
        if(selected_graph == '4'):
            image_url = '/plot25' # image_url8
        if(selected_graph == '5'):
            image_url = '/plot22' # image_url5
        if(selected_graph == '6'):
            image_url = '/plot23' # image_url6
        if(selected_graph == '7'):
            image_url = '/plot18' # image_url1
        if(selected_graph == '8'):
            image_url = '/plot24' # image_url7
        if(selected_graph == '9'):
            image_url = '/plot26' # image_url9
        if(selected_graph == '10'):
            # get rows for pivot table
            rows = create_pivot_table_2022()

        df = pd.read_csv('customer_shopping_data.csv')
        filtered_df = df
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        df['sales'] = df['quantity'] * df['price']
        filtered_df = df[df['invoice_date'].dt.year == 2022]
        monthly_sales = filtered_df.groupby(filtered_df['invoice_date'].dt.month)['quantity', 'sales'].sum()
        top_month = monthly_sales['sales'].idxmax()
        top_year_or_month = pd.to_datetime(str(top_month), format='%m').strftime('%B')

    elif(selected_year == '2023'):

        # 2023 DATA
        if(selected_graph == '1'):
            image_url = '/plot28' # image_url2
        if(selected_graph == '2'):
            image_url = '/plot29' # image_url3
        if(selected_graph == '3'):
            image_url = '/plot30' # image_url4
        if(selected_graph == '4'):
            image_url = '/plot34' # image_url8
        if(selected_graph == '5'):
            image_url = '/plot31' # image_url5
        if(selected_graph == '6'):
            image_url = '/plot32' # image_url6
        if(selected_graph == '7'):
            image_url = '/plot27' # image_url1
        if(selected_graph == '8'):
            image_url = '/plot33' # image_url7
        if(selected_graph == '9'):
            image_url = '/plot35' # image_url9
        if(selected_graph == '10'):
            # get rows for pivot table
            rows = create_pivot_table_2023()

        df = pd.read_csv('customer_shopping_data.csv')
        filtered_df = df
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        df['sales'] = df['quantity'] * df['price']
        filtered_df = df[df['invoice_date'].dt.year == 2023]
        monthly_sales = filtered_df.groupby(filtered_df['invoice_date'].dt.month)['quantity', 'sales'].sum()
        top_month = monthly_sales['sales'].idxmax()
        top_year_or_month = pd.to_datetime(str(top_month), format='%m').strftime('%B')

    else:

        if(selected_graph == '1'):
            image_url = '/plot2_1' # image_url2
        if(selected_graph == '2'):
            image_url = '/plot2_2' # image_url3
        if(selected_graph == '3'):
            image_url = '/plot3' # image_url4
        if(selected_graph == '4'):
            image_url = '/plot7' # image_url8
        if(selected_graph == '5'):
            image_url = '/plot4' # image_url5
        if(selected_graph == '6'):
            image_url = '/plot5' # image_url6
        if(selected_graph == '7'):
            image_url = '/plot1' # image_url1
        if(selected_graph == '8'):
            image_url = '/plot6' # image_url7
        if(selected_graph == '9'):
            image_url = '/plot8' # image_url9
        if(selected_graph == '10'):
            # get rows for pivot table
            rows = create_pivot_table()
        if(selected_graph == '11'):
            df = pd.read_csv('customer_shopping_data.csv')
            # Calculate the correlation matrix
            correlation_matrix = df.corr()

            # Create a heatmap using plotly
            heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis'))

            # Configure the heatmap layout
            heatmap.update_layout(
            title='Correlation Matrix',
            width=600,
            height=600,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_tickangle=-45,
            yaxis_autorange='reversed')

            # Convert the heatmap to an HTML string
            heatmap_html = pio.to_html(heatmap, full_html=False)

        df = pd.read_csv('customer_shopping_data.csv')
        filtered_df = df
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        df['sales'] = df['quantity'] * df['price']
        sales_by_year = df.groupby(df['invoice_date'].dt.year)['sales'].sum()
        sorted_sales_by_year = sales_by_year.sort_values(ascending=False)
        top_year_or_month = sorted_sales_by_year.index[0]

    # Compute statistics`
    total_sales = filtered_df['sales'].sum()
    total_sales = '{:,.2f}'.format(total_sales)
    top_category = filtered_df.groupby('category')['sales'].sum().idxmax()
    top_mall = filtered_df.groupby('shopping_mall')['sales'].sum().idxmax()



    return render_template('index.html',image_url=image_url,selected_year=selected_year,selected_graph=selected_graph,menu_active=menu_active,total_sales=total_sales,
                           top_category=top_category,top_mall=top_mall, top_year_or_month=top_year_or_month, rows=rows,heatmap=heatmap_html)


@app.route('/plot1')
def plot1():
    image = create_figure1()
    return image


@app.route('/plot2_1')
def plot2_1():
    image = create_figure2_1()
    return image


@app.route('/plot2_2')
def plot2_2():
    image = create_figure2_2()
    return image


@app.route('/plot3')
def plot3():
    image = create_figure3()
    return image


@app.route('/plot4')
def plot4():
    image = create_figure4()
    return image


@app.route('/plot5')
def plot5():
    image = create_figure5()
    return image


@app.route('/plot6')
def plot6():
    image = create_figure6()
    return image


@app.route('/plot7')
def plot7():
    image = create_figure7()
    return image


@app.route('/plot8')
def plot8():
    image = create_figure8()
    return image


@app.route('/plot9')
def plot9():
    image = create_figure9()
    return image


@app.route('/plot10')
def plot10():
    image = create_figure10()
    return image


@app.route('/plot11')
def plot11():
    image = create_figure11()
    return image


@app.route('/plot12')
def plot12():
    image = create_figure12()
    return image


@app.route('/plot13')
def plot13():
    image = create_figure13()
    return image


@app.route('/plot14')
def plot14():
    image = create_figure14()
    return image


@app.route('/plot15')
def plot15():
    image = create_figure15()
    return image


@app.route('/plot16')
def plot16():
    image = create_figure16()
    return image


@app.route('/plot17')
def plot17():
    image = create_figure17()
    return image


@app.route('/plot18')
def plot18():
    image = create_figure18()
    return image


@app.route('/plot19')
def plot19():
    image = create_figure19()
    return image


@app.route('/plot20')
def plot20():
    image = create_figure20()
    return image


@app.route('/plot21')
def plot21():
    image = create_figure21()
    return image


@app.route('/plot22')
def plot22():
    image = create_figure22()
    return image


@app.route('/plot23')
def plot23():
    image = create_figure23()
    return image


@app.route('/plot24')
def plot24():
    image = create_figure24()
    return image


@app.route('/plot25')
def plot25():
    image = create_figure25()
    return image


@app.route('/plot26')
def plot26():
    image = create_figure26()
    return image


@app.route('/plot27')
def plot27():
    image = create_figure27()
    return image


@app.route('/plot28')
def plot28():
    image = create_figure28()
    return image


@app.route('/plot29')
def plot29():
    image = create_figure29()
    return image


@app.route('/plot30')
def plot30():
    image = create_figure30()
    return image


@app.route('/plot31')
def plot31():
    image = create_figure31()
    return image


@app.route('/plot32')
def plot32():
    image = create_figure32()
    return image


@app.route('/plot33')
def plot33():
    image = create_figure33()
    return image


@app.route('/plot34')
def plot34():
    image = create_figure34()
    return image


@app.route('/plot35')
def plot35():
    image = create_figure35()
    return image


@app.route('/statistics', methods=['GET', 'POST'])
def statistics():
    years = [2021, 2022, 2023]

    if request.method == 'POST':
        # Get the selected year from the form
        selected_year = int(request.form['year'])

        if selected_year == 0:
            # Compute overall statistics
            filtered_df = df
            top_year = 'Overall'
        else:
            # Filter the dataset based on the selected year
            filtered_df = df[df['year'] == selected_year]
            top_year = selected_year

        # Compute statistics
        total_sales = filtered_df['sales'].sum()
        top_category = filtered_df.groupby('category')['sales'].sum().idxmax()
        top_mall = filtered_df.groupby('shopping_mall')['sales'].sum().idxmax()

        # Render the template with the statistics
        return render_template('statistics.html', total_sales=total_sales, top_category=top_category,
                               top_mall=top_mall, top_year=top_year, years=years, selected_year=selected_year)
    else:
        # Compute overall statistics
        total_sales = df['sales'].sum()
        top_category = df.groupby('category')['sales'].sum().idxmax()
        top_mall = df.groupby('shopping_mall')['sales'].sum().idxmax()
        top_year = 'Overall'

        # Render the template with the overall statistics
        return render_template('statistics.html', total_sales=total_sales, top_category=top_category,
                               top_mall=top_mall, top_year=top_year, years=years, selected_year=0,
                               overall=True)


@app.route('/predict_time_series')
def go_to_predict_time_series():
    menu_active = 'predictive'
    predictive_active = 'time_series'
    return render_template('sales_prediction.html', menu_active=menu_active, predictive_active=predictive_active)


@app.route('/predict_clustering')
def go_to_predict_clustering():
    menu_active = 'predictive'
    predictive_active = 'clustering'
    return render_template('clustering.html', menu_active=menu_active, predictive_active=predictive_active)


@app.route('/csvdatatables')
def go_to_csvdata():
    menu_active = 'csvdatatables'
    # Open the CSV file and read the rows
    with open('customer_shopping_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Use the next() function to read the header row
        header = next(reader)
        # Limit the rows based on the number of rows to display
        limited_rows = [row for i, row in enumerate(reader) if i < 10000]

    return render_template('csv_data.html', rows=limited_rows, header=header, menu_active=menu_active)


@app.route('/upload_new_csvfile', methods=['GET', 'POST'])
def upload_new_csv():
    menu_active = 'upload_new_csv'

    if request.method == 'POST':

        file = request.files['csvfile']
        if file and file.mimetype == 'text/csv':
            # Delete the old CSV file, if it exists
            # if os.path.exists('customer_shopping_data.csv'):
            # remove first the old csv file
            os.remove('customer_shopping_data.csv')
            # Save the uploaded file to disk
            file = request.files['csvfile']
            file.save('customer_shopping_data.csv')
            upload_message = 'CSV FILE Successfully Uploaded'
        else:
            upload_message = 'Invalid file type. Please upload a CSV file only.'

        return render_template('upload_new_csv.html', upload_message=upload_message, menu_active=menu_active)

    else:
        return render_template('upload_new_csv.html', menu_active=menu_active)


@app.route('/times-series-prediction', methods=['GET', 'POST'])
def sales_prediction():

    menu_active = 'predictive'
    predictive_active = 'time_series'

    if request.method == 'POST':
        # Get the user input from the form
        num_months_input = request.form.get('num_months')
        specific_month = request.form.get('specific_month')
        specific_year = request.form.get('specific_year')
        selected_mall = request.form.get('selected_mall')
        prediction_method = request.form.get('prediction_method')


        plot_html = ""
        predicted_sales_text = ""
        error = ""

        if prediction_method == "specific_month_year":

            if(specific_month == "January" and int(specific_year) <= 2023):
                error = "specific_monthyear_error";
            elif(specific_month == "February" and int(specific_year) <= 2023):
                error = "specific_monthyear_error";
            elif(int(specific_year) < 2023):
                error = "specific_monthyear_error";
            else:

                # Filter the data based on the selected mall
                df_selected_mall = df[df['shopping_mall'] == selected_mall]
                df_selected_mall.reset_index(inplace=True)

                # Step 1: Prepare the Data
                df_selected_mall['invoice_date'] = pd.to_datetime(df_selected_mall['invoice_date'])
                df_selected_mall.set_index('invoice_date', inplace=True)
                df_monthly = df_selected_mall['sales'].resample('M').sum()

                # Step 2: Build the Time Series Model
                train_data = df_monthly[:-3]
                test_data = df_monthly[-3:]

                # Use AutoARIMA to automatically select the best ARIMA model
                model = auto_arima(train_data, seasonal=False, trace=True)

                # Convert the specific month and year to a datetime object
                specific_date = pd.to_datetime(f"{specific_month} {specific_year}")

                # Determine the number of months based on the specific date
                num_months = (specific_date.year - df_monthly.index[-1].year) * 12 + (
                            specific_date.month - df_monthly.index[-1].month) + 1

                forecast = model.predict(n_periods=num_months)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pd.date_range(start=df_monthly.index[-1], periods=num_months, freq='M'),
                                         y=forecast, mode='lines', name='Predicted'))
                fig.update_traces(hovertemplate='Month: %{x}<br>Predicted Sales: %{y}')
                fig.update_layout(title='Monthly Sales Prediction', xaxis_title='Date', yaxis_title='Sales')
                fig.update_layout(showlegend=True)
                plot_html = fig.to_html(full_html=False)

                predicted_sales_text = f"Predicted Sales for {selected_mall}:\n"
                for month, sales in zip(pd.date_range(start=df_monthly.index[-1], periods=num_months, freq='M'), forecast):
                    predicted_sales_text += f"{month.strftime('%B %Y')}: {sales}\n"

            return render_template('sales_prediction.html', plot_html=plot_html, num_months=num_months_input,
                                       menu_active=menu_active, predictive_active=predictive_active,
                                       predicted_sales_text=predicted_sales_text, selected_mall=selected_mall, error=error,
                                   prediction_method=prediction_method,specific_year=specific_year, specific_month=specific_month)

        elif prediction_method == "num_months":

            # Filter the data based on the selected mall
            df_selected_mall = df[df['shopping_mall'] == selected_mall]

            df_selected_mall.reset_index(inplace=True)

            # Step 1: Prepare the Data
            df_selected_mall['invoice_date'] = pd.to_datetime(df_selected_mall['invoice_date'])
            df_selected_mall.set_index('invoice_date', inplace=True)
            df_monthly = df_selected_mall['sales'].resample('M').sum()

            # Step 2: Build the Time Series Model
            train_data = df_monthly[:-3]
            test_data = df_monthly[-3:]

            # Use AutoARIMA to automatically select the best ARIMA model
            model = auto_arima(train_data, seasonal=False, trace=True)

            num_months = int(num_months_input)
            forecast = model.predict(n_periods=num_months)



            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.date_range(start=df_monthly.index[-1], periods=num_months, freq='M'),
                                     y=forecast, mode='lines', name='Predicted'))
            fig.update_traces(hovertemplate='Month: %{x}<br>Predicted Sales: %{y}')
            fig.update_layout(title='Monthly Sales Prediction', xaxis_title='Date', yaxis_title='Sales')
            fig.update_layout(showlegend=True)
            plot_html = fig.to_html(full_html=False)
            predicted_sales_text = f"Predicted Sales for {selected_mall}:\n"
            for month, sales in zip(pd.date_range(start=df_monthly.index[-1], periods=num_months, freq='M'), forecast):
                predicted_sales_text += f"{month.strftime('%B %Y')}: {sales}\n"


        return render_template('sales_prediction.html', plot_html=plot_html, num_months=num_months_input,
                               menu_active=menu_active, predictive_active=predictive_active,
                               predicted_sales_text=predicted_sales_text, selected_mall=selected_mall,error=error,prediction_method=prediction_method)
    else:
        num_months = 0
        return render_template('sales_prediction.html', menu_active=menu_active, num_months=num_months,
                               predictive_active=predictive_active)


@app.route('/association_rules')
def association_rules():
    # Create a new column 'gender_age' by combining 'gender' and 'age' as a single categorical variable
    df['gender_age'] = df['gender'] + '_' + df['age'].astype(str)

    # Select the relevant columns for association rule mining
    transactions = df.groupby('gender_age')['category'].apply(list).tolist()

    # Perform association rule mining and convert the generator to a list
    results = list(apriori(transactions, min_support=0.3, min_confidence=0.5))

    # Process each RelationRecord individually
    processed_results = []
    for relation_record in results:
        for ordered_statistic in relation_record.ordered_statistics:
            antecedent = [item for item in ordered_statistic.items_base]
            consequents = [item for item in ordered_statistic.items_add]
            support = relation_record.support
            confidence = ordered_statistic.confidence
            lift = ordered_statistic.lift

            if antecedent and len(consequents) > 1:  # Exclude rules with blank antecedents and single consequents
                processed_results.append((antecedent, consequents[:10]))

    menu_active = 'predictive'
    predictive_active = 'ARM'

    return render_template('association_rules.html', results=processed_results, menu_active=menu_active,
                           predictive_active=predictive_active)



@app.route('/sales-age-prediction', methods=['GET', 'POST'])
def sales_age_prediction():
    menu_active = 'predictive'
    predictive_active = 'predict_age_range'

    df = pd.read_csv('customer_shopping_data.csv')

    # Select the relevant features for prediction
    features = ['age', 'quantity', 'price', 'category']

    # Extract the features and target variable
    X = df[features[:-1]]
    y = df['price']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the numerical features by standardizing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    if request.method == 'POST':
        # Retrieve the age range from the form
        age_range = request.form['age_range']

        if age_range.isdigit():
            error_message = 'You have entered an incorrect age range'
            return render_template('predict_age_range.html', menu_active=menu_active,
                                   predictive_active=predictive_active, error_message=error_message)
        else:
            # Validate and extract lower and upper age bounds
            age_lower, age_upper = map(int, age_range.split('-'))

            # Filter data based on age range
            filtered_data = df[(df['age'] >= age_lower) & (df['age'] <= age_upper) & (df['age'] != -1)]

            # Check if filtered data is empty
            if filtered_data.empty:
                message = "No data available for the specified age range."
                return render_template('predict_age_range.html', menu_active=menu_active, predictive_active=predictive_active,
                                       age_range=age_range, message=message)

            # Extract input features for prediction
            new_data = filtered_data[features[:-1]]

            # Standardize the input features
            new_data_scaled = scaler.transform(new_data)

            # Predict customer spending
            predicted_spending = model.predict(new_data_scaled)

            # Calculate average predicted spending using the filtered data
            average_spending = predicted_spending.mean()

            # Create a DataFrame with predicted spending per category
            predicted_spending_df = pd.DataFrame(
                {'category': filtered_data['category'], 'predicted_spending': predicted_spending})

            # Calculate average predicted spending using the filtered data
            average_predicted_spending = predicted_spending.mean()

            # Calculate predicted spending per category
            predicted_spending_per_category = predicted_spending_df.groupby('category')[
                'predicted_spending'].mean().to_dict()

            return render_template('predict_age_range.html', menu_active=menu_active,
                                   predictive_active=predictive_active, average_spending=average_spending,
                                   average_predicted_spending=average_predicted_spending,
                                   predicted_spending_per_category=predicted_spending_per_category,
                                   age_range=age_range)

    else:
        age_range = ''
        return render_template('predict_age_range.html', menu_active=menu_active, predictive_active=predictive_active,
                               age_range=age_range)




@app.route('/cluster_analysis', methods=['GET', 'POST'])
def cluster_analysis():
    menu_active = 'predictive'
    predictive_active = 'clustering'

    # Read the CSV file
    df = pd.read_csv('customer_shopping_data.csv')

    # Select the relevant features for clustering
    numeric_features = ['quantity', 'price']
    categorical_features = ['category']

    # Preprocess the numerical features by standardizing
    scaler = StandardScaler()
    numeric_data_scaled = scaler.fit_transform(df[numeric_features])

    # Perform label encoding for categorical features
    label_encoder = LabelEncoder()
    categorical_data = df[categorical_features].apply(label_encoder.fit_transform)

    # Combine the preprocessed numerical and categorical data
    data_scaled = np.hstack((numeric_data_scaled, categorical_data))

    # Perform K-means clustering
    k = 4 # Number of clusters (you can adjust this as needed)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)

    # Assign cluster labels to the data points
    cluster_labels = kmeans.labels_

    # Add the cluster labels to the original dataset
    df['cluster'] = cluster_labels

    # Analyze the distribution of data points across clusters
    cluster_counts = df['cluster'].value_counts().to_dict()

    # Analyze cluster characteristics
    cluster_stats = df.groupby('cluster').agg({'category': lambda x: x.unique()[0], 'quantity': 'mean', 'price': 'mean'}).to_dict(orient='index')

    if request.method == 'POST':
        selected_value = request.form.get('select_cluster')
        cluster_type = selected_value
        return render_template('cluster_analysis.html', cluster_counts=cluster_counts,
                               cluster_stats=cluster_stats, cluster_type=cluster_type,
                               menu_active=menu_active, predictive_active=predictive_active)
    else:
        cluster_type = "1"
        return render_template('cluster_analysis.html', cluster_counts=cluster_counts,
                               cluster_stats=cluster_stats, cluster_type=cluster_type,
                               menu_active=menu_active, predictive_active=predictive_active)


if __name__ == '__main__':
    matplotlib.use('Agg')

    mpl.style.use(['ggplot'])
    df = pd.read_csv('customer_shopping_data.csv')
    # Let's create new column for each transaction of sales

    df['sales'] = df['quantity'] * df['price']  # Create a sales column
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)
    # Let's create column for each month,year, and day of week
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['day'] = df['invoice_date'].dt.day
    df['weekday'] = df['invoice_date'].dt.weekday

    dw_mapping = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }

    df['dayofweek'] = df['invoice_date'].dt.weekday.map(dw_mapping)

    app.run(port=3000, debug=True)
