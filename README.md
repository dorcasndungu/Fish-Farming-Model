# Fish-Farming-Model: Predicting Fish Growth and Water Quality for Kenyan Fish Farmers

## Overview
The goal of this project is to build a predictive model that helps Kenyan fish farmers improve their livelihoods through optimized fish farming practices. This aligns with **SDG 8 - Decent Work and Economic Growth**, by enabling farmers to enhance fish production. The model predicts fish **length** and **weight** based on water quality parameters such as temperature, turbidity, dissolved oxygen, pH, ammonia, and nitrate levels. Additionally, the model classifies the water quality as **Good, Moderate, or Poor**. The model offers a user-friendly interface for farmers to input real-time data and receive predictions on fish growth and water quality classification.
# Project Proposal - SDG 8: Decent Work and Economic Growth

## 1. SDG Selection
For this project, we focused on **SDG 8 - Decent Work and Economic Growth**, which aims to promote inclusive and sustainable economic growth, employment, and decent work for all. This SDG is crucial for fish farmers, as sustainable aquaculture practices can provide increased income opportunities and improve livelihoods, contributing to economic growth in local communities.

## 2. Specific Problem Addressed
The specific problem addressed in this project is improving **fish farming conditions** by predicting the ideal environmental factors (temperature, dissolved oxygen, pH, turbidity, ammonia, and nitrate) to maximize fish growth. Fish farming plays an important role in food production and income generation, and optimizing farming conditions is key to enhancing productivity.

Our goal is to develop a **machine learning model** that can be used by **automated pond management systems** to recommend actions for maintaining water quality. This can help fish farmers improve fish production, thereby increasing their economic returns and contributing to **decent work** and **economic growth**.

## 3. Planned ML Application
The ML application will use historical environmental data to predict fish growth metrics such as **fish length** and **fish weight** based on water quality parameters. These predictions will assist fish farmers in making data-driven decisions on how to adjust conditions for optimal fish farming. Additionally, the model will classify the water quality into three categories: **Good**, **Moderate**, and **Poor**, providing fish farmers with actionable insights on how to improve water quality and, in turn, fish production.

The model will also be integrated with a **UI interface**, where fish farmers can input water quality parameters manually to receive real-time predictions for fish growth (length and weight) and water quality classification. This allows farmers to interact with the system easily and make informed decisions on-site.

## 4. Overview of ML Techniques
We selected **Random Forest Regressor** models to predict both fish length and fish weight based on various environmental factors. Additionally, a **Multi-Output Random Forest Regressor** model will be used to predict both target variables simultaneously. These models were chosen due to their ability to handle non-linear relationships and provide feature importance insights, which are valuable in identifying the most significant environmental factors for fish growth.

For the water quality classification, we used a **Random Forest Classifier** to categorize the water quality into three levels: **Good**, **Moderate**, and **Poor**. This model provides a simple yet effective way to assist farmers in managing their ponds based on real-time water conditions.

## 5. Tasks Assigned to Members
- **Dorcas Mwihaki**: Conducted initial research on SDG 8 and the specific problem of optimizing fish farming conditions. Coordinated the dataset and model research.
- **Lee Ngari**: Handled data preprocessing, and feature engineering, ensuring the data was clean and ready for model training. Led the evaluation of model performance.

## 6. UI Interface for Manual Predictions
A user-friendly **UI interface** will be developed to allow fish farmers to input water quality parameters manually and receive predictions for both fish growth (length and weight) and water quality classification (Good, Moderate, Poor). This will empower farmers to make quick, informed decisions about their farming practices without needing technical expertise.

![UI Interface Screenshot](insert_screenshot_here)

## Dataset Information
The datasets used in this project are generated from **freshwater aquaponics catfish ponds** equipped with various water quality sensors. The datasets were collected automatically at 5-second intervals from June to mid-October 2021.
The dataset includes the number of fish in the pond, along with the target variables: fish length (cm) and weight (g). All attributes are continuous, and the model uses water quality parameters (temperature, turbidity, dissolved oxygen, pH, ammonia, nitrate) for prediction.



