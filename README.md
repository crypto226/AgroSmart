# smart-agro
Smart-Agro Fertilizer Recommendation System

This repository contains a simple Flask-based web application that provides fertilizer recommendations based on input parameters like temperature, humidity, moisture, soil type, crop type, and nutrient levels.

Prerequisites:

1.Python 3.x

2.Flask (pip install flask)

3.Pandas (pip install pandas)


Project Structure:

1.app(1).py: Main Flask application file.

2.recommendation.html: HTML template for the recommendation interface.

3.script.js: JavaScript file for client-side interactions.

4.style.css: CSS file for styling the application.

5.Fertilizer_Prediction_gpt(1).csv: Dataset file containing fertilizer recommendations.



Setup Instructions:

project_folder/

├── app(1).py

├── best_fertilizer_model.h5

├── static(folder)

│   └── script.js

├── templates(folder)

│   └── recommendation.html

|----Fertilizer_Prediction_gpt(1).csv

make sure that the files are arranged like this 

1.Clone the Repository:

If this is a cloned repository, ensure all files are present. Otherwise, download all files to a single directory.

2.Install Dependencies

Run the following command in your terminal or command prompt to install the required Python packages:
pip install flask pandas

3.Prepare the Dataset

Ensure the Fertilizer_Prediction_gpt(1).csv file is in the same directory as app(1).py. This file contains the data used for fertilizer recommendations.


4.Run the Flask Application

Navigate to the directory containing app(1).py in your terminal or command prompt and run:
python app(1).py
This will start the Flask development server. By default, it runs on http://127.0.0.1:5000/.


5.Access the Application

Open your web browser and go to http://127.0.0.1:5000/ to access the recommendation interface.


Usage:

1.Enter the required parameters (Temperature, Humidity, Moisture, Nitrogen, Phosphorus, Potassium, Soil Type, and Crop Type) in the form.

2.Click the "Recommend" button to get a fertilizer recommendation.
  The result will be displayed on the page.
