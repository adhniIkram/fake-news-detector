# Fake News Detector
#### Video Demo:  <[URL HERE](https://youtu.be/vVMusAIrXwo)>
#### Description: A machine laerning project to determine whether News articles are true our fake.


Fake News Detector (NLP & Flask Web App) - Final Project
Project Overview
Hey there! This project is my final submission for CS50, where I built an end-to-end solution for classifying news articles as either REAL or FAKE. I started with raw data, built a highly accurate machine learning model, and then wrapped the entire prediction engine into a live, functional web application using Python's Flask framework. This whole process required connecting multiple components—data cleaning logic, the trained model, and the frontend—into one seamless experience.

The core goal was not just to build a model, but to demonstrate proficiency in the complete data science pipeline, from feature engineering and model persistence to API development and user interface design.

Model & Performance Summary
The project utilizes a standard but robust NLP classification approach:

Classifier: Logistic Regression (LR) (Scikit-learn).

Feature Engineering: TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency). This step was crucial, as it converts the words in an article into numerical representations that the Logistic Regression model can understand, weighting them by their importance and rarity across the entire dataset.

Data Processing: Before vectorization, all text underwent rigorous cleaning, including conversion to lowercase, removal of special characters, stop word removal, and POS-tagged Lemmatization to reduce words to their base form. This specific cleaning step was essential for reducing noise and improving model accuracy.

Final Performance
The model achieved an impressive performance score, demonstrating strong generalization capability:

Final Accuracy: ≈98.68% on the held-out test set.

Key Metrics: The high accuracy was backed up by balanced Precision and Recall scores across both classes.

Confusion Matrix: The low number of False Positives (94) and False Negatives (54) further confirmed the model's reliability.

Application Architecture (How It Works)
The entire application runs as a cohesive three-tier system, stitched together by Python and JavaScript:

Model Persistence (models/): After training in the Jupyter Notebook (train_model.ipynb), the trained LR model and the fitted TF-IDF Vectorizer are saved to the models/ directory using Python's pickle library. This allows the Flask app to load the "brain" of the detector instantly without needing to retrain.

Backend API (app.py): The main Flask application handles the server logic.

It loads the pickle files on startup.

It defines the /analyze route which accepts the user's text via an AJAX POST request.

It applies the processText function (imported from src/) to the new input, transforms it using the loaded vectorizer, and runs the prediction.

It returns the result and confidence score as a JSON response.

Frontend UI (templates/index.html): The user interface is a clean, single-page application built with HTML, enhanced with custom CSS for clear visual communication (Green for REAL, Red for FAKE).

It uses the JavaScript fetch API (AJAX) to send the user's pasted text to the /analyze endpoint in the background, ensuring a smooth, non-reloading user experience.

Repository Structure Explained
The project follows a standard structure to separate data, notebooks, and production code:

app.py: The heart of the web application. Contains the Flask server setup and the prediction logic for the API.

.gitignore: Ensures large, unnecessary files (like the venv environment, temporary files, and data/ files) are not tracked by Git.

requirements.txt / environment.yml: Lists all required Python dependencies for local setup and deployment.

data/: Stores the initial datasets (Fake.csv, True.csv, etc.) used for model training.

models/: Contains the serialized machine learning assets (lr_model.pkl, tfidf_vectorizer.pkl) ready for production use.

notebooks/: Contains the Jupyter Notebooks (clean_text.ipynb, train_model.ipynb) used for exploratory data analysis, feature engineering development, and model training.

src/: Stores the reusable Python code, specifically the core text cleaning function (processText.py), which is critical for consistent preprocessing across training and production environments.

templates/: Stores the front-end files (index.html) used by Flask to render the user interface.

🛑 Limitations and Biases
While the model achieved high accuracy, it’s important to acknowledge its limitations:

Objective Truth vs. Pattern Recognition: The model does not understand true facts objectively. It only identifies linguistic patterns it learned from the training data. For example, a deeply researched article that uses slightly unusual phrasing might be flagged as "FAKE" if its linguistic structure deviates too far from the "REAL" training patterns.

Short Text Sensitivity: The model often defaults to FAKE for extremely short input (e.g., a few words). This happens because short text results in very few features for the TF-IDF vectorizer, and the sparse vector is statistically more likely to resemble the low-feature distribution of some "FAKE" headlines in the training data.

Dataset Bias: The model is inherently limited by the data it was trained on. If the original dataset contained biases (e.g., if "REAL" articles consistently came from a specific political viewpoint or publication type), the model will perpetuate that bias, potentially penalizing legitimate news from underrepresented sources. The biases within the source data are difficult to fully account for and address completely.