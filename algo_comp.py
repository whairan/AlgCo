import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score

class RecommendationComparisonTool:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data = None
        self.test_data = None
        self.baseline_model = None
        self.production_model = None
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset from the provided path.
        Assumes the dataset has columns 'user_id', 'item_id', and 'rating'.
        """
        data = pd.read_csv(self.data_path)
        self.train_data, self.test_data = train_test_split(data, test_size=0.2)
    
    def load_baseline_model(self):
        """
        Build a simple collaborative filtering baseline model using matrix factorization.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=10000, output_dim=64),  # User embedding
            tf.keras.layers.Embedding(input_dim=10000, output_dim=64),  # Item embedding
            tf.keras.layers.Dot(axes=-1)  # Dot product of embeddings
        ])
        self.baseline_model = model
    
    def load_production_model(self):
        """
        Build a neural collaborative filtering production model.
        """
        user_input = tf.keras.layers.Input(shape=(1,))
        item_input = tf.keras.layers.Input(shape=(1,))
        
        user_embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=64)(user_input)
        item_embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=64)(item_input)
        
        dot_product = tf.keras.layers.Dot(axes=-1)([user_embedding, item_embedding])
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)
        
        self.production_model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=output)
    
    def train_model(self, model, epochs=10, batch_size=32):
        """
        Train the given model on the training data.
        """
        if model == self.baseline_model:
            model.compile(optimizer='adam', loss='mse')
            model.fit(self.train_data[['user_id', 'item_id']], self.train_data['rating'], 
                      epochs=epochs, batch_size=batch_size)
        elif model == self.production_model:
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit([self.train_data['user_id'], self.train_data['item_id']], self.train_data['rating'], 
                      epochs=epochs, batch_size=batch_size)
    
    def evaluate_model(self, model):
        """
        Evaluate the given model on the test dataset using MSE, Precision, and Recall.
        """
        predictions = model.predict(self.test_data[['user_id', 'item_id']])
        actuals = self.test_data['rating']
        
        mse = mean_squared_error(actuals, predictions)
        precision = precision_score(actuals, (predictions > 0.5).astype(int), zero_division=0)
        recall = recall_score(actuals, (predictions > 0.5).astype(int), zero_division=0)
        
        return {"mse": mse, "precision": precision, "recall": recall}
    
    def compare_models(self):
        """
        Compare baseline and production models and display metrics side-by-side.
        """
        baseline_results = self.evaluate_model(self.baseline_model)
        production_results = self.evaluate_model(self.production_model)
        
        comparison_df = pd.DataFrame({
            "Metric": ["MSE", "Precision", "Recall"],
            "Baseline": [baseline_results["mse"], baseline_results["precision"], baseline_results["recall"]],
            "Production": [production_results["mse"], production_results["precision"], production_results["recall"]]
        })
        print(comparison_df)

# Usage example:
# tool = RecommendationComparisonTool("path_to_dataset.csv")
# tool.load_and_preprocess_data()
# tool.load_baseline_model()
# tool.load_production_model()
# tool.train_model(tool.baseline_model)
# tool.train_model(tool.production_model)
# tool.compare_models()
