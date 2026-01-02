import numpy as np
import pandas as pd
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate features
area = np.random.normal(1500, 500, n_samples)  # Area in sq ft
bedrooms = np.random.randint(1, 6, n_samples)  # Number of bedrooms
bathrooms = np.random.randint(1, 4, n_samples)  # Number of bathrooms
age = np.random.randint(0, 50, n_samples)  # Age of house in years
garage = np.random.randint(0, 3, n_samples)  # Number of garage spaces
yard_size = np.random.normal(1000, 300, n_samples)  # Yard size in sq ft
distance_to_city = np.random.normal(5, 2, n_samples)  # Distance to city center in miles

# Create a price formula with some randomness
price = (
    100000 +  # Base price
    150 * area +  # Price per sq ft
    25000 * bedrooms +  # Value per bedroom
    35000 * bathrooms +  # Value per bathroom
    (-500) * age +  # Depreciation with age
    15000 * garage +  # Value per garage space
    20 * yard_size +  # Value per sq ft of yard
    (-10000) * distance_to_city +  # Decrease with distance from city
    np.random.normal(0, 50000, n_samples)  # Random variation
)

# Ensure prices are positive and reasonable
price = np.maximum(price, 50000)

# Create a pandas DataFrame
data = pd.DataFrame({
    'area': np.round(area),
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'garage': garage,
    'yard_size': np.round(yard_size),
    'distance_to_city': np.round(distance_to_city, 1),
    'price': np.round(price)
})

# Create 'data' directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Save the dataset to a CSV file
data.to_csv('data/house_prices.csv', index=False)

print(f"Dataset generated with {n_samples} samples and saved to 'data/house_prices.csv'")
print("Dataset preview:")
print(data.head())
print("\nStatistical summary:")
print(data.describe())
