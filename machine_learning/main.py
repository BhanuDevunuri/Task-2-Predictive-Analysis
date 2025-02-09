from scripts.preprocess import load_and_preprocess
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model

def main():
    # File path to the dataset
    file_path = "data_set/country_wise_latest.csv"
    
    # Step 1: Load and preprocess the data
    data = load_and_preprocess(file_path)
    
    # Step 2: Train the model
    model, scaler, X_test, y_test = train_model(data)
    
    # Step 3: Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
