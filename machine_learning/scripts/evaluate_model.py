from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Visualize results
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Recovery Rates (%)")
    plt.ylabel("Predicted Recovery Rates (%)")
    plt.title("Actual vs Predicted Recovery Rates")
    plt.show()
    
    return mse, r2
