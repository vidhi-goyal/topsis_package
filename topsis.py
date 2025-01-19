import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, output_file):
    try:
        # Read the input Excel file
        data = pd.read_excel(input_file)
    except Exception as e:
        raise Exception(f"Error reading input file: {e}")

    # Validate the dataset
    if data.shape[1] < 3:
        raise Exception("Dataset must have at least 4 columns: one for names and the others for criteria.")
    if len(weights) != data.shape[1] - 1 or len(impacts) != data.shape[1] - 1:
        raise Exception("Number of weights and impacts must match the number of criteria.")

    # Extract criteria and names
    criteria = data.iloc[:, 1:].values
    names = data.iloc[:, 0]

    # Step 1: Normalize the criteria
    norm_criteria = criteria / np.sqrt((criteria ** 2).sum(axis=0))

    # Step 2: Apply weights
    weights = np.array([float(w) for w in weights])
    weighted_criteria = norm_criteria * weights

    # Step 3: Determine ideal best and ideal worst
    impacts = np.array([1 if imp == "+" else -1 for imp in impacts])
    ideal_best = np.max(weighted_criteria, axis=0) * impacts + np.min(weighted_criteria, axis=0) * (1 - impacts)
    ideal_worst = np.min(weighted_criteria, axis=0) * impacts + np.max(weighted_criteria, axis=0) * (1 - impacts)

    # Step 4: Calculate distances to ideal best and worst
    dist_best = np.sqrt(((weighted_criteria - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_criteria - ideal_worst) ** 2).sum(axis=1))

    # Step 5: Calculate the TOPSIS score
    topsis_score = dist_worst / (dist_best + dist_worst)

    # Step 6: Assign ranks
    ranks = topsis_score.argsort()[::-1] + 1

    # Add the results to the dataset
    data["Topsis Score"] = topsis_score
    data["Rank"] = ranks

    # Save the results to an output Excel file
    data.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

# Main execution block
if name == "main":
    # Define input and output file paths
    input_file = "102203659-data.xlsx"  # Replace with your input Excel file path
    output_file = "102203659-result.xlsx"  # Replace with your desired output file path

    # Define weights and impacts
    weights = [1, 1, 1, 1, 1]  # Adjust weights as needed
    impacts = ["+", "-", "+", "+", "-"]  # Adjust impacts as needed

    # Validate dataset and execute TOPSIS
    data = pd.read_excel(input_file)
    print("Dataset shape:", data.shape)
    print("Columns in the dataset:", data.columns)

    # Check lengths of weights and impacts
    if len(weights) != data.shape[1] - 1 or len(impacts) != data.shape[1] - 1:
        raise Exception("Number of weights and impacts must match the number of criteria.")

    # Run the TOPSIS method
    topsis(input_file, weights, impacts, output_file)