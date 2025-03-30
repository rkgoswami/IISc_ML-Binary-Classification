import json
from tabulate import tabulate

def console_report(results):
    """Prints a formatted table to the console dynamically based on available keys."""

    # Define all possible headers
    possible_headers = {
        "model_name": "Model",
        "val_accuracy": "Validation Accuracy",
        "test_accuracy": "Test Accuracy",
        "best_search_type": "Best Search Method",
        "best_params": "Best Params",
        "grid_test_accuracy": "Grid Test Accuracy",
        "bayesian_test_accuracy": "Bayesian Test Accuracy",
        "grid_best_score": "Grid Best Score",
        "grid_best_params": "Grid Best Params",
        "bayesian_best_score": "Bayesian Best Score",
        "bayesian_best_params": "Bayesian Best Params"
    }

    # Determine available keys dynamically
    available_keys = {key: name for key, name in possible_headers.items() if any(key in res for res in results)}

    # Extract headers and corresponding data dynamically
    headers = list(available_keys.values())
    table_data = []

    for res in results:
        row = []
        for key in available_keys.keys():
            if key in res:
                value = res[key]
                if isinstance(value, float):
                    row.append(round(value, 6))
                elif isinstance(value, dict):
                    row.append(json.dumps(value, indent=2))
                else:
                    row.append(value)
            else:
                row.append("N/A")  # Default for missing keys
        table_data.append(row)

    # Print table
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))