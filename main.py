import warnings

from sklearn.exceptions import ConvergenceWarning

from data_loader import load_and_preprocess_data
from knn import knn
from logistic import logistic_regression
from naive_bayes import naive_bayes
from print_console_report import console_report
from svm_linear import svm_linear
from svm_rbf import svm_rbf

# Suppress skopt duplicate evaluation warning
warnings.filterwarnings("ignore", category=UserWarning, module="skopt.optimizer.optimizer")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


""" This is orchestrator function """
if __name__ == "__main__":

    # Load the preprocessed data once
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    model_functions = [
        naive_bayes,
        logistic_regression,
        knn,
        svm_linear,
        svm_rbf
    ]
    all_comparisons = []
    all_results = []

    for func in model_functions:
        try:
            # Pass the preloaded data to the model
            comparisons, result = func(X_train, X_val, X_test, y_train, y_val, y_test)
            all_results.append(result)
            all_comparisons.append(comparisons)
            print(f"{chr(0x2714)} {func.__name__} completed successfully with test accuracy of {result.get('test_accuracy'):.4f}")
        except Exception as e:
            print(f"{chr(0x2716)} Error in {func.__name__}: {str(e)}")

    if all_results:
        # Print comparison report
        print("\n" + "=" * 125)
        print("HYPERPARAMETER TUNING COMPARISON".center(125))
        print("=" * 125)
        console_report(all_comparisons)
        print("=" * 125)

        # Print final results
        print("\n" + "=" * 125)
        print("FINAL MODEL PERFORMANCE".center(125))
        print("=" * 125)
        console_report(all_results)
        print("=" * 125)
    else:
        print("No results to display")