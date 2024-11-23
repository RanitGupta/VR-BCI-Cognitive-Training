from helpers import extract_eeg, preprocess, prepare_data, run_model, evaluate_model, compute_empirical_chance
from pathlib import Path
import os

# Set PARAMETRS HERE
MODEL = 'lda'                       # classifier ('svm', 'rf', 'lda', 'xgb')
PCA, N_COMPONENTS = True, 20        # PCA downsampling
TIME_DOWNSAMPLE_FACTOR = 4          # Time downsampling factor
N_FOLDS = 4                         # num. folds for K_folds


def main():

    # Data Split A
    train_pth_A = Path(r"data/train_A")              # ses1 
    test_pth_A_s2 = Path(r"data/test_A_ses2")        # ses2
    test_pth_A_s3 = Path(r"data/test_A_ses3")        # ses3

    # Data Split B 
    train_pth_B = Path(r"data/train_B")          # ses1 + ses2
    test_pth_B = Path(r"data/test_B")            # ses3

    # train data
    all_go_epochs_train_A, all_nogo_epochs_train_A = extract_eeg(train_pth_A)
    print(all_go_epochs_train_A)
    return
    all_go_epochs_train_B, all_nogo_epochs_train_B = extract_eeg(train_pth_B)
 
    # test data
    all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2 = extract_eeg(test_pth_A_s2)
    all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3 = extract_eeg(test_pth_A_s3)
    all_go_epochs_test_B, all_nogo_epochs_test_B = extract_eeg(test_pth_B)

    print("\n\n")
    print(" Step 1 Data Preprocessing ... ")
    all_go_epochs_train_A, all_nogo_epochs_train_A = preprocess(all_go_epochs_train_A, all_nogo_epochs_train_A)
    all_go_epochs_train_B, all_nogo_epochs_train_B = preprocess(all_go_epochs_train_B, all_nogo_epochs_train_B)

    all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2 = preprocess(all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2)
    all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3 = preprocess(all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3)
    all_go_epochs_test_B, all_nogo_epochs_test_B = preprocess(all_go_epochs_test_B, all_nogo_epochs_test_B)

    print("\n\n")
    print("Step 2: Prepare dataset ...")
    X_train_A, y_train_A, X_test_A_s2, y_test_A_s2 = prepare_data(all_go_epochs_train_A, all_nogo_epochs_train_A, all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2, 
                                                                  time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    
    _, _, X_test_A_s3, y_test_A_s3 = prepare_data(all_go_epochs_train_A, all_nogo_epochs_train_A, all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3, 
                                                                  time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)

    X_train_B, y_train_B, X_test_B, y_test_B = prepare_data(all_go_epochs_train_B, all_nogo_epochs_train_B, all_go_epochs_test_B, all_nogo_epochs_test_B, 
                                                                  time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)

    print("\n\n")
    print(" Step 3 Train the model ... ")

    ''' Train and Evaluate Model '''

    # Initialize and train the model
    # train split A
    print("\n\n ----- Train A -----")
    model_A, avg_accuracy_A, classification_reports_A, confusion_matrices_A = run_model(X_train_A, y_train_A, mdl=MODEL, n_splits=N_FOLDS) 


    # train split B
    print("\n\n ----- Train B -----")
    model_B, avg_accuracy_B, classification_reports_B, confusion_matrices_B = run_model(X_train_B, y_train_B, mdl=MODEL, n_splits=N_FOLDS)

    # Evaluate the model
    print("\n\n ----- Train: Session 1 & Test: Session 2 -----")
    y_pred_A_s2 = model_A.predict(X_test_A_s2)
    metrics_A_s2 = evaluate_model(y_test_A_s2, y_pred_A_s2)
    # empirical_chance_mean_A_s2, empirical_chance_std_A_s2 = compute_empirical_chance(X_test_A_s2, y_test_A_s2, model_A, n_iterations=100)

    print("\n\n ----- Train: Session 1 & Test: Session 3 -----")
    y_pred_A_s3 = model_A.predict(X_test_A_s3)
    metrics_A_s3 = evaluate_model(y_test_A_s3, y_pred_A_s3)
    # empirical_chance_mean_A_s3, empirical_chance_std_A_s3 = compute_empirical_chance(X_test_A_s3, y_test_A_s3, model_A, n_iterations=100)

    print("\n\n ----- Train: Session 1 & 2 & Test: Session 3 -----")
    y_pred_B = model_B.predict(X_test_B)
    metrics_B = evaluate_model(y_test_B, y_pred_B)
    # empirical_chance_mean_B, empirical_chance_std_B = compute_empirical_chance(X_test_B, y_test_B, model_A, n_iterations=100)

    # write results
    with open(f"outputs/{MODEL}_{N_COMPONENTS}.txt", 'w') as f:
        f.write(f"Model Type: {MODEL}\n")
        f.write(f"PCA: {PCA}\n")
        N_channels = N_COMPONENTS if PCA else "ALL"
        f.write(f"  N_Channels: {N_channels}\n")
        f.write(f"Time Downsample Factor: {TIME_DOWNSAMPLE_FACTOR}\n")
        f.write(f"Num. Folds for K-Folds: {N_FOLDS}\n\n\n")

        f.write("Results for Training on S1 and Test on S2\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_A * 100:.2f}%\n")
        f.write(f"Test Accuracy: {metrics_A_s2['accuracy'] * 100:.2f}%\n")
        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_A_s2['conf_matrix']}\n")
        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_A_s2['class_report']}\n\n\n")
        # f.write(f"Empirical Chance Level: {empirical_chance_mean_A_s2:.3f} ± {empirical_chance_std_A_s2:.3f}\n\n\n")


        f.write("Results for Training on S1 and Test on S3\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_A * 100:.2f}%\n")
        f.write(f"Test Accuracy: {metrics_A_s3['accuracy'] * 100:.2f}%\n")
        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_A_s3['conf_matrix']}\n")
        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_A_s3['class_report']}\n\n\n")
        # f.write(f"Empirical Chance Level: {empirical_chance_mean_A_s3:.3f} ± {empirical_chance_std_A_s3:.3f}\n\n\n")


        f.write("Results for Training on S1 + S2 and Test on S3\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_B * 100:.2f}%\n")
        f.write(f"Test Accuracy: {metrics_B['accuracy'] * 100:.2f}%\n")
        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_B['conf_matrix']}\n")
        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_B['class_report']}\n\n\n")
        # f.write(f"Empirical Chance Level: {empirical_chance_mean_B:.3f} ± {empirical_chance_std_B:.3f}\n\n\n")


if __name__ == "__main__":
    if not os.path.exists("outputs"):
        os.makedirs("outputs")  

    main()
