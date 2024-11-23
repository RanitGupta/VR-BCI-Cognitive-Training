from helpers import extract_eeg, preprocess, prepare_data, run_model, evaluate_model, compute_empirical_chance
from pathlib import Path
import os

# Set PARAMETRS HERE

# GENERAL DECODER
MODEL = 'lda'                       # classifier ('svm', 'rf', 'lda', 'xgb')
PCA, N_COMPONENTS = True, 20        # PCA downsampling
TIME_DOWNSAMPLE_FACTOR = 4          # Time downsampling factor
N_FOLDS = 4                         # num. folds for K_folds


# INDIVIDUAL DECODERS
# INDIV_MODEL = 'lda'                       # classifier ('svm', 'rf', 'lda', 'xgb')
# INDIV_PCA, INDIV_N_COMPONENTS = True, 20        # PCA downsampling
# INDIV_TIME_DOWNSAMPLE_FACTOR = 4          # Time downsampling factor
# N_FOLDS = 4                         # num. folds for K_folds


def main():

    # Data Split A
    train_pth_A = Path(r"data/train_A")                    # ses1 
    train_pth_A_jh = Path(r"data/train_A_jh")              # ses1 
    train_pth_A_rn = Path(r"data/train_A_rn")              # ses1 
    train_pth_A_tr = Path(r"data/train_A_tr")              # ses1 

    test_pth_A_s2 = Path(r"data/test_A_ses2")              # ses2
    test_pth_A_s2_jh = Path(r"data/test_A_ses2_jh")        # ses2
    test_pth_A_s2_rn = Path(r"data/test_A_ses2_rn")        # ses2
    test_pth_A_s2_tr = Path(r"data/test_A_ses2_tr")        # ses2
    
    test_pth_A_s3 = Path(r"data/test_A_ses3")              # ses3
    test_pth_A_s3_jh = Path(r"data/test_A_ses3_jh")        # ses3
    test_pth_A_s3_rn = Path(r"data/test_A_ses3_rn")        # ses3
    test_pth_A_s3_tr = Path(r"data/test_A_ses3_tr")        # ses3

    # Data Split B 
    train_pth_B = Path(r"data/train_B")                # ses1 + ses2
    train_pth_B_jh = Path(r"data/train_B_jh")          # ses1 + ses2
    train_pth_B_rn = Path(r"data/train_B_rn")          # ses1 + ses2
    train_pth_B_tr = Path(r"data/train_B_tr")          # ses1 + ses2

    test_pth_B = Path(r"data/test_B")                  # ses3
    test_pth_B_jh = Path(r"data/test_B_jh")            # ses3
    test_pth_B_rn = Path(r"data/test_B_rn")               # ses3
    test_pth_B_tr = Path(r"data/test_B_tr")               # ses3

    # train data load (use .fif if exists else create new)
    all_go_epochs_train_A, all_nogo_epochs_train_A = extract_eeg(folder_name=train_pth_A, fif_name=Path(r"loaded_data/train_A.fif"))
    all_go_epochs_train_A_jh, all_nogo_epochs_train_A_jh = extract_eeg(folder_name=train_pth_A_jh, fif_name=Path(r"loaded_data/train_A_jh.fif"))
    all_go_epochs_train_A_rn, all_nogo_epochs_train_A_rn = extract_eeg(folder_name=train_pth_A_rn, fif_name=Path(r"loaded_data/train_A_rn.fif"))
    all_go_epochs_train_A_tr, all_nogo_epochs_train_A_tr = extract_eeg(folder_name=train_pth_A_tr, fif_name=Path(r"loaded_data/train_A_tr.fif"))

    all_go_epochs_train_B, all_nogo_epochs_train_B = extract_eeg(folder_name=train_pth_B, fif_name=Path(r"loaded_data/train_B.fif"))
    all_go_epochs_train_B_jh, all_nogo_epochs_train_B_jh = extract_eeg(folder_name=train_pth_B_jh, fif_name=Path(r"loaded_data/train_B_jh.fif"))
    all_go_epochs_train_B_rn, all_nogo_epochs_train_B_rn = extract_eeg(folder_name=train_pth_B_rn, fif_name=Path(r"loaded_data/train_B_rn.fif"))
    all_go_epochs_train_B_tr, all_nogo_epochs_train_B_tr = extract_eeg(folder_name=train_pth_B_tr, fif_name=Path(r"loaded_data/train_B_tr.fif"))
 
    # test data
    all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2 = extract_eeg(folder_name=test_pth_A_s2, fif_name=Path(r"loaded_data/test_A_s2.fif"))
    all_go_epochs_test_A_s2_jh, all_nogo_epochs_test_A_s2_jh = extract_eeg(folder_name=test_pth_A_s2_jh, fif_name=Path(r"loaded_data/test_A_s2_jh.fif"))
    all_go_epochs_test_A_s2_rn, all_nogo_epochs_test_A_s2_rn = extract_eeg(folder_name=test_pth_A_s2_rn, fif_name=Path(r"loaded_data/test_A_s2_rn.fif"))
    all_go_epochs_test_A_s2_tr, all_nogo_epochs_test_A_s2_tr = extract_eeg(folder_name=test_pth_A_s2_tr, fif_name=Path(r"loaded_data/test_A_s2_tr.fif"))

    all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3 = extract_eeg(folder_name=test_pth_A_s3, fif_name=Path(r"loaded_data/test_A_s3.fif"))
    all_go_epochs_test_A_s3_jh, all_nogo_epochs_test_A_s3_jh = extract_eeg(folder_name=test_pth_A_s3_jh, fif_name=Path(r"loaded_data/test_A_s3_jh.fif"))
    all_go_epochs_test_A_s3_rn, all_nogo_epochs_test_A_s3_rn = extract_eeg(folder_name=test_pth_A_s3_rn, fif_name=Path(r"loaded_data/test_A_s3_rn.fif"))
    all_go_epochs_test_A_s3_tr, all_nogo_epochs_test_A_s3_tr = extract_eeg(folder_name=test_pth_A_s3_tr, fif_name=Path(r"loaded_data/test_A_s3_tr.fif"))

    all_go_epochs_test_B, all_nogo_epochs_test_B = extract_eeg(folder_name=test_pth_B, fif_name=Path(r"loaded_data/test_B.fif"))
    all_go_epochs_test_B_jh, all_nogo_epochs_test_B_jh = extract_eeg(folder_name=test_pth_B_jh, fif_name=Path(r"loaded_data/test_B_jh.fif"))
    all_go_epochs_test_B_rn, all_nogo_epochs_test_B_rn = extract_eeg(folder_name=test_pth_B_rn, fif_name=Path(r"loaded_data/test_B_rn.fif"))
    all_go_epochs_test_B_tr, all_nogo_epochs_test_B_tr = extract_eeg(folder_name=test_pth_B_tr, fif_name=Path(r"loaded_data/test_B_tr.fif"))

    print("\n\n")
    print(" Step 1 Data Preprocessing ... ")
    all_go_epochs_train_A, all_nogo_epochs_train_A = preprocess(all_go_epochs_train_A, all_nogo_epochs_train_A)
    all_go_epochs_train_A_jh, all_nogo_epochs_train_A_jh = preprocess(all_go_epochs_train_A_jh, all_nogo_epochs_train_A_jh)
    all_go_epochs_train_A_rn, all_nogo_epochs_train_A_rn = preprocess(all_go_epochs_train_A_rn, all_nogo_epochs_train_A_rn)
    all_go_epochs_train_A_tr, all_nogo_epochs_train_A_tr = preprocess(all_go_epochs_train_A_tr, all_nogo_epochs_train_A_tr)

    all_go_epochs_train_B, all_nogo_epochs_train_B = preprocess(all_go_epochs_train_B, all_nogo_epochs_train_B)
    all_go_epochs_train_B_jh, all_nogo_epochs_train_B_jh = preprocess(all_go_epochs_train_B_jh, all_nogo_epochs_train_B_jh)
    all_go_epochs_train_B_rn, all_nogo_epochs_train_B_rn = preprocess(all_go_epochs_train_B_rn, all_nogo_epochs_train_B_rn)
    all_go_epochs_train_B_tr, all_nogo_epochs_train_B_tr = preprocess(all_go_epochs_train_B_tr, all_nogo_epochs_train_B_tr)

    all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2 = preprocess(all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2)
    all_go_epochs_test_A_s2_jh, all_nogo_epochs_test_A_s2_jh = preprocess(all_go_epochs_test_A_s2_jh, all_nogo_epochs_test_A_s2_jh)
    all_go_epochs_test_A_s2_rn, all_nogo_epochs_test_A_s2_rn = preprocess(all_go_epochs_test_A_s2_rn, all_nogo_epochs_test_A_s2_rn)
    all_go_epochs_test_A_s2_tr, all_nogo_epochs_test_A_s2_tr = preprocess(all_go_epochs_test_A_s2_tr, all_nogo_epochs_test_A_s2_tr)

    all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3 = preprocess(all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3)
    all_go_epochs_test_A_s3_jh, all_nogo_epochs_test_A_s3_jh = preprocess(all_go_epochs_test_A_s3_jh, all_nogo_epochs_test_A_s3_jh)
    all_go_epochs_test_A_s3_rn, all_nogo_epochs_test_A_s3_rn = preprocess(all_go_epochs_test_A_s3_rn, all_nogo_epochs_test_A_s3_rn)
    all_go_epochs_test_A_s3_tr, all_nogo_epochs_test_A_s3_tr = preprocess(all_go_epochs_test_A_s3_tr, all_nogo_epochs_test_A_s3_tr)

    all_go_epochs_test_B, all_nogo_epochs_test_B = preprocess(all_go_epochs_test_B, all_nogo_epochs_test_B)
    all_go_epochs_test_B_jh, all_nogo_epochs_test_B_jh = preprocess(all_go_epochs_test_B_jh, all_nogo_epochs_test_B_jh)
    all_go_epochs_test_B_rn, all_nogo_epochs_test_B_rn = preprocess(all_go_epochs_test_B_rn, all_nogo_epochs_test_B_rn)
    all_go_epochs_test_B_tr, all_nogo_epochs_test_B_tr = preprocess(all_go_epochs_test_B_tr, all_nogo_epochs_test_B_tr)


    print("\n\n")
    print("Step 2: Prepare dataset ...")
    X_train_A, y_train_A, X_test_A_s2, y_test_A_s2 = prepare_data(all_go_epochs_train_A, all_nogo_epochs_train_A, all_go_epochs_test_A_s2, all_nogo_epochs_test_A_s2, 
                                                                  time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    X_train_A_jh, y_train_A_jh, X_test_A_s2_jh, y_test_A_s2_jh = prepare_data(all_go_epochs_train_A_jh, all_nogo_epochs_train_A_jh, all_go_epochs_test_A_s2_jh, all_nogo_epochs_test_A_s2_jh, 
                                                                             time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    X_train_A_rn, y_train_A_rn, X_test_A_s2_rn, y_test_A_s2_rn = prepare_data(all_go_epochs_train_A_rn, all_nogo_epochs_train_A_rn, all_go_epochs_test_A_s2_rn, all_nogo_epochs_test_A_s2_rn, 
                                                                             time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    X_train_A_tr, y_train_A_tr, X_test_A_s2_tr, y_test_A_s2_tr = prepare_data(all_go_epochs_train_A_tr, all_nogo_epochs_train_A_tr, all_go_epochs_test_A_s2_tr, all_nogo_epochs_test_A_s2_tr, 
                                                                             time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)

    _, _, X_test_A_s3, y_test_A_s3 = prepare_data(all_go_epochs_train_A, all_nogo_epochs_train_A, all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3, 
                                                  time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    _, _, X_test_A_s3_jh, y_test_A_s3_jh = prepare_data(all_go_epochs_train_A_jh, all_nogo_epochs_train_A_jh, all_go_epochs_test_A_s3_jh, all_nogo_epochs_test_A_s3_jh, 
                                                        time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    _, _, X_test_A_s3_rn, y_test_A_s3_rn = prepare_data(all_go_epochs_train_A_rn, all_nogo_epochs_train_A_rn, all_go_epochs_test_A_s3_rn, all_nogo_epochs_test_A_s3_rn, 
                                                        time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    _, _, X_test_A_s3_tr, y_test_A_s3_tr = prepare_data(all_go_epochs_train_A_tr, all_nogo_epochs_train_A_tr, all_go_epochs_test_A_s3_tr, all_nogo_epochs_test_A_s3_tr, 
                                                        time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)

    X_train_B, y_train_B, X_test_B, y_test_B = prepare_data(all_go_epochs_train_B, all_nogo_epochs_train_B, all_go_epochs_test_B, all_nogo_epochs_test_B, 
                                                            time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    X_train_B_jh, y_train_B_jh, X_test_B_jh, y_test_B_jh = prepare_data(all_go_epochs_train_B_jh, all_nogo_epochs_train_B_jh, all_go_epochs_test_B_jh, all_nogo_epochs_test_B_jh, 
                                                                        time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    X_train_B_rn, y_train_B_rn, X_test_B_rn, y_test_B_rn = prepare_data(all_go_epochs_train_B_rn, all_nogo_epochs_train_B_rn, all_go_epochs_test_B_rn, all_nogo_epochs_test_B_rn, 
                                                                        time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)
    X_train_B_tr, y_train_B_tr, X_test_B_tr, y_test_B_tr = prepare_data(all_go_epochs_train_B_tr, all_nogo_epochs_train_B_tr, all_go_epochs_test_B_tr, all_nogo_epochs_test_B_tr, 
                                                                        time_ds_factor=TIME_DOWNSAMPLE_FACTOR, use_pca=PCA, n_components=N_COMPONENTS)


    print("\n\n")
    print(" Step 3 Train the model ... ")

    ''' Train and Evaluate Model '''

        # Initialize and train the model
    # train split A
    print("\n\n ----- Train A -----")
    model_A, avg_accuracy_A, classification_reports_A, confusion_matrices_A = run_model(X_train_A, y_train_A, mdl=MODEL, n_splits=N_FOLDS)
    model_A_jh, avg_accuracy_A_jh, classification_reports_A_jh, confusion_matrices_A_jh = run_model(X_train_A_jh, y_train_A_jh, mdl=MODEL, n_splits=N_FOLDS)
    model_A_rn, avg_accuracy_A_rn, classification_reports_A_rn, confusion_matrices_A_rn = run_model(X_train_A_rn, y_train_A_rn, mdl=MODEL, n_splits=N_FOLDS)
    model_A_tr, avg_accuracy_A_tr, classification_reports_A_tr, confusion_matrices_A_tr = run_model(X_train_A_tr, y_train_A_tr, mdl=MODEL, n_splits=N_FOLDS)

    # train split B
    print("\n\n ----- Train B -----")
    model_B, avg_accuracy_B, classification_reports_B, confusion_matrices_B = run_model(X_train_B, y_train_B, mdl=MODEL, n_splits=N_FOLDS)
    model_B_jh, avg_accuracy_B_jh, classification_reports_B_jh, confusion_matrices_B_jh = run_model(X_train_B_jh, y_train_B_jh, mdl=MODEL, n_splits=N_FOLDS)
    model_B_rn, avg_accuracy_B_rn, classification_reports_B_rn, confusion_matrices_B_rn = run_model(X_train_B_rn, y_train_B_rn, mdl=MODEL, n_splits=N_FOLDS)
    model_B_tr, avg_accuracy_B_tr, classification_reports_B_tr, confusion_matrices_B_tr = run_model(X_train_B_tr, y_train_B_tr, mdl=MODEL, n_splits=N_FOLDS)

    # Evaluate the model
    print("\n\n ----- Train: Session 1 & Test: Session 2 -----")
    y_pred_A_s2 = model_A.predict(X_test_A_s2)
    metrics_A_s2 = evaluate_model(y_test_A_s2, y_pred_A_s2)
    y_pred_A_s2_jh = model_A_jh.predict(X_test_A_s2_jh)
    metrics_A_s2_jh = evaluate_model(y_test_A_s2_jh, y_pred_A_s2_jh)
    y_pred_A_s2_rn = model_A_rn.predict(X_test_A_s2_rn)
    metrics_A_s2_rn = evaluate_model(y_test_A_s2_rn, y_pred_A_s2_rn)
    y_pred_A_s2_tr = model_A_tr.predict(X_test_A_s2_tr)
    metrics_A_s2_tr = evaluate_model(y_test_A_s2_tr, y_pred_A_s2_tr)

    print("\n\n ----- Train: Session 1 & Test: Session 3 -----")
    y_pred_A_s3 = model_A.predict(X_test_A_s3)
    metrics_A_s3 = evaluate_model(y_test_A_s3, y_pred_A_s3)
    y_pred_A_s3_jh = model_A_jh.predict(X_test_A_s3_jh)
    metrics_A_s3_jh = evaluate_model(y_test_A_s3_jh, y_pred_A_s3_jh)
    y_pred_A_s3_rn = model_A_rn.predict(X_test_A_s3_rn)
    metrics_A_s3_rn = evaluate_model(y_test_A_s3_rn, y_pred_A_s3_rn)
    y_pred_A_s3_tr = model_A_tr.predict(X_test_A_s3_tr)
    metrics_A_s3_tr = evaluate_model(y_test_A_s3_tr, y_pred_A_s3_tr)

    print("\n\n ----- Train: Session 1 & 2 & Test: Session 3 -----")
    y_pred_B = model_B.predict(X_test_B)
    metrics_B = evaluate_model(y_test_B, y_pred_B)
    y_pred_B_jh = model_B_jh.predict(X_test_B_jh)
    metrics_B_jh = evaluate_model(y_test_B_jh, y_pred_B_jh)
    y_pred_B_rn = model_B_rn.predict(X_test_B_rn)
    metrics_B_rn = evaluate_model(y_test_B_rn, y_pred_B_rn)
    y_pred_B_tr = model_B_tr.predict(X_test_B_tr)
    metrics_B_tr = evaluate_model(y_test_B_tr, y_pred_B_tr)


    # write results
    with open(f"outputs/{MODEL}_{N_COMPONENTS}.txt", 'w') as f:
        f.write("GENERAL MODEL (ALL SUBJECTS)\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Decoder: {MODEL}\n")
        f.write(f"PCA: {PCA}\n")
        N_channels = N_COMPONENTS if PCA else "ALL"
        f.write(f"  N_Channels: {N_channels}\n")
        f.write(f"Time Downsample Factor: {TIME_DOWNSAMPLE_FACTOR}\n")
        f.write(f"Num. Folds for K-Folds: {N_FOLDS}\n\n")

        f.write("INDIVIDUAL MODEL\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Decoder: {MODEL}\n")
        f.write(f"PCA: {PCA}\n")
        N_channels = N_COMPONENTS if PCA else "ALL"
        f.write(f"  N_Channels: {N_channels}\n")
        f.write(f"Time Downsample Factor: {TIME_DOWNSAMPLE_FACTOR}\n")
        f.write(f"Num. Folds for K-Folds: {N_FOLDS}\n\n\n")

        f.write("Results for Training on S1 and Test on S2\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_A * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT JH: {avg_accuracy_A_jh * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT RN: {avg_accuracy_A_rn * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT TR: {avg_accuracy_A_tr * 100:.2f}%\n")

        f.write(f"Test Accuracy: {metrics_A_s2['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT JH: {metrics_A_s2_jh['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT RN: {metrics_A_s2_rn['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT TR: {metrics_A_s2_tr['accuracy'] * 100:.2f}%\n")
       
        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_A_s2['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT JH:\n")
        f.write(f"  {metrics_A_s2_jh['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT RN:\n")
        f.write(f"  {metrics_A_s2_rn['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT TR:\n")
        f.write(f"  {metrics_A_s2_tr['conf_matrix']}\n")

        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_A_s2['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT JH:\n")
        f.write(f"  {metrics_A_s2_jh['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT RN:\n")
        f.write(f"  {metrics_A_s2_rn['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT TR:\n")
        f.write(f"  {metrics_A_s2_tr['class_report']}\n\n\n")
        # f.write(f"Empirical Chance Level: {empirical_chance_mean_A_s2:.3f} ± {empirical_chance_std_A_s2:.3f}\n\n\n")


        f.write("Results for Training on S1 and Test on S3\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_A * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT JH: {avg_accuracy_A_jh * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT RN: {avg_accuracy_A_rn * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT TR: {avg_accuracy_A_tr * 100:.2f}%\n")

        f.write(f"Test Accuracy: {metrics_A_s3['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT JH: {metrics_A_s3_jh['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT RN: {metrics_A_s3_rn['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT TR: {metrics_A_s3_tr['accuracy'] * 100:.2f}%\n")

        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_A_s3['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT JH:\n")
        f.write(f"  {metrics_A_s3_jh['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT RN:\n")
        f.write(f"  {metrics_A_s3_rn['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT TR:\n")
        f.write(f"  {metrics_A_s3_tr['conf_matrix']}\n")

        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_A_s3['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT JH:\n")
        f.write(f"  {metrics_A_s3_jh['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT RN:\n")
        f.write(f"  {metrics_A_s3_rn['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT TR:\n")
        f.write(f"  {metrics_A_s3_tr['class_report']}\n\n\n")
        # f.write(f"Empirical Chance Level: {empirical_chance_mean_A_s3:.3f} ± {empirical_chance_std_A_s3:.3f}\n\n\n")


        f.write("Results for Training on S1 + S2 and Test on S3\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_B * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT JH: {avg_accuracy_B_jh * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT RN: {avg_accuracy_B_rn * 100:.2f}%\n")
        f.write(f"  Avg. Fold Train Accuracy SUBJECT TR: {avg_accuracy_B_tr * 100:.2f}%\n")

        f.write(f"Test Accuracy: {metrics_B['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT JH: {metrics_B_jh['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT RN: {metrics_B_rn['accuracy'] * 100:.2f}%\n")
        f.write(f"  Test Accuracy SUBJECT TR: {metrics_B_tr['accuracy'] * 100:.2f}%\n")

        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_B['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT JH:\n")
        f.write(f"  {metrics_B_jh['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT RN:\n")
        f.write(f"  {metrics_B_rn['conf_matrix']}\n")
        f.write("\n Test Confusion Matrix SUBJECT TR:\n")
        f.write(f"  {metrics_B_tr['conf_matrix']}\n")

        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_B['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT JH:\n")
        f.write(f"  {metrics_B_jh['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT RN:\n")
        f.write(f"  {metrics_B_rn['class_report']}\n")
        f.write("\n Test Classification Report SUBJECT TR:\n")
        f.write(f"  {metrics_B_tr['class_report']}")
        # f.write(f"Empirical Chance Level: {empirical_chance_mean_B:.3f} ± {empirical_chance_std_B:.3f}\n\n\n")



if __name__ == "__main__":
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if not os.path.exists("loaded_data"):
        os.makedirs("loaded_data")   

    main()
