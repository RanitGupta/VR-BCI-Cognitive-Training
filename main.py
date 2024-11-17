import numpy as np
from helpers import extract_eeg, preprocess, prepare_data, run_model, evaluate_model
from pathlib import Path
import os
import time

# Set PARAMETRS HERE
MODEL = 'lda'                       # classifier ('svm', 'rf', 'lda')
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
                                                                  time_ds_factor=4, use_pca=True, n_components=20)
    
    _, _, X_test_A_s3, y_test_A_s3 = prepare_data(all_go_epochs_train_A, all_nogo_epochs_train_A, all_go_epochs_test_A_s3, all_nogo_epochs_test_A_s3, 
                                                                  time_ds_factor=4, use_pca=True, n_components=20)

    X_train_B, y_train_B, X_test_B, y_test_B = prepare_data(all_go_epochs_train_B, all_nogo_epochs_train_B, all_go_epochs_test_B, all_nogo_epochs_test_B, 
                                                                  time_ds_factor=4, use_pca=True, n_components=20)

    print("\n\n")
    print(" Step 3 Train the model ... ")

    ''' Train and Evaluate Model '''

    # Initialize and train the model
    # train split A
    best_model_A, avg_accuracy_A, classification_reports_A, confusion_matrices_A = run_model(X_train_A, y_train_A, mdl=MODEL, n_splits=N_FOLDS) # best so far (with ::4 time downsample and PCA n_components = 20)

    # train split B
    best_model_B, avg_accuracy_B, classification_reports_B, confusion_matrices_B = run_model(X_train_B, y_train_B, mdl=MODEL, n_splits=N_FOLDS) # best so far (with ::4 time downsample and PCA n_components = 20)


    # Evaluate the model
    y_pred_A_s2 = best_model_A.predict(X_test_A_s2)
    metrics_A_s2 = evaluate_model(y_test_A_s2, y_pred_A_s2)

    y_pred_A_s3 = best_model_A.predict(X_test_A_s3)
    metrics_A_s3 = evaluate_model(y_test_A_s3, y_pred_A_s3)

    y_pred_B = best_model_B.predict(X_test_B)
    metrics_B = evaluate_model(y_test_B, y_pred_B)

    # write results
    with open(f"outputs/output{time.time()}.txt", 'w') as f:
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


        f.write("Results for Training on S1 and Test on S3\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_A * 100:.2f}%\n")
        f.write(f"Test Accuracy: {metrics_A_s3['accuracy'] * 100:.2f}%\n")
        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_A_s3['conf_matrix']}\n")
        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_A_s3['class_report']}\n\n\n")


        f.write("Results for Training on S1 + S2 and Test on S3\n")
        f.write("----------------------------------------------------\n")
        f.write(f"Avg. Fold Train Accuracy: {avg_accuracy_B * 100:.2f}%\n")
        f.write(f"Test Accuracy: {metrics_B['accuracy'] * 100:.2f}%\n")
        f.write("\nTest Confusion Matrix:\n")
        f.write(f"{metrics_B['conf_matrix']}\n")
        f.write("\nTest Classification Report:\n")
        f.write(f"{metrics_B['class_report']}")


    # import pdb; pdb.set_trace()

    # # Average the concatenated epochs
    # go_evoked = all_go_epochs.average()
    # nogo_evoked = all_nogo_epochs.average()

    # subj = "jh"
    # # str_png = "no_filter"
    # # str_png = "bandpass"
    # str_png = "bandpass_spatial"

    # # Plot Go ERP
    # print("\nPlot Go ERP")
    # fig_go = go_evoked.plot(picks=['CZ'])
    # fig_go.suptitle('ERP for Event ID 14 Followed by Go', fontsize=16)
    # fig_go.savefig(f'{subj}_go_{str_png}.png', dpi=300)  # Save as PNG with high resolution

    # # Plot NoGo ERP
    # print("\nPlot NoGo ERP")
    # fig_nogo = nogo_evoked.plot(picks=['CZ'])
    # fig_nogo.suptitle('ERP for Event ID 14 Followed by NoGo', fontsize=16)
    # fig_nogo.savefig(f'{subj}_nogo_{str_png}.png', dpi=300)  # Save as PNG with high resolution

    # # Show plots
    # plt.show()


if __name__ == "__main__":
    if not os.path.exists("outputs"):
        os.makedirs("outputs")  

    main()
