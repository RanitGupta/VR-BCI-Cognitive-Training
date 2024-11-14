import numpy as np
from helpers import extract_eeg, preprocess, prepare_data


if __name__ == "__main__":
    
    # Load XDF files iteratively from the folder py
    # folder_name = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-jh1/ses-S001/eeg"
    # folder_name = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-tr1/ses-S001/eeg"
    # folder_name = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-rn1/ses-S001/eeg"
    # folder_name = "/home/minsu/Dropbox/Projects/VRCogTraining/Data/sub-yy1/ses-S001/eeg"

    # train data
    folder_name_train = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-jh1/ses-S001/eeg"
    all_go_epochs_train, all_nogo_epochs_train = extract_eeg(folder_name_train)
    
    # test data
    folder_name_test = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-jh1/ses-S002/eeg"
    # folder_name_test = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-tr1/ses-S002/eeg"
    # folder_name_test = "/Users/jhpark/Desktop/VR-BCI-Cognitive-Training/data/sub-rn1/ses-S002/eeg"
    all_go_epochs_test, all_nogo_epochs_test = extract_eeg(folder_name_test)

    print("\n\n")
    print(" Step 1 Data Preprocessing ... ")
    all_go_epochs_train, all_nogo_epochs_train = preprocess(all_go_epochs_train, all_nogo_epochs_train)
    all_go_epochs_test, all_nogo_epochs_test = preprocess(all_go_epochs_test, all_nogo_epochs_test)

    print("\n\n")
    print("Step 2: Prepare dataset ...")
    X_train, y_train, X_test, y_test = prepare_data(all_go_epochs_train, all_nogo_epochs_train, all_go_epochs_test, all_nogo_epochs_test)

    print("\n\n")
    print(" Step 3 Train the model ... ")

    ''' Train and Evaluate SVM Model '''
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Initialize and train the SVM model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

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
