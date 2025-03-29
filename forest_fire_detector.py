#Importing the Libraries
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import time
from pprint import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification")
    parser.add_argument("--data-dir", type=str, default="Dataset", help="Path to the dataset folder")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--model-save-path", type=str, default="best_model.pth", help="Path to save the best model")
    parser.add_argument("--retrain", action="store_true", help="Flag to retrain the model")
    return parser.parse_args()

# Custom dataset class that inherits from torchvision.datasets.ImageFolder
class CustomDataset(datasets.ImageFolder):
    def __init__(self, data_dir, mean=None, std=None, *args, **kwargs):
        transform_list = [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        if mean is not None and std is not None:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        transform = transforms.Compose(transform_list)
        super().__init__(data_dir, transform=transform, *args, **kwargs)

    @staticmethod
    # Function to compute the mean and standard deviation of the dataset
    def calculate_mean_std(dataset, batch_size=64):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        mean = 0.0
        var = 0.0
        num_samples = 0

        progress_bar = tqdm(dataloader, desc="Analyzing the Dataset")

        for images, _ in progress_bar:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            var += images.var(2).sum(0)
            num_samples += batch_samples

        mean /= num_samples
        var /= num_samples
        std = torch.sqrt(var)
        return mean, std

# Image classifier class to handle training, validation, and testing
class ImageClassifier:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    # Function to train the model for one epoch
    def train_epoch(self, train_loader, epoch, num_epochs, device):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": running_loss / (i + 1)})

    # Function to validate the model
    def validate(self, validation_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    # Function to test the model
    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    # Function to calculate metrics and plot the ROC curve
    def calculate_metrics_and_plot_roc(self, test_loader):
        self.model.eval()

        y_true = []
        y_pred = []
        y_score = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_score.extend(probabilities[:, 1].cpu().numpy())

        accuracy = np.round(accuracy_score(y_true, y_pred), 3)
        precision = np.round(precision_score(y_true, y_pred), 3)
        recall = np.round(recall_score(y_true, y_pred), 3)
        f1 = np.round(f1_score(y_true, y_pred), 3)
        conf_matrix = confusion_matrix(y_true, y_pred)
        auc = np.round(roc_auc_score(y_true, y_score), 3)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"AUC: {auc}")

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

#Function to train model for 'num_epochs' epochs
def train_model(classifier, train_loader, validation_loader, num_epochs, model_save_path):
    best_val_accuracy = 0
    early_stop_threshold = 99.5
    for epoch in range(num_epochs):
        classifier.train_epoch(train_loader, epoch, num_epochs, device)

        # Validating the model
        val_accuracy = classifier.validate(validation_loader)
        print(f'Accuracy on validation set for epoch {epoch + 1}: {np.round(val_accuracy, 3)}%')

        # Saving the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(classifier.model)
            torch.save(best_model.state_dict(), model_save_path)
            print("Model updated")

            # Checking if the validation accuracy is greater than the early stop threshold
            if best_val_accuracy >= early_stop_threshold:
                print(f"Early stopping at epoch {epoch + 1} as the validation accuracy reached {early_stop_threshold}%")
                break
    return best_model

def main(args):
    data_dir = args.data_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    retrain = args.retrain
    model_save_path = args.model_save_path

    # Loading dataset without normalization to compute mean and std
    unnormalized_dataset = CustomDataset(data_dir)
    mean, std = CustomDataset.calculate_mean_std(unnormalized_dataset)
    print(f"Mean: {mean}, Standard Deviation: {std}")

    # Loading normalized dataset
    dataset = CustomDataset(data_dir, mean, std)

    # Splitting the dataset into training, validation, and testing sets
    train_size = int(0.6 * len(dataset))  # 60% of the dataset for training
    validation_size = int(0.3 * len(dataset))  # 30% of the dataset for validation
    test_size = len(dataset) - train_size - validation_size  # 10% of the dataset for testing
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # if not retraining test everything
    if (not retrain):
        test_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Loading the pre-trained model (Transfer Learning)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)

    # Initializing the image classifier
    best_val_accuracy = 0
    best_model = None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    classifier = ImageClassifier(model, criterion, optimizer)

    if os.path.exists(model_save_path) and not retrain:
        user_input = ''
        while user_input not in ('yes', 'no'):
            user_input = input(f"{model_save_path} exists. Do you want to retrain the model? (yes/no, default: \033[1myes\033[0m): ").lower()
            if user_input == '':
                user_input = 'yes'
            if user_input not in ('yes', 'no', ''):
                print("Invalid input. Please enter 'yes', 'no', or press Enter for the default option.")
        
        if user_input == 'yes' or user_input == '':
            retrain = True
        elif user_input == 'no':
            retrain = False

    if os.path.exists(model_save_path) and not retrain:
        print(f"Loading the existing model from '{model_save_path}'.")
        best_model = models.resnet50(pretrained=True)
        num_features = best_model.fc.in_features
        best_model.fc = nn.Linear(num_features, 2)
        best_model.load_state_dict(torch.load(model_save_path))
        best_model = best_model.to(device)
    else:
        tic = time.perf_counter()
        best_model = train_model(classifier, train_loader, validation_loader, num_epochs, model_save_path)
        toc = time.perf_counter()
        print(f"Trained model in n {toc - tic:0.0f} seconds")

    # Testing the best model
    best_classifier = ImageClassifier(best_model, criterion, optimizer)
    test_accuracy = best_classifier.test(test_loader)
    print(f'Accuracy on test set: {np.round(test_accuracy, 3)}%')

    #pprint(vars(test_loader.dataset))

    # Calculate metrics and plot ROC curve
    best_classifier.calculate_metrics_and_plot_roc(test_loader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
