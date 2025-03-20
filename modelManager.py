import os
import torch
import gc
import torchvision.models as models
from torchvision.models import VGG16_Weights, ResNet50_Weights, Swin_V2_B_Weights
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch import nn
import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, recall_score
import pandas as pd

class ModelManager:
    def __init__(self, modelType, numClass, modelWeights=None):
        assert modelType in ['vgg16','resnet50','swinT'], "Model type not supported"
        self.modelType = modelType
        self.modelWeights = modelWeights
        if modelWeights is not None:
            assert os.path.exists(modelWeights), "Model weights file not found"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.numClass = numClass
        self.model = self.createModel()


    def createModel(self):
        if self.modelType == 'vgg16':
            model = models.vgg16(weights=VGG16_Weights.DEFAULT)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, self.numClass)
        elif self.modelType == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.numClass)
        elif self.modelType == 'swinT':
            model = models.swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, self.numClass)
        
        if self.modelWeights is not None:
            model.load_state_dict(torch.load(self.modelWeights, map_location=self.device, weights_only=True))
        model = model.to(self.device)
        self.model = model
        return model
    
    def train(self, train_loader, valid_loader=None,optimizer = torch.optim.Adam, criterion = nn.BCEWithLogitsLoss(),num_epochs=10, lr=3e-5, verbose=True, accuracy=True,saveModel=True,modelName=None):
        optimizer = optimizer(self.model.parameters(), lr=lr)
        print(f"Training {self.modelType} model")
        for epoch in range(num_epochs):
            self.model.train()
            for images, labels in tqdm(train_loader, total=len(train_loader)):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print("Evaluation of Epoch", epoch+1)
            if valid_loader is not None:
                self.validate(valid_loader, criterion)
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item()}", f"Train Accuracy: {round(self.accuracy(train_loader)*100,2)}" if accuracy else "")
                if valid_loader is not None:
                    print(f"Validation Loss: {self.validate(valid_loader, criterion)}", f"Validation Accuracy: {round(self.accuracy(valid_loader)*100,2)}" if accuracy else "")
        if saveModel:
            if modelName is not None:
                torch.save(self.model.state_dict(), f"{modelName}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{num_epochs}.pth")
            else:
                torch.save(self.model.state_dict(), f"{self.modelType}_model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_{num_epochs}.pth")           
    
    def validate(self, valid_loader, criterion=nn.BCEWithLogitsLoss()):
        self.model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
        return loss.item()
    
    def accuracy(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                if self.numClass == 1 or (self.numClass == 2 and labels.shape[1] == 1):
                    # Binary classification single output
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += labels.size(0)
                elif self.numClass == 2 and labels.shape[1] == 2:
                    # Binary two outputs
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += labels.size(0) * labels.size(1)
                else:
                    # Multi-class
                    _, predicted = torch.max(outputs.data, 1)
                    if labels.dim() > 1:  
                        _, labels = torch.max(labels.data, 1)
                    total += labels.size(0)
                
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0

    def accuracy(self, loader, f1_recall=False):
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                # Since it's a binary classification task with a single output:
                # Apply a sigmoid, then threshold at 0.5
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                # Collect predictions/labels in CPU lists
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to NumPy arrays
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        # Accuracy (simple mean over correct predictions)
        accuracy = (all_preds == all_labels).mean()
        
        if f1_recall:
            # F1 score and recall
            f1 = f1_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')

            return accuracy, f1, recall
        else :
            return accuracy

    def show_model_parameters(self):
        print(self.model)


    def inference(self, image, transform=None, returnData='probs', metricsStyle = False):
        assert returnData in ['probs', 'logits'], "returnData must be 'probs' or 'logits'"
        self.model.eval()
        
        # Check if image is numpy array and convert to tensor if needed
        if isinstance(image, np.ndarray):
            # Handle numpy arrays - check dimensions
            if image.ndim == 3:  # Single image with channels
                image = torch.from_numpy(image).float()
                # Ensure channels first format (C,H,W)
                if image.shape[2] in [1, 3, 4]:  # If channels are last
                    image = image.permute(2, 0, 1)
                image = image.unsqueeze(0)  # Add batch dimension
            elif image.ndim == 4:  # Batch of images
                image = torch.from_numpy(image).float()
                # Ensure channels first format (B,C,H,W)
                if image.shape[3] in [1, 3, 4]:  # If channels are last
                    image = image.permute(0, 3, 1, 2)
        else:
            if image.dim() == 3:
                image = image.unsqueeze(0)
                
        if transform is not None:
            image = transform(image)
        
        image = image.to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probs = torch.sigmoid(logits)

        if metricsStyle:
            return probs
        
        if returnData == 'logits':
            return logits.cpu().numpy()
        else:
            return probs.cpu().numpy()

    def __del__(self):
        if hasattr(self, 'model'):
            if self.model is not None:
                try:
                    self.model = self.model.to('cpu')
                    del self.model
                except:
                    pass
            gc.collect()
            if torch.cuda is not None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()



if __name__ == '__main__':
    from datasetManager import datasetManager

    # modelManager = ModelManager('vgg16', 2)
    # modelManager.show_model_parameters()
    # modelManager = ModelManager('resnet50', 2)
    # modelManager.show_model_parameters()
    # modelManager = ModelManager('swinT', 2)
    # modelManager.show_model_parameters()

    datasetManagerOb = datasetManager(dataset=1, batch_size=8, num_workers=4, transform=T.Compose([T.Resize((224, 224))]))
    train_loader = datasetManagerOb.get_dataloader(split='train')
    test_loader = datasetManagerOb.get_dataloader(split='test')
    numClass = datasetManagerOb.numClass


    # modelManager = ModelManager('vgg16', numClass,"vgg16_model_2025-03-06_12-56_3.pth")
    # print(modelManager.accuracy(test_loader))
    # print(modelManager.inference(datasetManagerOb.get_random_samples(1)[0][0]))
    
    
    # for modelType in ['vgg16', 'resnet50', 'swinT']:
    #     modelManager = ModelManager(modelType, numClass)
    #     modelManager.train(train_loader, test_loader, num_epochs=3)
    #     print(modelManager.accuracy(test_loader))
    #     del modelManager

    Lmodels = [('swinT_model_2025-03-06_13-35_3.pth',"swinT"),('resnet50_model_2025-03-06_13-31_3.pth','resnet50'),('vgg16_model_2025-03-06_13-28_3.pth','vgg16')]
    #on fait marcher l'inférence sur les 3 modèles
    scoring_df = []
    for model in Lmodels:
        modelManager = ModelManager(model[1], numClass,model[0])
        #print(modelManager.accuracy(test_loader, f1_recall=True))
        acc, f1, recall = modelManager.accuracy(test_loader, f1_recall=True)
        scoring_df.append((model[1], acc, f1, recall))
        #samples = datasetManagerOb.get_random_samples(3)
        #print(modelManager.inference(samples))
        del modelManager

    scoring_df = pd.DataFrame(scoring_df, columns=['Model', 'Accuracy', 'F1 score', 'Recall score'])
    print(scoring_df)