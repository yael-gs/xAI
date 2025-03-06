from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchvision.transforms as T
import os 
from PIL import Image


class datasetManager:
    def __init__(self, dataset : int = 1, batch_size:int = 32, num_workers:int = 1, transform : T.Compose = None):
        self.dataset = dataset
        assert self.dataset in [0,1,2], "Dataset not found"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
        if self.dataset in [0,1,2]:
            if self.dataset == 0:
                self.label_cols = ["retinopathy_grade", "risk_of_macular_edema"]
            elif self.dataset == 1:
                self.label_cols = ["retinopathy_grade"]
            elif self.dataset == 2:
                self.label_cols = ["risk_of_macular_edema"]
            self.dfTrain = pd.read_csv("dataset/train/labels.csv")
            # print(self.dfTrain["retinopathy_grade"].value_counts())
            # print(self.dfTrain["risk_of_macular_edema"].value_counts())
            threshold_retinopathy = 1
            self.dfTrain.loc[self.dfTrain["retinopathy_grade"] <= threshold_retinopathy, "retinopathy_grade"] = 0
            self.dfTrain.loc[self.dfTrain["retinopathy_grade"] > threshold_retinopathy, "retinopathy_grade"] = 1
            
            self.dfTrain.loc[self.dfTrain["risk_of_macular_edema"] < 1, "risk_of_macular_edema"] = 0
            self.dfTrain.loc[self.dfTrain["risk_of_macular_edema"] >= 1, "risk_of_macular_edema"] = 1
            
            self.dfTrain["retinopathy_grade"] = self.dfTrain["retinopathy_grade"].astype("int8")
            self.dfTrain["risk_of_macular_edema"] = self.dfTrain["risk_of_macular_edema"].astype("int8")
            
            self.train_img_dir = "dataset/train/images"
            
            self.dfTest = pd.read_csv("dataset/test/labels.csv")
            self.dfTest.loc[self.dfTest["retinopathy_grade"] <= threshold_retinopathy, "retinopathy_grade"] = 0
            self.dfTest.loc[self.dfTest["retinopathy_grade"] > threshold_retinopathy, "retinopathy_grade"] = 1
            
            self.dfTest.loc[self.dfTest["risk_of_macular_edema"] < 1, "risk_of_macular_edema"] = 0
            self.dfTest.loc[self.dfTest["risk_of_macular_edema"] >= 1, "risk_of_macular_edema"] = 1
            
            self.dfTest["retinopathy_grade"] = self.dfTest["retinopathy_grade"].astype("int8")
            self.dfTest["risk_of_macular_edema"] = self.dfTest["risk_of_macular_edema"].astype("int8")
            
            self.test_img_dir = "dataset/test/images"
            self.numClass = 2
            # print(self.dfTrain["retinopathy_grade"].value_counts())
            # print(self.dfTrain["risk_of_macular_edema"].value_counts())
    
    def get_dataloader(self, split='train',shuffle=None):
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        dataset = RetinopathyDataset(
            self.dfTrain if split == 'train' else self.dfTest,
            self.train_img_dir if split == 'train' else self.test_img_dir,
            self.label_cols, 
            self.transform,
            encoding=False if self.dataset == 0 else True
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle if shuffle is not None else (split == 'train'), 
            num_workers=self.num_workers
        )
        
    
    def get_sample_by_class(self, n_samples=5, split='train', retinopathy_class=None, edema_class=None, return_original=False,return_labels=False):
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        df = self.dfTrain if split == 'train' else self.dfTest
        img_dir = self.train_img_dir if split == 'train' else self.test_img_dir
        
        filtered_df = df.copy()
        if retinopathy_class is not None:
            filtered_df = filtered_df[filtered_df['retinopathy_grade'] == retinopathy_class]
        if edema_class is not None:
            filtered_df = filtered_df[filtered_df['risk_of_macular_edema'] == edema_class]
            
        if len(filtered_df) == 0:
            return []
            
        sample_rows = filtered_df.sample(min(n_samples, len(filtered_df)))
        
        result = []
        for _, row in sample_rows.iterrows():
            img_path = os.path.join(img_dir, (row['id'] + ".jpg"))
            image = Image.open(img_path).convert("RGB")
            originalImage = image.copy()
            # Get labels
            retinopathy_grade = "Severe" if row['retinopathy_grade'] == 1 else "Mild"
            edema_risk = "High" if row['risk_of_macular_edema'] == 1 else "Low"
            
            # Process image for model if needed
            model_image = image
            if self.transform:
                model_image = self.transform(model_image)
            model_image = T.ToTensor()(model_image)
            
            # Prepare return values based on parameters
            if return_original and return_labels:
                result.append((model_image, originalImage, retinopathy_grade, edema_risk))
            elif return_original and not return_labels:
                result.append((model_image, originalImage))
            elif return_labels and not return_original:
                result.append((model_image, retinopathy_grade, edema_risk))
            else:
                result.append(model_image)
                
        if not return_labels and not return_original:
            result = torch.stack(result)
        return result
    
    def get_random_samples(self, n_samples=5,split='train'):
        return self.get_sample_by_class(split=split, n_samples=n_samples)
    


class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, label_cols, transform=None,encoding=False):
        self.df = df
        self.encoding = encoding
        self.img_dir = img_dir
        self.label_cols = label_cols
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, (row['id'] + ".jpg"))
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        image = T.ToTensor()(image)

        if self.encoding: 
            labels = torch.zeros(2)
            labels[row[self.label_cols]] = 1
        else:
            labels = torch.tensor(row[self.label_cols].values.astype('float32'))
        return image, labels
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((256, 256)),
    ])

    dm = datasetManager(dataset=0, batch_size=4, transform=transform)

    train_loader = dm.get_dataloader(split='train')
    test_loader = dm.get_dataloader(split='test')

    images, labels = next(iter(train_loader))

    plt.figure(figsize=(12, 6))
    for i in range(min(4, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        
        retinopathy = "Severe" if labels[i][0] == 1 else "Mild"
        edema_risk = "High" if labels[i][1] == 1 else "Low"
        
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(f"Retinopathy: {retinopathy}\nEdema Risk: {edema_risk}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Train dataset size: {len(dm.dfTrain)}")
    print(f"Test dataset size: {len(dm.dfTest)}")
    print(f"Batch shape: {images.shape}")

    print("\nTesting get_sample_by_class function:")

    severe_samples = dm.get_sample_by_class(n_samples=2, retinopathy_class=1, return_labels=True)
    plt.figure(figsize=(10, 5))
    for i, (img, retinopathy, edema) in enumerate(severe_samples):
        plt.subplot(1, 2, i+1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"Retinopathy: {retinopathy}\nEdema Risk: {edema}")
        plt.axis('off')
    plt.suptitle("Samples with Severe Retinopathy")
    plt.tight_layout()
    plt.show()

    edema_samples = dm.get_sample_by_class(n_samples=2, edema_class=1, return_labels=True)
    plt.figure(figsize=(10, 5))
    for i, (img, retinopathy, edema) in enumerate(edema_samples):
        plt.subplot(1, 2, i+1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"Retinopathy: {retinopathy}\nEdema Risk: {edema}")
        plt.axis('off')
    plt.suptitle("Samples with High Edema Risk")
    plt.tight_layout()
    plt.show()

    print("\nTesting get_random_samples function:")
    random_samples = dm.get_random_samples(n_samples=3)
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(random_samples):
        plt.subplot(1, 3, i+1)
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"Random Sample {i+1}")
        plt.axis('off')
    plt.suptitle("Random Samples")
    plt.tight_layout()
    plt.show()