from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchvision.transforms as T
import os 
from PIL import Image
import numpy as np

class datasetManager:
    def __init__(self, dataset : int = 1, batch_size:int = 32, num_workers:int = 1, transform : T.Compose = None):
        self.dataset = dataset
        assert self.dataset in [0,1,2], "Dataset not found"

        self.batch_size = batch_size
        self.num_workers = num_workers
        #on ajoute le to Tensor pour les images
        if transform is not None:
            transform = T.Compose([transform, T.ToTensor()])
        else:
            transform = T.ToTensor()
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
        
    
    def get_sample_by_class(self, n_samples=5, split='all', retinopathy_class=None, edema_class=None, return_labels=False, rawImage=False, retrun_id=False):
        assert split in ['train', 'test','all'], "Split must be 'train' or 'test'"
        df = self.dfTrain if split == 'train' else self.dfTest
        img_dir = self.train_img_dir if split == 'train' else self.test_img_dir
        if split == 'all':
            df = pd.concat([self.dfTrain, self.dfTest], ignore_index=True)
            indexTrain = [0]*len(self.dfTrain)
            indexTest = [1]*len(self.dfTest)
            df["split"] = indexTrain + indexTest
            img_dir = "dataset"
        
        filtered_df = df.copy()
        if retinopathy_class is not None:
            filtered_df = filtered_df[filtered_df['retinopathy_grade'] == retinopathy_class]
        if edema_class is not None:
            filtered_df = filtered_df[filtered_df['risk_of_macular_edema'] == edema_class]
            
        if len(filtered_df) == 0:
            Warning("No samples found for the given class")
            return []
            
        sample_rows = filtered_df.sample(min(n_samples, len(filtered_df)))
        sample_rows = sample_rows.sample(frac=1)
        
        images = []
        retinopathy_labels = []
        edema_labels = []
        
        for _, row in sample_rows.iterrows():
            if split == 'all':
                img_dir = "dataset/train/images" if row["split"] == 0 else "dataset/test/images"
            img_path = os.path.join(img_dir, (row['id'] + ".jpg"))
            image = Image.open(img_path).convert("RGB")
            retinopathy_grade = "Severe" if row['retinopathy_grade'] == 1 else "Mild"
            edema_risk = "High" if row['risk_of_macular_edema'] == 1 else "Low"
            
            if not rawImage and self.transform:
                image = self.transform(image)
            elif rawImage:
                image = np.array(image)
            
            images.append(image)
            
            if return_labels:
                retinopathy_labels.append(retinopathy_grade)
                edema_labels.append(edema_risk)
        
        if not rawImage:
            images = torch.stack(images)
        if return_labels and not retrun_id:
            result = (images, retinopathy_labels, edema_labels)
        else:
            result = images
        if return_labels and retrun_id:
            result = (images, retinopathy_labels, edema_labels, sample_rows["id"].values)
        
        return result
    
    def get_random_samples(self, n_samples=5,split='all'):
        return self.get_sample_by_class(split=split, n_samples=n_samples)

    def get_ground_segmentation(self, img_id, apply_transform =True):
        img_dir = "dataset/test/segmentation"
        img_path = os.path.join(img_dir, (img_id + ".tif"))
        # TypeError: Invalid shape (3, 256, 256) for image data
        image = Image.open(img_path)
        if apply_transform :
            image = self.transform(image)
        else :
            image = T.Compose([T.ToTensor()])(image)
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        return image
    
    

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

        if self.encoding: 
            labels = torch.zeros(2)
            #labels[row[self.label_cols]] = 1
            index_for_label = int(row.loc[self.label_cols].iloc[0])
            labels[index_for_label] = 1
        else:
            labels = torch.tensor(row[self.label_cols].values.astype('float32'))
        return image, labels
    





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((224, 224)),
    ])

    dm = datasetManager(dataset=0, batch_size=4, transform=transform)

    train_loader = dm.get_dataloader(split='train')
    test_loader = dm.get_dataloader(split='test')

    images, labels = next(iter(train_loader))

    # plt.figure(figsize=(12, 6))
    # for i in range(min(4, len(images))):
    #     img = images[i].permute(1, 2, 0).numpy()
        
    #     retinopathy = "Severe" if labels[i][0] == 1 else "Mild"
    #     edema_risk = "High" if labels[i][1] == 1 else "Low"
        
    #     plt.subplot(1, 4, i+1)
    #     plt.imshow(img)
    #     plt.title(f"Retinopathy: {retinopathy}\nEdema Risk: {edema_risk}")
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    print(f"Train dataset size: {len(dm.dfTrain)}")
    print(f"Test dataset size: {len(dm.dfTest)}")
    print(f"Batch shape: {images.shape}")
    print("\nTesting get_sample_by_class function:")

    # Get severe retinopathy samples with labels
    # samples = dm.get_sample_by_class(n_samples=2, retinopathy_class=1, return_labels=True)
    # images, retinopathy_labels, edema_labels = samples
    
    # plt.figure(figsize=(10, 5))
    # for i in range(len(images)):
    #     plt.subplot(1, 2, i+1)
    #     plt.imshow(images[i].permute(1, 2, 0).numpy())
    #     plt.title(f"Retinopathy: {retinopathy_labels[i]}\nEdema Risk: {edema_labels[i]}")
    #     plt.axis('off')
    # plt.suptitle("Samples with Severe Retinopathy")
    # plt.tight_layout()
    # plt.show()

    # Get high edema risk samples with labels
    # samples = dm.get_sample_by_class(n_samples=2, edema_class=1, return_labels=True)
    # images, retinopathy_labels, edema_labels = samples
    
    # plt.figure(figsize=(10, 5))
    # for i in range(len(images)):
    #     plt.subplot(1, 2, i+1)
    #     plt.imshow(images[i].permute(1, 2, 0).numpy())
    #     plt.title(f"Retinopathy: {retinopathy_labels[i]}\nEdema Risk: {edema_labels[i]}")
    #     plt.axis('off')
    # plt.suptitle("Samples with High Edema Risk")
    # plt.tight_layout()
    # plt.show()

    # print("\nTesting get_random_samples function:")
    # random_samples = dm.get_random_samples(n_samples=3)
    # plt.figure(figsize=(15, 5))
    # for i in range(random_samples.shape[0]):
    #     plt.subplot(1, 3, i+1)
    #     plt.imshow(random_samples[i].permute(1, 2, 0).numpy())
    #     plt.title(f"Random Sample {i+1}")
    #     plt.axis('off')
    # plt.suptitle("Random Samples")
    # plt.tight_layout()
    # plt.show()

    #on récupère une segmentation ground truth4
    random_samples = dm.get_sample_by_class(n_samples=1, split='test', retinopathy_class=1, return_labels=True, rawImage=True, retrun_id=True)
    print(random_samples)
    ground_truth_mask = dm.get_ground_segmentation(random_samples[-1][0])
    unique_labels = np.unique(ground_truth_mask.reshape(-1, ground_truth_mask.shape[2]), axis=0)
    print(len(unique_labels))
    plt.figure(figsize=(15, 5))
    
    plt.imshow(ground_truth_mask)
    plt.title(f"Segmentation ground truth")
    plt.tight_layout()
    plt.show()