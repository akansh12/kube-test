
import warnings
warnings.filterwarnings('ignore')
import torch
from torchvision import transforms, models, datasets
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import cv2
import subprocess
from collections import OrderedDict
# import timm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, auc
# from sklearn.metrics import precision_score,recall_score, f1_score
import glob



# import timm
subprocess.run(["nvidia-smi"])



# import efficientnet_pytorch.EfficientNet

print("You have been assigned an " + str(torch.cuda.get_device_name(0)) +" ! Don't forget to have fun while you explore. :-)")

# In[8]:


from collections import OrderedDict


# In[9]:


import matplotlib.pyplot as plt
from PIL import Image


# In[10]:

labels_csv = {'train': "/scratch/scratch6/akansh12/DeepEXrays/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv",
             'test': "/scratch/scratch6/akansh12/DeepEXrays/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv"
             }

data_dir = {'train': "/scratch/scratch6/akansh12/DeepEXrays/data/data_256/train/",
           'test': "/scratch/scratch6/akansh12/DeepEXrays/data/data_256/test/"}
# In[11]:


local_label = ['Aortic enlargement', 'Atelectasis',
       'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation',
       'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration',
       'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',
       'Nodule/Mass', 'Pleural effusion', 'Pneumothorax',
       'Pulmonary fibrosis', 'Rib fracture', 'Other lesion','Pleural thickening', 'No finding']

# In[12]:

#dataset
#dataset
class Vin_big_dataset(Dataset):
    def __init__(self, image_loc, label_loc, transforms, data_type, selec_radio, radio_id = None):
        local_label = ['Aortic enlargement', 'Atelectasis',
       'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation',
       'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration',
       'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',
       'Nodule/Mass', 'Pleural effusion', 'Pneumothorax',
       'Pulmonary fibrosis', 'Rib fracture', 'Other lesion','Pleural thickening', 'No finding']
        
        if data_type == 'train':
            label_df = pd.read_csv(label_loc)
            if selec_radio == 'rand_one':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames = np.unique(label_df.index.values).tolist()
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
                self.labels = []
                for i in (filenames):
                    self.labels.append(label_df[local_label].loc[i].values.tolist()[np.random.choice([0,1,2])])
                self.labels = torch.tensor(self.labels)
            if selec_radio == 'agree_two':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames_temp = np.unique(label_df.index.values).tolist()
                self.labels = []
                filenames = []
                for i in filenames_temp:
                    a,b = np.unique(label_df.loc[i][local_label].values, axis = 0, return_counts=True)
                    if b[0] >= 2:
                        filenames.append(i)
                        self.labels.append(a[0])
                self.labels = torch.tensor(self.labels)
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
            if selec_radio == 'agree_three':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames_temp = np.unique(label_df.index.values).tolist()
                self.labels = []
                filenames = []
                for i in filenames_temp:
                    a,b = np.unique(label_df.loc[i][local_label].values, axis = 0, return_counts=True)
                    if b[0] == 3:
                        filenames.append(i)
                        self.labels.append(a[0])
                self.labels = torch.tensor(self.labels)
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
            if selec_radio == 'radio_per_epoch':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames = np.unique(label_df.index.values).tolist()
                self.labels = []
                for i in filenames:
                    self.labels.append(label_df.loc[i][local_label].values[radio_id].tolist())
                self.labels = torch.tensor(self.labels)
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
            if selec_radio == 'all': 
                label_df['labels'] = label_df['image_id'] +'_'+ label_df['rad_id']
                label_df.set_index("labels", inplace = True)
                filenames = label_df.index.values.tolist()
            
                self.full_filenames = [os.path.join(image_loc, i.split('_')[0]+'.png') for i in filenames]
                self.labels = []
                for i in tqdm(filenames):
                    self.labels.append(label_df[local_label].loc[i].values.tolist())         
                self.labels = torch.tensor(self.labels)
                
        if data_type == 'test':                     
            filenames = os.listdir(image_loc)
            self.full_filenames = [os.path.join(image_loc, i) for i in filenames]
            label_df = pd.read_csv(label_loc)
            label_df.set_index("image_id", inplace = True)
            self.labels = [label_df[local_label].loc[filename[:-4]].values for filename in filenames]
            
        self.transforms = transforms
#         self.data_type = data_type
    def __len__(self):
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transforms(image)
        
        return image, self.labels[idx]

# In[13]:


data_transforms = { 
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5), 
        transforms.RandomPerspective(distortion_scale=0.3),
        transforms.RandomRotation((-30,30)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]),
    
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])        
    ])
    
}

print("Data Loading...")
# In[14]:

train_data = Vin_big_dataset(image_loc = data_dir['train'],
                          label_loc = labels_csv['train'],
                          transforms = data_transforms['train'],
                          data_type = 'train', selec_radio = 'all')

test_data = Vin_big_dataset(image_loc = data_dir['test'],
                          label_loc = labels_csv['test'],
                          transforms = data_transforms['test'],
                          data_type = 'test', selec_radio = None)

print("Data Loaded")
# In[20]:

trainloader = DataLoader(train_data,batch_size = 8,shuffle = True)
testloader = DataLoader(test_data,batch_size = 8,shuffle = False)

# In[16]:

# model = timm.create_model('efficientnet_b6', pretrained=False)
# model.load_state_dict(torch.load("/storage/home/akansh12/Vin-ChestXR-Abnormality-detection/model/tf_efficientnet_b6_aa-80ba17e4.pth"))

# model.classifier = nn.Sequential(OrderedDict([
#     ('fcl1', nn.Linear(2304,6)),
#     ('out', nn.Sigmoid()),
# ]))


model = models.densenet201(pretrained=False)

state_dict = torch.load("/storage/home/akansh12/Vin-ChestXR-Abnormality-detection/model/imageNet_DenseNet201.pt", map_location = 'cpu')
for keyA, keyB in zip(state_dict, model.state_dict()):
    state_dict = OrderedDict((keyB if k == keyA else k, v) for k, v in state_dict.items())

model.load_state_dict(state_dict)


model.classifier = nn.Sequential(OrderedDict([
    ('fcl1', nn.Linear(1920,22)),
    ('out', nn.Sigmoid()),
]))




device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    model = model.cuda()


# In[21]:


def train_one_epoch(model, optimizer, lr_scheduler,
                    dataloader, epoch, criterion):
    
    print("Start Train ...")
    model.train()
    subprocess.run(["nvidia-smi"])

    losses_train = []
    model_train_result = []
    train_target = []


    for data, targets in tqdm(dataloader):
        data = data.to(device)
        targets = targets.to(device).type(torch.float)


        outputs = model(data)
        model_train_result.extend(outputs.detach().cpu().numpy().tolist())
        train_target.extend(targets.cpu().numpy())


        loss = criterion(outputs, targets)

        losses_train.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    train_auc = roc_auc_score(train_target, np.array(model_train_result), average=None)

        
    if lr_scheduler is not None:
        lr_scheduler.step()

    lr = lr_scheduler.get_last_lr()[0]
    print("Epoch [%d]" % (epoch),
          "Mean loss on train:", np.array(losses_train).mean(), 
          "AUC score:",np.array(train_auc),
          "Mean AUC score on train:", np.array(train_auc).mean(), 
          "Learning Rate:", lr)

    
    return np.array(losses_train).mean(), np.array(train_auc), lr


def val_epoch(model, dataloader, epoch, criterion):
    
    print("Start Validation ...")
    model.eval()
    
    model_val_result = []
    val_target = []
    losses_val = []

    with torch.no_grad():
        for data, targets in tqdm(dataloader):

            data = data.to('cuda')
            targets = targets.to('cuda').type(torch.float)

            outputs = model(data)
            
            #loss
            loss = criterion(outputs, targets)
            losses_val.append(loss.item())

            
            model_val_result.extend(outputs.detach().cpu().numpy().tolist())
            val_target.extend(targets.cpu().numpy())
            
        val_auc = roc_auc_score(val_target, np.array(model_val_result), average=None)




        print("Epoch:  " + str(epoch) + " AUC valid Score:", np.array(val_auc), 
              "Mean valid AUC score", np.array(val_auc).mean())
        
    return np.array(losses_val).mean(), np.array(val_auc)

# In[22]:
criterion = nn.BCELoss()


# In[23]:

weights_dir = "/scratch/scratch6/akansh12/DeepEXrays/local_label"
# os.makedirs(weights_dir)

for param in model.parameters():
    param.requires_grad = True
    
params = [p for p in model.parameters() if p.requires_grad]

stage_epoch =  [5, 5, 10] #[12, 8, 5]
stage_optimizer = [
    torch.optim.Adamax(params, lr=0.0002),
    torch.optim.SGD(params, lr=0.00009, momentum=0.9),
    torch.optim.Adam(params, lr=0.00005)
]

stage_scheduler = [
    torch.optim.lr_scheduler.CosineAnnealingLR(stage_optimizer[0], 4, 1e-6),
    torch.optim.lr_scheduler.CyclicLR(stage_optimizer[1], base_lr=1e-5, max_lr=2e-4),
    torch.optim.lr_scheduler.CosineAnnealingLR(stage_optimizer[2], 4, 1e-6),
]


train_loss_history = []
val_loss_history = []
train_AUC_history = []
val_AUC_history = []
lr_history = []

for k, (num_epochs, optimizer, lr_scheduler) in enumerate(zip(stage_epoch, stage_optimizer, stage_scheduler)):
    for epoch in range(num_epochs):
        
        
        train_loss, train_auc, lr = train_one_epoch(model, optimizer, lr_scheduler,trainloader, epoch, criterion)
    
        val_loss, val_auc = val_epoch(model, testloader, epoch, criterion)
        
        
        # train history
        train_loss_history.append(train_loss)
        train_AUC_history.append(train_auc)
        lr_history.append(lr)
        
        #val history
        val_loss_history.append(val_loss)
        val_AUC_history.append(val_auc)
        
        # save best weights
        best_auc = max(np.mean(val_AUC_history, axis =1 ))
        if np.mean(val_auc) >= best_auc:
            torch.save({'state_dict': model.state_dict()},
                        os.path.join(weights_dir, f"{np.mean(val_auc):0.6f}_.pth"))
    
    print("\nNext stage\n")
    # Load the best weights
    best_weights =  sorted(glob.glob(weights_dir + "/*"),
                       key= lambda x: x[8:-5])[-1]

    state_dict = torch.load(best_weights)['state_dict']

    for keyA, keyB in zip(state_dict, model.state_dict()):
        state_dict = OrderedDict((keyB if k == keyA else k, v) for k, v in state_dict.items())
    model.load_state_dict(state_dict)
    # model.load_state_dict(checkpoint['state_dict'])

    print(f'Loaded model: {best_weights.split("/")[1]}')



np.save("/storage/home/akansh12/Vin-ChestXR-Abnormality-detection/model/train_loss_hist_local_label.npy", np.array(train_loss_history))
np.save("/storage/home/akansh12/Vin-ChestXR-Abnormality-detection/model/test_loss_hist_local_label.npy", np.array(val_loss_history))
np.save("/storage/home/akansh12/Vin-ChestXR-Abnormality-detection/model/train_AUC_hist_local_label.npy", np.array(train_AUC_history))
np.save("/storage/home/akansh12/Vin-ChestXR-Abnormality-detection/model/test_AUC_hist_local_label.npy", np.array(val_AUC_history))

