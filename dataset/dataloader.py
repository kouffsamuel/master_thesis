import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split,Subset
import numpy as np
import torch


Sequences = {'Validation':
             ['RECORD@2020-11-22_12.49.56',
              'RECORD@2020-11-22_12.11.49',
              'RECORD@2020-11-22_12.28.47',
              'RECORD@2020-11-21_14.25.06'],
            'Test':[
                'RECORD@2020-11-22_12.45.05',
                'RECORD@2020-11-22_12.25.47',
                'RECORD@2020-11-22_12.03.47',
                'RECORD@2020-11-22_12.54.38']}

def RADIal_collate(batch):
    radar_FFTs = [b[0] for b in batch]
    segmaps = [b[1] for b in batch]
    out_labels = [b[2] for b in batch]
    box_labels = [b[3] for b in batch]
    images = [b[4] for b in batch]

    FFTs = torch.from_numpy(np.stack(radar_FFTs)).permute(0,3,1,2)
    segmaps = torch.from_numpy(np.stack(segmaps))
    encoded_label = torch.from_numpy(np.stack(out_labels))
    images = torch.from_numpy(np.stack(images))
    labels = [torch.from_numpy(b[:,:-2].astype(np.float32)) for b in box_labels]
    meta   = [b[:, -2:] for b in box_labels]
    
    return FFTs, encoded_label, segmaps, labels, images, meta

def CreateDataLoaders(dataset,config=None,seed=0):

    if(config['mode']=='random'):
        # generated training and validation set
        # number of images used for training and validation
        n_images = dataset.__len__()

        split = np.array(config['split'])
        if(np.sum(split)!=1):
            raise NameError('The sum of the train/val/test split should be equal to 1')
            return

        n_train = int(config['split'][0] * n_images)
        n_val = int(config['split'][1] * n_images)
        n_test = n_images - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val,n_test], generator=torch.Generator().manual_seed(seed))

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Train Val ratio:', config['split'])
        print('      Training:', len(train_dataset),' indexes...',train_dataset.indices[:3])
        print('      Validation:', len(val_dataset),' indexes...',val_dataset.indices[:3])
        print('      Test:', len(test_dataset),' indexes...',test_dataset.indices[:3])
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'], shuffle=False, num_workers=config['val']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        test_loader = DataLoader(test_dataset, batch_size=config['test']['batch_size'], shuffle=False, num_workers=config['test']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)

        return train_loader, val_loader, test_loader
    elif(config['mode']=='sequence'):
        dict_index_to_keys = {s:i for i,s in enumerate(dataset.sample_keys)}

        Val_indexes = []
        for seq in Sequences['Validation']:
            idx = np.where(dataset.labels[:,14]==seq)[0]
            Val_indexes.append(dataset.labels[idx,0])
        Val_indexes = np.unique(np.concatenate(Val_indexes))

        Test_indexes = []
        for seq in Sequences['Test']:
            idx = np.where(dataset.labels[:,14]==seq)[0]
            Test_indexes.append(dataset.labels[idx,0])
        Test_indexes = np.unique(np.concatenate(Test_indexes))

        val_ids = [dict_index_to_keys[k] for k in Val_indexes]
        test_ids = [dict_index_to_keys[k] for k in Test_indexes]
        train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))

        train_dataset = Subset(dataset,train_ids)
        val_dataset = Subset(dataset,val_ids)
        test_dataset = Subset(dataset,test_ids)

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Training:', len(train_dataset))
        print('      Validation:', len(val_dataset))
        print('      Test:', len(test_dataset))
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'], shuffle=False, num_workers=config['val']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        test_loader = DataLoader(test_dataset, batch_size=config['test']['batch_size'], shuffle=False, num_workers=config['test']['num_workers'], pin_memory=True, collate_fn=RADIal_collate)
        return train_loader, val_loader, test_loader

    else:
        raise NameError(config['mode'], 'is not supported !')
        return