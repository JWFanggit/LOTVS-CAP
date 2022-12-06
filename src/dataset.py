import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange, repeat, reduce
import glob
from PIL import Image


class DADA(Dataset):
    def __init__(self, root_path, phase, interval,transform,
                  data_aug=False):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.transforms= transform
        self.data_aug = data_aug
        self.fps = 30
        self.num_classes = 2
        self.data_list, self.labels, self.clips, self.toas ,self.texts= self.get_data_list()

    def get_data_list(self):
        list_file = os.path.join(self.root_path, self.phase, self.phase + '.txt')
        assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
        fileIDs, labels, clips, toas,texts= [], [], [], [],[]
        samples_visited, visit_rows = [], []
        with open(list_file, 'r',encoding='utf-8') as f:
            # for ids, line in enumerate(f.readlines()):
            for ids, line in enumerate(f.readlines()):
                sample = line.strip().split(',')
                # print(sample )
                sample1=sample[0].strip().split(' ')
                word = sample[1].replace('\xa0', ' ')
                word.strip()
                fileIDs.append(sample1[0])  # 1/002
                labels.append(int(sample1[1]))  # 1: positive, 0: negative
                clips.append([int(sample1[2]), int(sample1[3])])  # [start frame, end frame]
                toas.append(int(sample1[4]))  # time-of-accident (toa)
                texts.append(word.strip())
                sample_id = sample1[0] + '_' + sample1[1]
                if sample_id not in samples_visited:
                    samples_visited.append(sample_id)
                    visit_rows.append(ids)
        # if not self.data_aug:
        #     fileIDs = [fileIDs[i] for i in visit_rows]
        #     labels = [labels[i] for i in visit_rows]
        #     clips = [clips[i] for i in visit_rows]
        #     toas = [toas[i] for i in visit_rows]
        # print(fileIDs,labels,clips,toas,texts )
        return fileIDs, labels, clips, toas, texts

    def __len__(self):
        return len(self.data_list)


    def read_rgbvideo(self, video_file, start, end):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        video_datas = []
        for fid in range(start, end+1, self.interval):
            video_data=video_file[fid]
            video_data=Image.open(video_data)
            if self.transforms:
                video_data = self.transforms(video_data)
                video_data= np.asarray(video_data, np.float32)
                video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32) # 4D tensor
        return video_data


    def read_foucsvideo(self, video_file, start, end):
        video_datas = []
        for fid in range(start, end + 1, self.interval):
            video_data = video_file[fid]
            video_data = Image.open(video_data)
            video_data = np.asarray(video_data, np.float32)/255
            video_data = video_data[None, ...]
            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        return video_data

    def gather_info(self, index):
        accident_id = int(self.data_list[index].split('/')[0])
        video_id = int(self.data_list[index].split('/')[1])
        texts=self.texts[index]
        # toa info
        start, end = self.clips[index]
        if self.labels[index] > 0: # positive sample
            self.labels[index]= 0,1
            assert self.toas[index] >= start and self.toas[index] <= end, "sample id: %s" % (self.data_list[index])
            toa = int((self.toas[index] - start) / self.interval)
        else:
            self.labels[index] = 1, 0
            toa = int(self.toas[index])  # negative sample

        data_info = np.array([accident_id, video_id, start, end,toa], dtype=np.int32)
        y=torch.tensor(self.labels[index], dtype=torch.float32)
        data_info=torch.tensor(data_info)
        return data_info,y,texts


    def __getitem__(self, index):
        # clip start and ending
        start, end = self.clips[index]
        # read RGB video (trimmed)
        video_path = os.path.join(self.root_path, self.phase, 'rgb_videos', self.data_list[index])
        video_path=glob.glob(video_path+'/'+"*.jpg")
        video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        video_data = self.read_rgbvideo(video_path, start, end)
        #read focus video
        focus_path = os.path.join(self.root_path, self.phase, 'focus_videos', self.data_list[index])
        focus_path = glob.glob(focus_path + '/' + "*.jpg")
        focus_path = sorted(focus_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        focus_data = self.read_foucsvideo(focus_path, start, end)
        data_info,y,texts= self.gather_info(index)
        return  video_data, focus_data, data_info,y,texts




if __name__=="__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    # device = torch.device('cuda:0')
    num_epochs = 50
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # learning_rate = 0.0001
    batch_size = 1
    shuffle = True
    pin_memory = True
    num_workers = 1
    rootpath = r'G:\full-test'
    frame_interval = 1
    input_shape = [224, 224]
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # transform_dict = {'image': transforms.Compose([ProcessImages(input_shape)])}
    train_data = DADA(rootpath, 'training', interval=1,transform=transform )

    # val_data = DADA2KS(rootpath, 'testing', interval=1,transform=transforms

    traindata_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False ,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    for video_data, focus_data, data_info,y,texts in traindata_loader:
        if video_data.shape[1]==150:
            print(video_data.shape,data_info[0:2],texts )
        else:
            print("True")
            print(data_info[0:2],texts)


