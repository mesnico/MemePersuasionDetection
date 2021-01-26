import itertools

from PIL import Image
from torch.utils.data import Dataset
import torch
import json
import os

from transformers import BertTokenizer
from format_checker.task1_3 import read_classes


class SemEvalDataset(Dataset):
    def __init__(self, cfg, split='train', val_fold=0, transforms=None):
        self.data_root = cfg['dataset']['root']
        task = cfg['dataset']['task']
        self.transforms = transforms

        # load folds from json file
        with open(os.path.join(self.data_root, 'folds.json'), 'r') as f:
            folds = json.load(f)
        if split == 'train':
            ids = [v for k, v in folds.items() if int(k) != val_fold]
            ids = list(itertools.chain.from_iterable(ids))
        elif split == 'val':
            ids = [v for k, v in folds.items() if int(k) == val_fold]
            ids = ids[0]
        else:
            ids = None

        if split == 'train' or split == 'val':
            self.split_name = 'training'
        elif split == 'dev':
            self.split_name = 'dev'
        elif split == 'test':
            self.split_name = 'test'

        if task == 3:
            label_file = os.path.join(self.data_root, '{}_set_task3'.format(self.split_name), '{}_set_task3.txt'.format(self.split_name))
            self.class_list = read_classes('techniques_list_task3.txt')
        elif task == 1:
            label_file = os.path.join(self.data_root, '{}_set_task1.txt'.format(self.split_name))
            self.class_list = read_classes('techniques_list_task1-2.txt')

        with open(label_file, 'r', encoding='utf8') as f:
            self.targets = json.load(f)
        for t in self.targets:
            t['path'] = os.path.join(self.data_root, '{}_set_task3'.format(self.split_name))

        if task == 3:
            label_file_dev = os.path.join(self.data_root, 'dev_set_task3_labeled', 'dev_set_task3_labeled.txt')
        elif task == 1:
            label_file_dev = os.path.join(self.data_root, 'dev_set_task1.txt')
        if os.path.isfile(label_file_dev) and self.split_name == 'training':
            with open(label_file_dev, 'r', encoding='utf8') as f:
                targets = json.load(f)
                for t in targets:
                    t['path'] = os.path.join(self.data_root, 'dev_set_task3_labeled')
                self.targets.extend(targets)

        # filter targets using the ids
        if split == 'train' or split == 'val':
            self.targets = [t for t in self.targets if t['id'] in ids]
        print('ok')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        info = self.targets[item]
        if 'labels' in info:
            classes = info['labels']
            classes_ids = [self.class_list.index(x) for x in classes]
        else:
            classes_ids = None

        text = info['text'].split('\n\n')

        if 'image' in info:
            # task 3
            img_path = os.path.join(info['path'], info['image'])
            image = Image.open(img_path).convert("RGB")
            if self.transforms is not None:
                image = self.transforms(image)
        else:
            # task 1
            image = None

        return image, text, classes_ids, info['id']


class Collate:
    def __init__(self, config, classes):
        self.vocab_type = config['text-model']['name']
        self.data_root = config['dataset']['root']
        if self.vocab_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
        self.class_list = classes # read_classes('techniques_list_task3.txt')

    def __call__(self, data):
        images, texts, classes, ids = zip(*data)

        tokenized_texts = []
        for ts in texts:
            tokenized = [self.tokenizer.cls_token_id]
            for c in ts:
                tokenized.extend(self.tokenizer.encode(c, add_special_tokens=False))
                tokenized.append(self.tokenizer.sep_token_id)
            tokenized_texts.append(torch.LongTensor(tokenized))

            # texts = [torch.LongTensor(self.tokenizer.encode(c, max_length=max_len, pad_to_max_length=True, add_special_tokens=False))
            #                 for c in texts]

        text_lengths = [len(c) for c in tokenized_texts]
        max_len = max(text_lengths)

        bs = len(texts)
        out_texts = torch.zeros(bs, max_len).long()
        for ot, tt, l in zip(out_texts, tokenized_texts, text_lengths):
            ot[:l] = tt

        images = torch.stack(images, 0) if images[0] is not None else None # in case of task 1 images is None

        # out classes is a one-hot vector of labels (for multi-classification)
        out_classes = torch.zeros(bs, len(self.class_list))
        for oc, c in zip(out_classes, classes):
            oc[c] = 1

        return images, out_texts, text_lengths, out_classes, ids


import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T

if __name__ == '__main__':
    cfg_file = 'cfg/config_task3.yaml'
    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f)

    # initialize dataset
    transforms = T.Compose([T.Resize(256),
                           T.CenterCrop(224),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])])
    dataset = SemEvalDataset(cfg, split='train', transforms=transforms)
    print(dataset[0])

    # initialize dataloader
    collate_fn = Collate(cfg)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    for images, texts, text_lengths, out_classes, ids in dataloader:
        print('ciao')


