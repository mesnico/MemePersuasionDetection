import argparse
import os
import json

import torch
import tqdm
from torch.utils.data import DataLoader

import models
from dataset import SemEvalDataset, Collate
from format_checker.task1_3 import read_classes, check_format_task1_task3
from scorer.task1_3 import evaluate

from torchvision import transforms as T


def id_to_classes(classes_ids, labels):
    out_classes = []
    for elem in classes_ids:
        int_classes = []
        for idx, ids in enumerate(elem):
            if ids:
                int_classes.append(labels[idx])
        out_classes.append(int_classes)
    return out_classes

def main(opt):
    checkpoint = torch.load(opt.checkpoint, map_location='cpu')
    cfg = checkpoint['cfg']
    if 'task' not in cfg['dataset']:
        cfg['dataset']['task'] = 3 # for back compatibility
        print('Manually assigning: task 3')

    if cfg['dataset']['task'] == 3:
        classes = read_classes('techniques_list_task3.txt')
    elif cfg['dataset']['task'] == 1:
        classes = read_classes('techniques_list_task1-2.txt')

    if opt.ensemble:
        checkpoints_folder = os.path.split(opt.checkpoint)[0]
        checkpoints_files = [os.path.join(checkpoints_folder, f) for f in os.listdir(checkpoints_folder) if '.pt' in f]
    else:
        checkpoints_files = [opt.checkpoint]

    ensemble_models = []
    for chkp in checkpoints_files:
        model = models.MemeMultiLabelClassifier(cfg, classes)
        checkpoint = torch.load(chkp, map_location='cpu')
        # Load weights to resume from
        if not cfg['text-model']['fine-tune'] and not cfg['image-model']['fine-tune']:
            # the visual and textual modules are already fine
            model.joint_processing_module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        model.cuda().eval()
        ensemble_models.append(model)

    # Load data loaders
    test_transforms = T.Compose([T.Resize(256),
                                 T.CenterCrop(224),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

    if opt.validate:
        dataset = SemEvalDataset(cfg, split='val', transforms=test_transforms, val_fold=opt.val_fold)
    else:
        dataset = SemEvalDataset(cfg, split='dev', transforms=test_transforms)

    collate_fn = Collate(cfg, classes)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
                                  num_workers=2, collate_fn=collate_fn)

    resumed_logdir, resumed_filename = os.path.split(opt.checkpoint)
    del checkpoint  # current, saved
    print('Model {} resumed from {}, saving results on this directory...'.format(resumed_filename, resumed_logdir))

    predictions = []
    metrics = {}
    thr = opt.threshold
    for it, (image, text, text_len, labels, ids) in enumerate(tqdm.tqdm(dataloader)):
        if torch.cuda.is_available():
            image = image.cuda() if image is not None else None
            text = text.cuda()
            # labels = labels.cuda()

        ensemble_predictions = []
        with torch.no_grad():
            for model in ensemble_models:
                pred_probs = model(image, text, text_len, return_probs=True)
                ensemble_predictions.append(pred_probs)
            prob_ensemble = torch.stack(ensemble_predictions, dim=1).mean(dim=1)
            class_ensemble = prob_ensemble > thr
            pred_classes = id_to_classes(class_ensemble, classes)

        for id, labels in zip(ids, pred_classes):    # loop over every element of the batch
            predictions.append({'id': id, 'labels': labels})

    if opt.validate:
        macro_f1, micro_f1 = evaluate(predictions, dataloader.dataset.targets, classes)
        print('MacroF1: {}\nMicroF1: {}'.format(macro_f1, micro_f1))
    else:
        # dump predictions on json file
        out_json = os.path.join(resumed_logdir, 'predictions_thr{}.json'.format(thr))
        with open(out_json, 'w') as f:
            json.dump(predictions, f)

        # cross check
        if not check_format_task1_task3(out_json, CLASSES=classes):
            print('Saved file has incorrect format! Retry...')
        print('Detection dumped on {}'.format(out_json))
        print('Num memes: {}'.format(len(predictions)))

    print('DONE!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.3, type=float, help="Threshold to use for classification")
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none). Loads only the model')
    parser.add_argument('--validate', action='store_true', help="If not set, default is inference on the dev set")
    parser.add_argument('--val_fold', default=0, type=int, help="Which fold we validate on (use with --validate)")
    parser.add_argument('--ensemble', action='store_true', help='Enables model ensembling')
    # parser.add_argument('--config', type=str, help="Which configuration to use. See into 'config' folder")

    opt = parser.parse_args()
    print(opt)

    main(opt)