# AIMH at SemEval-2021 Task 6: Multimodal Classification Using an Ensemble of Transformer Models

This repo contains the code for replicating our system for the SemEval-2021 Task 6 challenge: [Detection of Persuasive Techniques in Texts and Images](https://propaganda.math.unipd.it/semeval2021task6/). Our paper is available [here](https://aclanthology.org/2021.semeval-1.140/).

<p align="center">
  <img src="https://user-images.githubusercontent.com/25117311/152802885-88b8b26e-8e86-4805-96d6-169163294cfa.png">
</p>


## Setup
Clone this repo:
```
git clone https://github.com/mesnico/MemePersuasionDetection
```

Then, install the requisites (virtualenv or conda are recommended):
```
pip install -r requirements.txt
```

Extract the images in the data folder
```
cd data
for z in *.zip; do unzip $z; done
cd ..
```

## Train and Validation
To train the network issue the following command:
```
python train.py --config cfg/config_task3.yaml --logger_name runs/task3 --val_step 100 --num_epochs 50 
```
N.B.: `runs/task3` is the folder where the checkpoints and the tensorboard files will be saved. Opening a tensorboard on this directory will show the training and validation curves.

To perform inference on the best-performing model, issue the following command:
```
python inference.py --checkpoint runs/task3/model_best_fold0.pt --validate 
```

## Citation

If you found our work useful for your research, please cite our paper:

    @inproceedings{messina2021aimh,
      title={AIMH at SemEval-2021 Task 6: multimodal classification using an ensemble of transformer models},
      author={Messina, Nicola and Falchi, Fabrizio and Gennaro, Claudio and Amato, Giuseppe},
      booktitle={Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)},
      pages={1020--1026},
      year={2021}
    }
