# cross_version_learning

This is a Tensorflow code repository accompanying the following paper:

```bibtex
@inproceedings{KrauseWM23_CrossVersionRepresentationLearning_ISMIR,
  author    = {Michael Krause and Christof Wei{\ss} and Meinard M{\"u}ller},
  title     = {A Cross-Version Approach to Audio Representation Learning for Orchestral Music},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  pages     = {XXX--XXX},
  address   = {Milano, Italy},
  year      = {2023}
}
```

This repository contains code and trained models for the paper's experiments. The annotations used in the paper are available on the [project website](https://www.audiolabs-erlangen.de/resources/MIR/2023-ISMIR-CrossVersionLearning).
For details and references, please see the paper.

# Installation and Data Preparation

```bash
cd cross_version_learning
conda env create -f environment.yml
conda activate cross_version_learning
```

Extract the dataset in a ```data``` subdirectory of this repository. You will need to obtain the audio files and correctly name them according to the names of the annotation files. See the dataset website for details. Furthermore, extract the trained models from the project website in the ```outputs/models``` subdirectory.

# Running Experiments

Run scripts using, e.g., the following commands:  
```bash
export CUDA_VISIBLE_DEVICES=0
python 02_extract_embeddings.py CV
```
where an additional parameter ```CV``` is submitted here to extract embeddings for the ```CV``` model.

The individual scripts perform the following steps:
- ```01_train_model.py```: Train a representation model from scratch. This will overwrite the stored model checkpoints. Submit either ```CV``` or ```SV``` as argument to this script to train the corresponding representation learning method.
- ```02_extract_embeddings.py```: Extract embeddings from recordings using an already trained model. Submit either ```CV```, ```SV``` or ```Sup``` to extract embeddings for the corresponding model.
- ```03_ssm_evaluation_quantitative.py```: Reproduce Figures 4 and 5 from the paper. Here, learned representations are evaluated by computing self-similarity matrices (SSM) based on them and then comparing their structural boundaries with those from reference matrices. Chroma and MFCC features are used as baselines here. The resulting plots are found in ```outputs/ssm_boundaries```.
- ```04_probing_evaluation.py```: Perform probing evaluation as in Section 5.4/Tables 2 and 3 in the paper. Submit two parameters. First, either ```CV```, ```SV```, ```Sup```, ```Chroma``` or ```MFCC``` (corresponding to the type of feature to be evaluated). Second, either ```Inst``` or ```PitchClass``` corresponding to the target task for probing. So calling ```python 04_probing_evaluation.py CV Inst``` will evaluate the proposed CV representations for instrument classification. The evaluation results are then found in ```outputs/probing```.

Note that there may be minor differences in results compared to what is reported in the paper due to random effects of training, newer versions of the packages being used here (e.g. librosa), and slight changes to the code.