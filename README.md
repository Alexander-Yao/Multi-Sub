# Multi-Sub(Neruips'24)

This repository contains the implementation of Multi-Sub.

## Folder Structure

The project is organized as follows:

```bash
.
├── clip/                  
├── dataset/               # Contains datasets for training and evaluation
│   ├── fruit/             # Specific dataset for 'fruit'
│   │   ├── color/         # Sub-dataset for fruit color
│   │   ├── instance/      # Sub-dataset for fruit instances
│   │   └── species/       # Sub-dataset for fruit species
│   ├── cifar10/           # Specific dataset for CIFAR-10
│   │   ├── type/          # Sub-dataset for CIFAR-10 type clustering (e.g., transportation, animals)
│   │   └── environment/   # Sub-dataset for CIFAR-10 environment clustering (e.g., land, air, water)
├── gpt.py                 # Implementation related to GPT
├── main.py                # Main script to run training and evaluation
├── parse.py               # Argument parsing for command-line execution
├── README.md              # This README file
├── requirements.txt       # Dependencies required for running the project
└── setup.py               # Installation setup
```

## Requirements
To run this project, ensure you have the following dependencies installed:
```python
pip install -r requirements.txt
```

## Usage
```python
python main.py
```

## Citation
If you use this code in your research, please cite our paper:
@inproceedings{yourpaper2024,
  author    = {Your Name and Co-author Name},
  title     = {Your Paper Title},
  booktitle = {Conference Name, {YEAR}},
  year      = {2024}
}