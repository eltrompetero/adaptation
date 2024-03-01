# Optimal adaptation model

## Introduction
Code for the manuscript "A prediction of optimal timescales for niche
construction" by Edward D. Lee, Jessica C.  Flack, and David C.  Krakauer. 

Preprint located at [https://arxiv.org/abs/2209.00476](https://arxiv.org/abs/2209.00476).

Code is written by Eddie Lee. 

## Instructions
Functions for reproducing figures are in the file "pipeline.py". The
specification of the Python 3 environment used to generate the results are in
"adaptation.yml".

First clone the repository using
```bash
git clone https://github.com/eltrompetero/adaptation.git
```

After navigating into the directory of the repository, install the conda environment.
```bash
conda env create -f pyutils/adaptation.yml
```

Create the cache folder to store results.
```bash
mkdir cache
```

You should be able to run some of the pipeline code as, for example,
```python
from pyutils import *

pipe.tau_range()
```

## System specification
This code runs using Python 3.11 and was run on a 64-bit Ubuntu system. Note
that some of the simulations require a substantial amount of RAM, so the
recommendation is to have at least 128GB and additional 64-128GB of swap space to
run all the pipeline code.

## Troubleshooting
For any questions about the code and results, please open an issue on the GitHub
page [https://github.com/eltrompetero/adaptation].
