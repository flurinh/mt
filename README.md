# MT

## Installation on Merlin

Create a conda env:

    module load anaconda
    conda env create -f environment.yml --prefix=/data/project/bio/schertler/Flurin/mtenv

Activate (command line)

    conda activate /data/project/bio/schertler/Flurin/mtenv

To just the environment to show up in jupyter, you may need to run (within the environment)

    python -m ipykernel install --user --name mtenv --display-name "Python [conda env:mtenv]"

### Jupyterhub (Merlin)

1. Clone the repo

    git clone git@github.com:flurinh/mt.git

2. Install the env into jupyter. (needs testing)

    conda activate /data/project/bio/schertler/Flurin/mtenv
    python -m ipykernel install --user --name mtenv --display-name "Python [conda env:mtenv]"

I'm not sure whether this is necessary or sufficient to make the environemnt
show up in jupyterhub. See [merlin-jupyter
docs](https://lsm-hpce.gitpages.psi.ch/merlin6/jupyterhub.html)

2. Launch jupyterhub: [https://merlin-jupyter.psi.ch:8000](https://merlin-jupyter.psi.ch:8000)

3. Nagivate to your clone and open `WALKTHROUGH.ipynb`

4. Go to Kernel > Change kernel > Python [conda env:mtenv]

5. Go to Cell > Run all


## Adding dependencies

New dendencies should be added to environment.yml. After adding something, update:

    conda env update -f environment.yml

