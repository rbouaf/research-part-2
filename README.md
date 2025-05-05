### Reproducing and adapting a 2019 paper
The original paper is by Christopher Thomas and Adriana Kovashka.
<br> All data and info and the paper is located at https://people.cs.pitt.edu/~chris/politics/
<br> You need to redownload the code and you need to download the data from the original paper (It's 130GB and thus can't fit in this repo).

Original Message from original paper:
```
Images are contained in the imgs/ folder
    The path is of the form imgs/(LEFT or RIGHT)/(POLITICAL ISSUE)/(MEDIA SOURCE)/imgs....
    
We also provide the "Human Concepts" images we harvested from Google in the mturk_concepts/ folder

We provide our MTurk interface in the attached HTML file which was used to collect annotations. The file mturk_db.pickle contains the human annotations in our large scale study. We also provide the annotations from our small mturk study in small_mturk_experiment.pickle 

The dataset metadata pickle contains the full dataset annotations broken down by left / right, issue, etc. We also provide annotations for the img path, the website url, the image description, the article text, etc. 

Last updated: 12/6/2019 ----------------------------------------

Code requires PyTorch and Python 3.7+. Anaconda distribution is highly recommended.

This code implements our model as presented in our NeurIPS 2019 paper.

It is assumed that the dataset is in the same folder as this code. Alternatively, create a symbolic link to the imgs folder of the dataset in the directory of the code.

First, run train_model.py to train a stage 1 model. We provide the precomputed Doc2Vec features as an attached pickle file. 
If you wish to run on your own data, first run GenSym on your own documents and extract Doc2Vec features. 

Next, run extract_features.py after choosing the best model on the validation set. We also provide pre-extracted features for the test set.

Finally, run train_eval_stage_2.py which trains the classifier using the extracted features and evaluates it on the test set.

We also provide a pretrained classifier model. Note that due to random variations in training and initialization, results may slightly deviate from the paper each time the code is run. If you wish to train your own model, you must first extract features on the train / test dataset and uncomment lines to train in the stage 2 file.

----

Additional notes: The actual project contains many more files than is contained in this zip, only the most important was given in the interest of space.
```

We run these files using McGill's SLURM GPUs, thanks to science.it@mcgill.ca.
In `mimi.cs.mcgill`, create an anaconda environment with all the correct dependencies (yes, this is necessary) then do `module load slurm` and 
`sbatch slurm_all.sh` but you should figure this out, check the documentation from McGill.

We modified the original code to have checkpointing (`our_train_model.py`), because SLURM has time limits. So we have a script called `every4hours.sh` to run it continuously.
Watch out, there is a max limit of jobs (its very big so dont stress too much, and you can always ask for more) you can run so pay attention to your script and end it with ctrl+c once youre done training 50 epochs.

We implemented checkpointing for `our_extract_features.py` and `our_train_eval_stage_2.py` but we didn't need to, just run `og_extract_features.py` and `og_train_eval_stage_2.py`.
You can modify other settings in `slurm_all.sh`. 


