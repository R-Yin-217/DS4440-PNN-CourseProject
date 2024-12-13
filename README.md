# DS4440-PNN-CourseProject
This is the repository for the DS4440 Practical Neural Network Course Project at Northeastern University.

- data

    This directory includes the file needed to crop the original dataset: the IMDb dataset.

- src 

    This directory includes the source files required to conduct the experiments.

    - baseline

        This folder contains the code for the baselines, including BERT and BEiT, both pretrained and randomly initialized.

    - experiment

        This folder contains the experiments for inverse modality, where BEiT processes text and BERT processes images, both pretrained and randomly initialized.

    - constrastive

        This folder includes the code to extract features from images to create 768-dimensional vectors. It also contains the model used to process these vectors, both pretrained and randomly initialized.