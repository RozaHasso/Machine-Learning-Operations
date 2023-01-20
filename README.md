# Image classifier for detecting cats and dogs
## Pipeline
[architecture](reports/figures/architecture.png)

## Dataset

To pull and process the dataset run the following command

<pre><code>make data
</code></pre>

### Version control

The dataset is version controlled with dvc. There exists 3 tags for the datasets

- <b>all_data:</b> <u>raw and processed</u> dataset of 279 cats and 278 dogs for training and 70 cats and 70 dogs for testing
- <b>raw_only:</b> <u>raw</u> dataset of 279 cats and 278 dogs for training and 70 cats and 70 dogs for testing
- <b>expanded_dataset:</b> expands the dataset with ~25k pictures og dogs and cats with a ~80/20 train/test split

To select a specific dataset run the following command before **make data**
<pre><code>git checkout <b>tag</b> data.dvc
</code></pre>

## Docker

The training can be containerized with docker. To build the docker images with the included docker file, from the root folder run

<pre><code>docker build -f trainer.dockerfile . -t trainer:latest
</code></pre>

This docker image can be passed the following arguments

- <b>-lr:</b> learning rate of the model. Default = 1e-4
- <b>-e:</b> Number of epochs to train for. Default = 5
- <b>-bs:</b> Batch size to use for the dataloader. Default = 16
- <b>-o:</b> Optimizer to use in training. Default = Adam
- <b>-pt:</b> Whether or not to use a pretrained ResNet50 CNN as the backbone. Default = True

The training script will log and report performance to wandb. Make sure you are logged into wandb by passing wandb login. Then when running the docker image, you must pass docker-run through wandb. Eg:

<pre><code>wandb docker-run --name experiment5 trainer:latest -lr 0.0001 -e 5 -bs 16 -o Adam -pt True</pre></code>


## Project plan

The project is done by 5 members: Abdulrahman Ramadan, Cristina Ailoaei, Jakob Ryttergaard Poulsen,  Roza Hasso, Teakosheen Joulak

1. Dataset: Cats and Dogs image classification: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
which consists of 697 files/images of cats and dogs.

2. The  project goal:  The goal of the project is to classify a given image whether it includes a cat or a dog object, we want to create a structure repository to train a neural network model 
logging the results and the performance with reproducible  experiments. 

3. Framework: We will use Pytorch Image Models TIMM, because it includes the necessery classes and code for initializing the neural network model. 

4. Deep Learning Model: We will use the Neural Network NN model to classify cats and dogs images

The tentative project plan is to use the following tools


### Code structure and versioning
- Cookiecutter for a structured repository template
- Git for version control of code
- DVC for version control of data

### Reproducibility
- Docker for system configuration
- Conda for Python environment configuration

### Experiment logging and monitoring
- Hydra for hyperparameter specification
- Wandb for experiment logging and model performance

### Code performance and structure
- Snakeviz for inspecting code performance
- Using flake8 testing to check for Pep8 compliance in our code
- Using isort for import structure



![project plan](reports/figures/project_plan.png)
