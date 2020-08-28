# Constellation

Constellation is a visualization tool created to observe the progression of neural network binary classifiers during training.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```

Then clone the [GitHub repository](https://github.com/RECHE23/Constellation).

```bash
git clone https://github.com/RECHE23/Constellation.git
cd Constellation
```

## Usage
### Setup models and datasets
To use Constellation, you first need to establish the models and the datasets you want to use in the ``user_data`` directory.

#### Establish the models
To establish the models, you can do so by importing them in ``models.py`` or by defining them directly in the file.

To make the models available to Constellation, they have to be referenced in the ``models`` dictionary using the following structure:
```python
models = {'<NAME>': {'network': <CONSTRUCTOR>,
                     'optimizer': (<OPTIMIZER>, <OPTIMIZER_PARAMS>),
                     'loss function': <LOSS FUNCTION>,
                     'scheduler': (<SCHEDULER>, <SCHEDULER_PARAMS>)}
         }
```
Where ``<NAME>`` is the displayed name of the network, ``<CONSTRUCTOR>`` is the PyTorch constructor of the network, ``<OPTIMIZER>`` is the PyTorch optimizer to use along with the dictionary of parameters to use ``<OPTIMIZER_PARAMS>``. 

Finally, ``'scheduler'`` can be set to ``None`` if no scheduler is required. Otherwise, ``<SCHEDULER>`` is the specified Poutyne scheduler and ``<SCHEDULER_PARAMS>`` is the corresponding dictionary of parameters.

For more information about the use of the Poutyne scheduler, visit [poutyne.org](https://poutyne.org/index.html).

#### Establish the datasets
To establish the models, you can do so by importing them in ``datasets.py`` or by defining them directly in the file.

The wrapper class ``DS`` has to be used. For examples of datasets, you can look at the content of ``record/datasets/collection``.

The simplest use of the ``DS`` wrapper is ``DS(data=X, target=y)``, where ``X`` is a Numpy array of the samples and ``y`` is a Numpy array of the targets (Following the same shape as Sci-Kit Learn datasets when using ``return_X_y``.).

To make the datasets available to Constellation, they have to be inserted in the ``datasets`` list.

### Record the training
You can start a recording of the training using the following command:
```bash
python main.py record
```

#### Number of epochs
To specify a specific number of epochs, you can use the ``-e`` option.

For example:
```bash
python main.py record -e 100
```
The number of epochs, by default, is 10.

#### Batch size
To specify a specific batch size, you can use the ``-b`` option.

For example:
```bash
python main.py record -b 64
```
The batch size, by default, is 32.

#### Representations
To specify the representations, you can use the ``-r`` option. The possible values are ``Original``, ``PCA``, ``LDA``, ``Fast_ICA``, ``Kernel_PCA``, and/or ``UMAP``.

For example:
```bash
python main.py record -r umap,kernel_pca,fast_ica
```
The representations, by default, are ``pca``, ``lda`` and ``original``.

#### Random seed
It is also possible to specify whether to use or not a random seed.

To specify a random seed, you can use the ``-s`` option.
```bash
python main.py record -s 777
```
By default, the seed ``42`` is used.

In order to deactivate the random seed, you can use the ``--no_seed`` flag.

For example:
```bash
python main.py record --no_seed
```

### Animate the recordings
You can animate all the recordings, individually, by using the following command:
```bash
python main.py animate all
```

#### Animate a specific recording
To animate a specific recording, individually (``2_Neurons_per_layer_Moons.record`` for example), you can either use the following command:
```bash
python main.py animate -s 2_Neurons_per_layer_Moons.record
```
Or you can select the file in the menu after using the following command:
```bash
python main.py animate
```

#### Animate a specific representation
To specify representations, you can use the ``-r`` option.

For example:
```bash
python main.py animate all -r pca,fast_ica
```
If no representation is specified, all the representations are animated.

#### Animate a layout (In development...)
It is also possible to generate a grid layout in order to compare the recordings. To do so, use the following command and follow the instructions from the menu:
```bash
python main.py animate layout
```
This feature isn't fully implemented and may contain bugs.