# CATPro

Clinical Acoustics & Text Processing (CATPro) provides tools for monitoring and predicting mental health disorders using acoustic signal processing and natural language processing.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
pip3 install sklearn
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

<!-- * [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds -->

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Daniel M. Low** 
* **Debanjan Ghosh**
* **Satra Ghosh** 

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

* Free software: Apache Software License 2.0
* Documentation: https://catpro.readthedocs.io.


```
catpro
|-- ...
|-- catpro
        |-- data (files & scripts to fetch them)
                |--datasets: large csv npy files (in .gitignore)
                |--outputs: outputs of models
                        |-- interpretation
                |--daic.py: preprocess DAIC dataset and return csv to ./datasets/
        |-- config.py (in .gitignore)
        |-- lstm.py
        |-- lstm_ht.py: for hyperparameter tuning
        |-- baseline_ht.py: SVMs for baseline with hyperparameter tuning (e.g., gridsearch)
        |-- baseline.py: final baseline used for publication with parameters from baseline_ht.py
        |-- interpretation.py
                - show top feature scores for a given input in a heatmap

        
```





## Datasets

- X_train_df.csv:  
        - X_train for text only

        - contains the following columns: 'id', 'train_test', 'y_binary', 'y_24', 'gender','start_time', 'stop_time', 'speaker', 'X_train_text'.

- X_train_text_audio.csv:
        - X_train for text and speech.
        - contains the following columns: 'id', 'train_test', 'y_binary', 'y_24', 'gender','start_time', 'stop_time', 'speaker', 'X_train_text', 'X_trian_audio_covarep0', ..., 'X_trian_audio_covarep74', 'X_trian_audio_formant0', ..., 'X_trian_audio_formant4'.  




## Scripts
### Training/testing models
- preprocess_data.py
        
        - Unzips original dataset and leaves only audio and text files (since vision files are large)
        
        - Creates a single data frame with all data: X_train_df_text. 

        - Merges covarep and formant features and outputs covarep_formant_concatenated.csv

        - Averages multiple audio windows (over 10 msec) to match segments of text which have a start and end time (e.g., 26:34-28:16 seconds) and adds it to single data frame: X_train_df.


- train_regression.py

        Hello


- train_regression.py

        Hello



## Acknowledgments

* Tuka
* Ev Fedorenko
* This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

        - Cookiecutter: https://github.com/audreyr/cookiecutter

        - `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage



TODO (see commented text)
-----


[//]: <> (add this at some point
.. image:: https://img.shields.io/pypi/v/catpro.svg
        :target: https://pypi.python.org/pypi/catpro
.. image:: https://img.shields.io/travis/danielmlow/catpro.svg
        :target: https://travis-ci.org/danielmlow/catpro
.. image:: https://readthedocs.org/projects/catpro/badge/?version=latest
        :target: https://catpro.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
.. image:: https://pyup.io/repos/github/danielmlow/catpro/shield.svg
     :target: https://pyup.io/repos/github/danielmlow/catpro/
     :alt: Updates 
)
