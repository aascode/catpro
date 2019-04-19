# CATPro

Clinical Acoustics & Text Processing (CATPro) provides tools for monitoring and predicting mental health disorders using acoustic signal processing and natural language processing.

## Getting Started

```
TODO 
```

### Requirements

```
TODO
requirements.txt


```

### Installing


```
TODO
```

End with an example of getting some data out of the system or using it for a little demo


#### Mozilla speech-to-text
https://github.com/mozilla/DeepSpeech

```bash
# Install:
cd HOME_DIR
virtualenv -p python3 ./deepspeech-venv/
source ./deepspeech-venv/bin/activate
pip3 install deepspeech
brew install sox

# Pretrianed model:
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-models.tar.gz

# Run:

deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio ./data/datasets/banda/556_b.wav
tar xvfz deepspeech-0.4.1-models.tar.gz
```

## Running the tests

```
TODO
```

### Break down into end to end tests

```
Give an example
```

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

Repo structure with descriptions:



```
# TODO: put preprocessing, data_helpers, and models into directories. 
# TODO: unify data helpers and preprocessing (into two .py: NLP and speech)

catpro
|-- ...
|-- catpro
        |-- data (files & scripts to fetch them)
                |-- datasets: dir with large files (in .gitignore)
                        | -- daic
                        | -- uic
                        | -- banda
                |-- outputs: dir with outputs of models
                        | -- daic
                        | -- uic
                        | -- banda
                |-- create_dataset_daic.py: fix filenames, convert to .wav, put in final directory in ./datasets/
                |-- create_dataset_uic.py
                |-- create_dataset_banda.py
                |-- config_create_dataset.py: paths
                |-- fetch_daic.py
                |-- fetch_uic.py
                |-- fetch_banda.py

        # Preprocessing and feature extraction
        |-- config_preprocess.py
        |-- preprocess_speech.py: denoise, trim silence, make segments and return csv to ./datasets/
        |-- extract_speech_features.py 
        |-- audio_transcribe.py
        |-- doc2vec.py: create doc2vec vectors for each text segment
        |-- feature_generator.py: NLP features
        |-- features: dir with scripts to generate NLP features
                |-- depression_ngram_genr.py
                ...

        # Data helpers
        |-- data_helpers.py
        |-- data_handler.py
        |-- util.py
        |-- interpretation.py: eg, show top feature scores for a given input in a heatmap

        # Models
        |-- config.py: (in .gitignore) paths and parameters for models below
        |-- gss.py: group shuffle split with permutation test
        |-- lstm.py
        |-- lstm_ht.py: for hyperparameter tuning
        |-- baseline_ht.py: SVMs for baseline with hyperparameter tuning (e.g., gridsearch, randomsearch)
        |-- baseline_PROJECT_NAME.py: final baseline used for publication with parameters from baseline_ht.py
        |-- gpt2.py

        
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
