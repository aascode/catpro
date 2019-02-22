# Predicting Depression from Speech

TODO Description

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

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Tuka




## Data

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

