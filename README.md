# MLMI Project - Transcribing MIDI from WAVE
Code repository for research project done as part of the MLMI MPhil at the University of Cambridge.

## Repository Structure
**`audio_samples`:** this directory contains audio files created by the agent at different stages during training. These audio samples illustrate the progress the agent makes during training. The training procedure was the modified deep Q-learning discussed in the report.

**`code`:** this directory contains the code that produced the RL experiments presented in the report. It has the following structure (with `#...` being clarifying comments):

```python
code
│
├── transcribe      # MAIN code base directory
│   │
│   ├── algorithms      # RL and other algorithms used
│   │   ├── deep_q_learning.py
│   │   ├── reinforce.py
│   │   ├── sampling_inference.py
│   │   └── ...
│   ├── environments
│   │   ├── piano.py    # environments for piano
│   │   │               # transcription MDP
│   │   │               # (incl. stopping, repeating var.)
│   │   │
│   │   ├── rewards.py  # reward functions used in MDP
│   │   └── ...
│   ├── constants
│   │   └── ...
│   ├── models         # neural network approximator
│   │   └── ...        # models
│   ├── utils
│   │   ├── pianoteq.py # Pianoteq CLI wrapper
│   │   └── ...
│   ├── deepqlearning_experiment.py  # sacred experiment scripts
│   ├── reinforce_experiment.py      # for both DQL and REINFORCE
│   └── data.py
│
└── external
    └── ...

```

## Required Packages

See `./requirements.txt` for detailed Python package requirements. Note that for most experiments also an installation of the [Pianoteq 6 software](https://www.modartt.com/) is required.


## License

All code is published under [MIT License](https://github.com/rdnfn/mlmi-project/blob/main/LICENSE) by @rdnfn except the scripts in the `./code/external` directory, which are published under [MIT License](https://github.com/rdnfn/mlmi-project/blob/main/code/external/LICENSE) by Jong Wook Kim.





