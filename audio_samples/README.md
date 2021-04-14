# Audio Samples
This directory contains audio samples of transcriptions created by an agent during training. The samples are taken at different times during the training procedure and represent how the agent's skill evolves.  The agent is trained using a *deep Q-learning* variant. The filenames indicate after which episode the transcription was made.
So file `transcription-episode-01.mp3` was created after episode 1 (note that episode indices here start at 0).
In the recordings it can be observed that the agent slowly begins to learn how to correctly distiguish the different notes. We have the following samples:

Filename | Description
---|---
 **`original.flac`**| The original recording to be transcribed in the *Entchen Task*.
**`transcription-episode-01.mp3`** | After episode 1 the agent has not learned anything meaningful yet. The transcription is mostly random, although similar sounds are transcribed to the same wrong note. The randomness is due to the initialisation of the deep Q-network.
**`transcription-episode-20.mp3`** | After episode 20 the agent has not managed to reach the end of the transcription yet but has learned a fair amount about the first few notes and transcribes these correctly.
**`transcription-episode-35.mp3`** | After episode 35 the agent has reached the end of the transcription and learned to correctly transcribed the entire song.

Note that the transcriptions are created by acting greedily with respect to the action-value function approximation after that episode.

# How to Listen
You unfortunately have to download the files, GitHub itself does currently not support audio playback.
