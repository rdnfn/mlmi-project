import subprocess
import shutil
import IPython
import pathlib


class PianoteqClient:
    """Wrapper class to access the Pianoteq 6 command line interface via Python."""
    
    def __init__(self, installation_path, tmp_dir="./tmp/tmp_pianoteq/", verbose=0):
        
        self.pianoteq_path = installation_path
        self.verbose = verbose
        self.tmp_dir = tmp_dir
        
        
    def get_help(self):
        try:
            out = subprocess.check_output([self.pianoteq_path, "--help"])
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, 
                                                                                     e.returncode,
                                                                                    e.output))
        print(out.decode("utf-8") )
        
        
    def create_audio(self, midi_path,
                     out_path=None,
                     tmp_midi_path=None,
                     out_rate=16000,
                     out_mono=True):
        """Creates audio file from MIDI."""
        
        # Note this also copies the MIDI file there because of 
        # an apperent bug in the Pianoteq interface that randomly
        # selects a MIDI file from the chosen directory.
        
        if not out_path:
            out_path = self.tmp_dir + "pianoteq_audio.mp3"
        if not tmp_midi_path:
            tmp_midi_path = self.tmp_dir + "pianoteq_midi.midi"
        
        if out_path[-4:]==".mp3":
            out_type="mp3"
        elif out_path[-5:]==".flac":
            out_type="flac"
        elif out_path[-4:]==".wav" or out_path[-5:]==".wave":
            out_type="wav"
        else:
            raise ValueError("out_path needs to be a file path ending with one of '.mp3','.flac','.wav' or '.wave'.")
        
        pathlib.Path(tmp_midi_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        
        if midi_path != tmp_midi_path:
            shutil.copy(midi_path, tmp_midi_path)
        
        commands_pianoteq = [self.pianoteq_path,
                             "--midi", tmp_midi_path,
                             "--{}".format(out_type), out_path,
                             "--rate", str(out_rate)]
        
        if out_mono:
            commands_pianoteq.append("--mono")
        
        commands_pianoteq.append("--headless")
        
        if self.verbose > 0:
            print("Processing MIDI with Pianoteq...")
        try:
            out = subprocess.check_output(commands_pianoteq)
            if self.verbose > 0:
                print("PIANOTEQ OUTPUT:\n", out.decode("utf-8") )
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd,
                                                                                     e.returncode,
                                                                                     e.output))
        
        
    def play_midi(self, midi_path, autoplay=True):
        """Play MIDI file via Pianoteq in Jupyter Notebook.
        
        The time until playback mostly depends on the connection speed
        to the host machine of Jupyter Notebook server.
        """
        
        tmp_mp3_path = "./tmp_pianoteq/tmp_audio.mp3"
        
        self.create_audio(midi_path,tmp_mp3_path)
        
        if self.verbose > 0:
            print("Processing done. Loading audio...")
        
        display(IPython.display.Audio(tmp_mp3_path, autoplay=autoplay))
        
