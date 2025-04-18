import os
import argparse
import functools
import numpy as np
import tensorflow.compat.v2 as tf
import gin
import jax
import librosa
import note_seq
import seqio
import t5
import t5x

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies

SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'


class InferenceModel(object):

    def __init__(self, checkpoint_path, model_type='mt3'):

        # Model Constants
        if model_type == 'ismir2021':
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == 'mt3':
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError('unknown model_type: %s' % model_type)

        gin_files = [
                    './content/gin/model.gin',
                    f'./content/gin/{model_type}.gin'
        ]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {'inputs': self.inputs_length,
                                'targets': self.outputs_length}

        self.partitioner = t5x.partitioning.PjitPartitioner(
            num_partitions=1)


        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=num_velocity_bins))
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            'targets': seqio.Feature(vocabulary=self.vocabulary),
        }

        # Create a T5X model.
        self._parse_gin(gin_files)
        self.model = self._load_model()

        # Restore from checkpoint.
        self.restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self):
        return {
              'encoder_input_tokens': (self.batch_size, self.inputs_length),
              'decoder_input_tokens': (self.batch_size, self.outputs_length)
        }

    def _parse_gin(self, gin_files):
        """Parse gin files used to train the model."""
        gin_bindings = [
            'from __gin__ import dynamic_registration',
            'from mt3 import vocabularies',
            'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
            'vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS'
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                gin_files, gin_bindings, finalize_config=False)

    def _load_model(self):
        """Load up a T5X `Model` after parsing training gin config."""
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features['inputs'].vocabulary,
            output_vocabulary=self.output_features['targets'].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config))

    def restore_from_checkpoint(self, checkpoint_path):
        """Restore training state from checkpoint, resets self._predict_fn()."""
        train_state_initializer = t5x.utils.TrainStateInitializer(
          optimizer_def=self.model.optimizer_def,
          init_fn=self.model.get_initial_variables,
          input_shapes=self.input_shapes,
          partitioner=self.partitioner)

        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode='specific', dtype='float32')

        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        """Generate a partitioned prediction function for decoding."""
        def partial_predict_fn(params, batch, decode_rng):
          return self.model.predict_batch_with_aux(
              params, batch, decoder_params={'decode_rng': None})
        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec('data',), None),
            out_axis_resources=t5x.partitioning.PartitionSpec('data',)
        )

    def predict_tokens(self, batch, seed=0):
        """Predict tokens from preprocessed dataset batch."""
        prediction, _ = self._predict_fn(
            self._train_state.params, batch, jax.random.PRNGKey(seed))
        return self.vocabulary.decode_tf(prediction).numpy()

    def __call__(self, audio):
        """Infer note sequence from audio samples.

        Args:
          audio: 1-d numpy array of audio samples (16kHz) for a single example.

        Returns:
          A note_sequence of the transcribed audio.
        """
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)

        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length)
        model_ds = model_ds.batch(self.batch_size)

        inferences = (tokens for batch in model_ds.as_numpy_iterator()
                      for tokens in self.predict_tokens(batch))

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
          predictions.append(self.postprocess(tokens, example))

        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec)
        return result['est_ns']

    def audio_to_dataset(self, audio):
        """Create a TF Dataset of spectrograms from input audio."""
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({
            'inputs': frames,
            'input_times': frame_times,
        })

    def _audio_to_frames(self, audio):
        """Compute spectrogram frames from audio."""
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode='constant')
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key='inputs',
                additional_feature_keys=['input_times']),
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=self.spectrogram_config)
        ]
        for pp in pp_chain:
          ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = self._trim_eos(tokens)
        start_time = example['input_times'][0]
        # Round down to nearest symbolic token step.
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {
            'est_tokens': tokens,
            'start_time': start_time,
            # Internal MT3 code expects raw inputs, not used here.
            'raw_inputs': []
        }

    @staticmethod
    def _trim_eos(tokens):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
          tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens


def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    """Load audio file using librosa."""
    print(f"Loading audio file: {file_path}")
    y, sr = librosa.load(file_path, sr=sample_rate)
    return y


def transcribe_audio(audio_path, model_type, checkpoint_path, output_midi_path):
    """Transcribe audio file to MIDI."""
    # Load the model
    print(f"Loading {model_type} model from checkpoint: {checkpoint_path}")
    inference_model = InferenceModel(checkpoint_path, model_type)
    
    # Load audio
    audio = load_audio_file(audio_path)
    
    # Transcribe
    print("Transcribing audio... (this may take a few minutes)")
    est_ns = inference_model(audio)
    
    # Save MIDI file
    print(f"Saving transcription to: {output_midi_path}")
    note_seq.sequence_proto_to_midi_file(est_ns, output_midi_path)
    
    # Statistics
    num_notes = sum(1 for note in est_ns.notes if not note.is_drum)
    num_drum_notes = sum(1 for note in est_ns.notes if note.is_drum)
    num_programs = len(set(note.program for note in est_ns.notes if not note.is_drum))
    
    print(f"Transcription complete.")
    print(f"Number of notes: {num_notes}")
    print(f"Number of drum notes: {num_drum_notes}")
    print(f"Number of unique programs/instruments: {num_programs}")
    
    return est_ns


def main():
    parser = argparse.ArgumentParser(description='MT3 Audio Transcription')
    parser.add_argument('--audio_path', required=True, help='Path to audio file to transcribe')
    parser.add_argument('--model_type', default='mt3', choices=['ismir2021', 'mt3'],
                        help='Model type: "ismir2021" for piano only with velocities, "mt3" for multi-instrument')
    parser.add_argument('--checkpoint_path', required=True, help='Path to model checkpoint directory')
    parser.add_argument('--output_midi', default='transcribed.mid', help='Output MIDI file path')
    
    args = parser.parse_args()
    
    transcribe_audio(
        audio_path=args.audio_path,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        output_midi_path=args.output_midi
    )


if __name__ == "__main__":
    main()
