import sys
import os
import logging
import json

from PyQt6.QtCore import Qt, QTimer, QObject, QEvent, qInstallMessageHandler, QPointF
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel, QGraphicsView, QGraphicsLineItem, QMessageBox, QHBoxLayout
from PyQt6.QtGui import QShortcut, QKeySequence, QPainter, QKeyEvent, QPen, QColor, QCursor

from .textbubbles import TextBubbleScene, TextBubble
from . import ui_helpers
from . import io
from .ui_helpers import Keys, measure

from . import spectogram

import soundfile
import sounddevice
import pylibrb
from collections import defaultdict

logger = logging.getLogger('toneswiper')
measurer = logging.getLogger('measurer')

# TODO: Currently abusing return values for measurer logs... Better with getters?


class AudioPlayer(QObject):
    """
    Wraps QMediaPlayer, mainly to facilitate displaying a more smoothly moving progress bar (because at least
    some audio back-ends may update position only once every 50-100ms), by virtue of self.estimate_current_position,
    as well as a delayed progress bar, by virtue of self.get_delayed_position
    """

    # UI things
    SEEK_STEP_FRAMES = 24000

    GHOST_DELAY_FRAMES = 8000
    GHOST_DELAY_DELTA_FRAMES = 2000
    GHOST_COUNTDOWN_PERIOD = 3

    PLAYBACK_RATE = 1
    PLAYBACK_RATE_DELTA = .1
    CHUNK_SIZE = 4096  # smaller gives more clicks; bigger is slower and less reactive

    MAX_PLAYBACK_RATE = 1.0
    MIN_PLAYBACK_RATE = 0.4

    def __init__(self):
        """
        Instantiates the audio player, connects it to the audio output, and sets up
        bookkeeping attributes for estimation of current position.

        If getposition_interval_ms is specified, will sync the queue with calls to estimate_current_position.
        """
        super().__init__()

        # Bookkeeping to extrapolate current position for a smoother moving progress bar:
        self.last_pos_played = 0
        self.time_of_last_pos = None  # for latency
        self.current_pos_consumed = 0
        self.current_pos_played = 0
        self.will_end_in_frames = None

        self.audiostream = None
        self.audio_in = None
        self.sample_rate = None
        self.n_channels = 1
        self.audio_data = None
        self.is_playing = False
        self.duration_ms = None
        self.n_frames = None
        self.sample_rate_ms = None

        # Also store a 'ghost' of the playback position, with some delay
        self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        self.decreasing_ghost_delay = False
        self.ghost_countdown = QTimer(self)
        self.ghost_countdown.setTimerType(Qt.TimerType.PreciseTimer)
        self.ghost_countdown.setInterval(self.GHOST_COUNTDOWN_PERIOD)
        self.ghost_countdown.timeout.connect(self.manage_ghost_delay)
        self.ghost_countdown.start()

    def load_file(self, path: str) -> None:
        """
        Loads an audio file, setting up the rubberband stretcher and audiostream
        """
        self.is_playing = False
        if self.audiostream:
            self.audiostream.stop()
            self.audiostream.close()

        self.audio_data, self.sample_rate = soundfile.read(path)
        self.n_frames = len(self.audio_data)
        self.sample_rate_ms = self.sample_rate / 1000
        self.duration_ms = int((self.n_frames / self.sample_rate_ms))
        self.n_channels = self.audio_data.shape[1] if len(self.audio_data.shape) > 1 else 1

        self.stretcher = pylibrb.RubberBandStretcher(
            sample_rate=self.sample_rate,
            channels=self.n_channels,
            options=pylibrb.Option.PROCESS_REALTIME | pylibrb.Option.ENGINE_FINER,
        )
        self.stretcher.time_ratio = self.PLAYBACK_RATE
        self.audio_in = pylibrb.create_audio_array(channels_num=self.n_channels, samples_num=self.CHUNK_SIZE)

        self.will_end_in_frames = None
        self.decreasing_ghost_delay = False
        self.actual_ghost_delay = self.GHOST_DELAY_FRAMES

        self.audiostream = sounddevice.OutputStream(
            samplerate=self.sample_rate,
            channels=self.n_channels,
            callback=self.audio_callback,
            blocksize=self.CHUNK_SIZE,
        )

        self.set_position(0)

        logger.debug(f'Audio loaded: {self.n_frames=} ({self.sample_rate=}; {self.n_channels})')

    def audio_callback(self, outdata, n_frames, time, status):
        """
        As called by a sounddevice.OutputStream().
        """
        if self.current_pos_played >= len(self.audio_data):
            outdata.fill(0)
            return
        self.stretcher.time_ratio = 1/self.PLAYBACK_RATE

        chunk = self.audio_data[self.current_pos_consumed: self.current_pos_consumed + n_frames]
        # TODO By always taking n_frames, this sometimes overflows stretcher's buffer,
        #   but self.stretcher.get_samples_required() is consistently too low to rely on...

        n_input_frames = len(chunk)
        self.audio_in[:, :len(chunk)] = chunk
        self.audio_in[:, len(chunk):] = 0

        self.stretcher.process(self.audio_in, final=False)
        audio_stretched = self.stretcher.retrieve(n_frames).transpose()

        n_output_frames = min(len(audio_stretched), len(outdata))
        outdata[:n_output_frames, :] = audio_stretched[:n_output_frames, :]
        if n_output_frames < len(outdata):
            outdata[n_output_frames:, :] = 0

        self.last_pos_played = self.current_pos_played
        self.time_of_last_pos = time.outputBufferDacTime * 1000
        self.current_pos_consumed += n_input_frames
        self.current_pos_played += n_output_frames * (self.PLAYBACK_RATE if self.PLAYBACK_RATE < 1.0 else 1.0)  # n_output_frames only goes BELOW, never ABOVE chunk size

        latency_ms = (time.outputBufferDacTime * 1000 - self.audiostream.time * 1000)

        logger.debug(f'Playback: {self.current_pos_consumed=:.0f}, {self.current_pos_played=:.0f} ({latency_ms=}; total {self.n_frames=})')

        # project when the end of audio will be (for ghost line position):
        if self.current_pos_played >= self.n_frames and not self.last_pos_played >= self.n_frames:
            self.will_end_in_frames = int(self.n_frames - self.last_pos_played + self.ms_to_frame(latency_ms))
            logger.debug(f'Projecting audio end: {self.will_end_in_frames=}')

    def set_position(self, frame):
        """
        Called to change the current position in the audio signal (frame), doing bookkeeping mostly to make sure
        that position estimation and audio stretching keep working.
        """
        logger.debug(f'setPosition {frame}')
        self.current_pos_consumed = self.current_pos_played = int(min(max(0, frame), self.n_frames))
        self.stretcher.reset()
        self.last_pos_played = self.current_pos_consumed
        self.time_of_last_pos = self.audiostream.time * 1000

    def estimate_current_position(self) -> int:
        """
        Estimates current position from last position and time_of_last_position.
        (Because at least some audio back-ends update position only once every 50-100ms.)
        """
        if self.is_playing and self.time_of_last_pos is not None:
            delta = self.audiostream.time * 1000 - self.time_of_last_pos
        else:
            delta = 0
        estimated_position = self.last_pos_played + self.ms_to_frame(delta)
        estimated_position = int(max(min(estimated_position, self.n_frames), 0))
        return estimated_position

    def ms_to_frame(self, ms):
        """
        Convert a duration in ms of audio to the corresponding number of audio frames, taking
        playback rate into account.
        """
        return int(ms * self.sample_rate_ms) * self.PLAYBACK_RATE

    def frame_to_original_ms(self, frame):
        """
        Converts a specific audio frame to the original audio ms position, disregarding playback rate.
        This enables mapping current position back to absolute time stamps for the annotations.
        """
        return int(frame / self.sample_rate_ms)

    def frame_to_ms(self, frame):
        """
        Converts number of frames to number of milliseconds, taking playback rate into account.
        """
        return int(frame / self.sample_rate_ms) / self.PLAYBACK_RATE

    def manage_ghost_delay(self):
        """
        The 'ghost' position always has the same delay behind the playback position, unless the playback position reaches
        the end of the audio; then it starts to slowly catch up, with rate depending on the playback rate.
        This function is repeatedly called on a tight timer.
        """
        if self.is_playing:
            if self.will_end_in_frames is not None:
                if self.will_end_in_frames > 0:
                    self.will_end_in_frames -= self.GHOST_COUNTDOWN_PERIOD * self.PLAYBACK_RATE * self.sample_rate_ms
                if self.will_end_in_frames <= 0:
                    self.set_position(self.n_frames)
                    self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
                    self.decreasing_ghost_delay = True
                    self.will_end_in_frames = None
                    logger.debug(f'Starting ghost catchup: {self.actual_ghost_delay=}')
            if self.decreasing_ghost_delay:
                self.actual_ghost_delay = int(self.actual_ghost_delay - self.ms_to_frame(self.GHOST_COUNTDOWN_PERIOD))
                if self.actual_ghost_delay <= 0:
                    logger.debug(f'Ghost caught up!')
                    self.decreasing_ghost_delay = False
                    self.pause()

    def pause(self):
        if self.audiostream:
            pos = self.estimate_current_position()
            self.audiostream.stop()
            self.set_position(pos)
            self.is_playing = False

    # Remainder of this class are functions for handling the various hotkeys:
    @measure
    def decrease_ghost_delay(self):
        self.GHOST_DELAY_FRAMES = max(0, self.GHOST_DELAY_FRAMES - self.GHOST_DELAY_DELTA_FRAMES)
        if not self.decreasing_ghost_delay:
            self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        return self.frame_to_ms(self.GHOST_DELAY_FRAMES)

    @measure
    def increase_ghost_delay(self):
        self.GHOST_DELAY_FRAMES = min(self.GHOST_DELAY_FRAMES + self.GHOST_DELAY_DELTA_FRAMES, self.n_frames)
        if not self.decreasing_ghost_delay:
            self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        return self.frame_to_ms(self.GHOST_DELAY_FRAMES)

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.time_of_last_pos = None
            if self.current_pos_consumed == self.n_frames:
                if self.actual_ghost_delay <= 0:
                    self.set_position(0)
                    self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
                    self.decreasing_ghost_delay = False
            self.is_playing = True
            self.audiostream.start()

    @measure
    def toggle_play_pause_manual(self):
        self.toggle_play_pause()
        return self.frame_to_original_ms(self.estimate_current_position())

    @measure
    def seek_forward(self):
        self.seek_position_relative(self.SEEK_STEP_FRAMES)
        return self.frame_to_original_ms(self.estimate_current_position())

    @measure
    def seek_backward(self):
        self.seek_position_relative(-self.SEEK_STEP_FRAMES)
        return self.frame_to_original_ms(self.estimate_current_position())

    @measure
    def seek_home(self):
        self.will_end_in_frames = None
        self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        self.decreasing_ghost_delay = False
        self.set_position(0)
        return 0

    @measure
    def seek_end(self):
        self.will_end_in_frames = None
        self.actual_ghost_delay = 0
        self.decreasing_ghost_delay = False
        self.set_position(self.n_frames)
        self.pause()
        return self.n_frames

    def seek_position_relative(self, delta_frames: int) -> None:
        self.seek_position_absolute(self.estimate_current_position() + delta_frames)

    def seek_position_absolute(self, frame: int) -> None:
        is_playing = self.is_playing
        self.pause()
        self.will_end_in_frames = None
        self.decreasing_ghost_delay = False
        self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        self.set_position(frame)
        if is_playing:
            self.toggle_play_pause()

    @measure
    def increase_playback_rate(self):
        if self.PLAYBACK_RATE >= self.MAX_PLAYBACK_RATE:
            return
        was_playing = self.is_playing
        self.current_pos_consumed = self.estimate_current_position()
        self.pause()
        self.PLAYBACK_RATE = min(self.MAX_PLAYBACK_RATE, self.PLAYBACK_RATE + self.PLAYBACK_RATE_DELTA)
        self.stretcher.reset()
        if was_playing:
            self.toggle_play_pause()
        return self.PLAYBACK_RATE

    @measure
    def decrease_playback_rate(self):
        if self.PLAYBACK_RATE <= self.MIN_PLAYBACK_RATE:
            return
        was_playing = self.is_playing
        self.current_pos_consumed = self.estimate_current_position()
        self.pause()
        self.PLAYBACK_RATE = max(self.MIN_PLAYBACK_RATE, self.PLAYBACK_RATE - self.PLAYBACK_RATE_DELTA)
        self.stretcher.reset()
        if was_playing:
            self.toggle_play_pause()
        return self.PLAYBACK_RATE

    def __str__(self):
        return "AudioPlayer"

class TranscriptionPanel(QGraphicsView):
    """
    Completing the PyQt graphics hierarchy of a view, a scene, and the graphic objects it contains, in this case
    text bubbles representing annotations.
    Overrides scrolling behavior; passes keypresses down to the containing TextBubble objects (when focused).
    """

    PX_PER_S = 180
    PADDING = 30  # TODO Not quite what I need... some text bubbles fall off the edge.

    PROGRESS_LINE_PEN = QPen(QColor(255, 255, 255, 150), 3)
    TRANSCRIPTION_LINE_PEN = QPen(QColor(100, 255, 255, 150), 3)
    CURSOR_LINE_PEN = QPen(QColor(255, 255, 0, 150), 1)

    REFRESH_TIME = 12

    def __init__(self, audioplayer):
        """
        Sets a new TextBubbleScene as its viewed scene, prepares for spectogram and moving
        vertical bars.
        """
        scene = self.text_bubble_scene = TextBubbleScene()
        super().__init__(scene)

        self.setViewportMargins(self.PADDING, 10, self.PADDING, 10)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.audioplayer = audioplayer
        self.spectogram = None

        # Lines that will be moving:
        self.playback_bar = QGraphicsLineItem()
        self.ghost_bar = QGraphicsLineItem()
        self.cursor_bar = QGraphicsLineItem()

        self.playback_bar.setPen(self.PROGRESS_LINE_PEN)
        self.ghost_bar.setPen(self.TRANSCRIPTION_LINE_PEN)
        self.cursor_bar.setPen(self.CURSOR_LINE_PEN)

        scene.addItem(self.playback_bar)
        scene.addItem(self.ghost_bar)
        scene.addItem(self.cursor_bar)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.REFRESH_TIME)

        # To override scrolling behavior:
        self.setHorizontalScrollBar(ui_helpers.InterceptingScrollBar(Qt.Orientation.Horizontal, self, self.scroll_bar_interceptor))
        self._is_scrolling_from_code = False

    def load_visualisation_for_audio(self, path: str, duration: float):
        """
        Sets up the scene width according to the duration, and passes the path into parselmouth for a spectogram.
        Spectograms are cached.
        """
        width_px = int(duration * (self.PX_PER_S / 1000))
        self.scene().setSceneRect(0, 0, width_px, 400)

        if self.spectogram is not None:
            self.scene().removeItem(self.spectogram)
        self.spectogram = spectogram.make_image_cached(path, width_px, 200)
        self.spectogram.setPos(0, 0)
        self.scene().addItem(self.spectogram)

        self.playback_bar.setZValue(1)
        self.ghost_bar.setZValue(1)
        self.cursor_bar.setZValue(2)

        # TODO: The following better handled through a signal? And what if audio not yet ready?
        if not self.audioplayer.is_playing and self.audioplayer.audiostream is not None:
            self.audioplayer.toggle_play_pause()

    def refresh(self):
        """
        To be called every REFRESH_TIME ms, scrolls to current position and updates all vertical bars.
        """
        if not self.spectogram:
            return

        rect = self.spectogram.boundingRect()

        playback_x = rect.left() + self.frame_to_pix(self.audioplayer.estimate_current_position())
        delayed_x = max(0, playback_x - self.frame_to_pix(self.audioplayer.actual_ghost_delay))

        self.playback_bar.setLine(playback_x, rect.top(), playback_x, rect.bottom())
        self.ghost_bar.setLine(delayed_x, rect.top(), delayed_x, rect.bottom())

        self._is_scrolling_from_code = True
        self.centerOn(QPointF(playback_x, 0))
        self._is_scrolling_from_code = False

        global_cursor_pos = QCursor.pos()  # checked AFTER scrolling
        local_cursor_pos = self.mapFromGlobal(global_cursor_pos)
        cursor_x = local_cursor_pos.x() - self.mapFromScene(self.sceneRect().left(), 0).x() - self.PADDING
        self.cursor_bar.setLine(cursor_x, rect.top(), cursor_x, rect.bottom())

    def keyPressEvent(self, event: QKeyEvent):
        """
        Pass keypresses on to any focused child. Ignore otherwise, in order for the main window to process the
        keypresses instead.
        """
        if any(item.hasFocus() for item in self.scene().items()):
            super().keyPressEvent(event)
        else:
            event.ignore()

    # Next: a bunch of functions to override scrolling behavior, routing it through the audioplayer and self.refresh.
    def centerOn(self, pos):
        if self._is_scrolling_from_code:
            super().centerOn(pos)
        else:
            self.manualCenterOn(pos)

    def scrollContentsBy(self, dx, dy):
        if self._is_scrolling_from_code:
            super().scrollContentsBy(dx, dy)
            self.viewport().update()
        else:
            self.manualScrollContentsBy(dx, dy)

    @measure
    def manualCenterOn(self, pos):
        self.audioplayer.seek_position_absolute(self.pix_to_frame(pos))
        return self.audioplayer.estimate_current_position()

    @measure
    def manualScrollContentsBy(self, dx, dy):
        x = self.frame_to_pix(self.audioplayer.estimate_current_position()) - dx  # minus because wrt to CONTENT
        self.audioplayer.seek_position_absolute(self.pix_to_frame(x))
        return self.audioplayer.estimate_current_position()

    def wheelEvent(self, event):
        # if event.modifiers() & Qt.KeyboardModifier.ControlModifier:  # TODO maybe for zoom?
        delta = event.angleDelta().y()
        self.scrollContentsBy(delta, 0)
        event.accept()
        self.register_wheel_event()

    @measure
    def register_wheel_event(self):
        return self.audioplayer.estimate_current_position()

    @measure
    def scroll_bar_interceptor(self, dx):
        x = self.frame_to_pix(self.audioplayer.estimate_current_position()) + dx
        self.audioplayer.set_position(self.pix_to_frame(x))
        return self.audioplayer.estimate_current_position()
    # End of scroll-overriding functions

    def get_annotations(self) -> list[TextBubble]:
        return self.scene().list_all_text_bubbles()

    def add_annotations(self, annotations: list[tuple[float, str]]):
        for time, text in annotations:
            self.text_bubble_scene.new_item_relx(time / self.audioplayer.duration_ms, text)

    @measure
    def remove_all_annotations(self):
        if len(self.get_annotations()) > 10 and QMessageBox.question(self, "Confirm Deletion", f"This will remove all {len(self.get_annotations())} annotations for this audio file. Are you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
            return

        for bubble in self.get_annotations():
            bubble.scene().removeItem(bubble)

    @measure
    def remove_last_annotation(self):
        bubbles = self.get_annotations()
        if bubbles:
            last_bubble = bubbles.pop(0)
            last_bubble.scene().removeItem(last_bubble)

    def frame_to_pix(self, pos):
        """
        Converts audio frame to pixel position in the scene.
        """
        pix_per_frame = self.PX_PER_S / self.audioplayer.sample_rate
        return int(pos * pix_per_frame)

    def pix_to_frame(self, pix):
        """
        Converts pixel position in the scene to audio frame.
        """
        pix_per_frame = self.PX_PER_S / self.audioplayer.sample_rate
        return int(pix / pix_per_frame)

    @measure
    def add_transcription(self, time_ms, text):
        self.text_bubble_scene.new_item_relx(time_ms, text)

    def __str__(self):
        return "TranscriptionPanel"


class CurrentlyPressedKeysTracker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pressed_keys = set()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat() and not event.key() in ui_helpers.Keys.DOWNSTEP | ui_helpers.Keys.UNCERTAIN:
            self.pressed_keys.add(event.key())
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self.pressed_keys.discard(event.key())
        return False  # don't filter out, for processing further down


class ToneSwiperWindow(QMainWindow):
    """
    Main window of the app, wrapping an AudioPlayer, Audioviewer and TextBubbleSceneView.
    Handles most keyboard controls for transcription and audioplayer.
    """

    COYOTE_TIME = 50

    def __init__(self, wavfiles: list[str], save_as_textgrids: str = None, save_as_json: str = None):
        """
        Takes a list of .wav files to be annotated, and optionally where to load/save annotations from
        (textgrids or json). Sets up window layout, loads the first sound file, and sets up some
        bookkeeping for registering key sequences.
        """
        super().__init__()
        self.wavfiles = wavfiles
        self.stored_durations = defaultdict(int)  # only for writing textgrids in the end
        self.save_as_textgrid_tier = save_as_textgrids
        self.save_as_json = save_as_json

        # For registering ToDI transcription key sequences
        self.current_key_sequence = []
        self.current_key_sequence_time = None
        self.currently_pressed_keys_tracker = CurrentlyPressedKeysTracker()
        QApplication.instance().installEventFilter(self.currently_pressed_keys_tracker)

        # Loading/saving
        self.transcriptions = [[] for _ in self.wavfiles]
        if self.save_as_json and os.path.exists(self.save_as_json):
            from_json = io.load_from_json(self.save_as_json)
            self.transcriptions = [from_json.get(filename, []) for filename in self.wavfiles]
        if self.save_as_textgrid_tier:
            from_textgrids = io.load_from_textgrids(self.wavfiles, self.save_as_textgrid_tier)
            self.transcriptions = [from_textgrids.get(wavfile, []) for wavfile in self.wavfiles]

        # Window layout
        self.setWindowTitle('ToneSwiper')
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)
        topwidget = QWidget()
        top_layout = QHBoxLayout(topwidget)
        layout.addWidget(topwidget)

        self.filename_label = QLabel('', self)
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QApplication.font()
        font.setPointSize(14)
        self.filename_label.setFont(font)
        top_layout.addStretch(1)
        top_layout.addWidget(self.filename_label)

        self.playback_rate_label = QLabel('⏱ 1.0 ×', self)
        self.playback_rate_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        font = QApplication.font()
        font.setPointSize(14)
        self.playback_rate_label.setFont(font)
        top_layout.addStretch(1)
        top_layout.addWidget(self.playback_rate_label)

        self.audioplayer = AudioPlayer()
        self.transcription_panel = TranscriptionPanel(self.audioplayer)
        layout.addWidget(self.transcription_panel)

        self.current_file_index = None
        self.transcription_loaded = False
        self.load_sound_by_index(0)

    @measure
    def load_sound_by_index(self, idx: int) -> str:
        """
        For an index to a sound file, this function stores current annotations in memory, loads and plays
        the requested sound file, and (re)loads the corresponding audioviewer (spectogram) and transcription
        panels (the latter only once the audio's duration is known).

        Returns the sound filename for logging convenience only.
        """
        n_files = len(self.wavfiles)
        idx = idx % n_files

        if idx == self.current_file_index:
            return

        if self.transcription_loaded:  # also crucial in case of next/prev before it gets fully loaded
            self.transcriptions[self.current_file_index] = [(b.relative_x * self.audioplayer.duration_ms, b.toPlainText()) for b in self.transcription_panel.get_annotations()]
            for item in self.transcription_panel.get_annotations():
                item.scene().removeItem(item)
        self.transcription_loaded = False

        self.current_file_index = idx
        current_wav_path = self.wavfiles[idx]

        filename_label_str = f"📁 {current_wav_path} ({self.current_file_index + 1}/{n_files})"
        if self.save_as_textgrid_tier:
            filename_label_str += f', tier \'{self.save_as_textgrid_tier}\''
        self.filename_label.setText(filename_label_str)

        self.audioplayer.pause()
        self.audioplayer.load_file(current_wav_path)
        self.stored_durations[current_wav_path] = duration = self.audioplayer.duration_ms

        self.transcription_panel.load_visualisation_for_audio(current_wav_path, duration=duration)
        self.transcription_panel.add_annotations(self.transcriptions[self.current_file_index])
        self.transcription_loaded = True

        return current_wav_path

    def keyPressEvent(self, event):
        """
        Handles most keyboard inputs, as defined in the Keys class, for controlling the audioplayer and
        for making the annotations.
        """
        key = event.key()

        if event.isAutoRepeat() and key not in Keys.FORWARD | Keys.BACKWARD:
            return

        if key in Keys.PAUSE:
            self.audioplayer.toggle_play_pause_manual()
            return
        elif key in Keys.FORWARD:
            self.audioplayer.seek_forward()
        elif key in Keys.BACKWARD:
            self.audioplayer.seek_backward()
        elif key in Keys.MOREDELAY:
            self.audioplayer.increase_ghost_delay()
        elif key in Keys.LESSDELAY:
            self.audioplayer.decrease_ghost_delay()
        elif key in Keys.SLOWER:
            self.audioplayer.decrease_playback_rate()
            self.playback_rate_label.setText(f'⏱ {self.audioplayer.PLAYBACK_RATE:.1f} ×')
        elif key in Keys.FASTER:
            self.audioplayer.increase_playback_rate()
            self.playback_rate_label.setText(f'⏱ {self.audioplayer.PLAYBACK_RATE:.1f} ×')
        elif key in Keys.NEXT or (key == Qt.Key.Key_Right and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.next()
        elif key in Keys.PREVIOUS or (key == Qt.Key.Key_Left and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.prev()
        elif key in Keys.HOME:
            self.audioplayer.seek_home()
        elif key in Keys.END:
            self.audioplayer.seek_end()
        elif key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.transcription_panel.remove_last_annotation()
        elif key == Qt.Key.Key_X and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.transcription_panel.remove_all_annotations()
        elif key in Keys.TODI_KEYS:
            self.current_key_sequence.append(key)
            if Qt.Key.Key_Control in Keys.DOWNSTEP and key != Qt.Key.Key_Control and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.current_key_sequence.append(Qt.Key.Key_Control)
            elif Qt.Key.Key_Shift in Keys.UNCERTAIN and key != Qt.Key.Key_Shift and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.current_key_sequence.append(Qt.Key.Key_Shift)
            if key not in Keys.DOWNSTEP | Keys.UNCERTAIN:
                self.current_key_sequence_time = self.audioplayer.frame_to_original_ms(self.audioplayer.estimate_current_position() - self.audioplayer.actual_ghost_delay)
                if self.audioplayer.is_playing:
                    self.current_key_sequence_time += int(self.COYOTE_TIME * self.audioplayer.PLAYBACK_RATE)
        else:
            self.current_key_sequence = []
            self.current_key_sequence_time = None

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """
        Key sequences are built up until no more keys are currently pressed. Then they are released and
        turned into a transcription, which if feasible results in a new text bubble in the transcription panel.
        """
        if event.isAutoRepeat():
            return

        if self.currently_pressed_keys_tracker.pressed_keys:  # Sequence not yet completed
            return

        if self.current_key_sequence and self.current_key_sequence_time:
            try:
                transcription = ui_helpers.key_sequence_to_transcription(self.current_key_sequence)
            except ValueError as e:
                logger.warning(e)
            else:
                transcription_time = self.current_key_sequence_time / self.audioplayer.duration_ms
                self.transcription_panel.add_transcription(transcription_time, transcription)

        self.current_key_sequence = []
        self.current_key_sequence_time = None


    @measure
    def next(self):
        """
        Go to next audio file (modulo-ized).
        """
        self.load_sound_by_index((self.current_file_index + 1) % len(self.wavfiles))

    @measure
    def prev(self):
        """
        Go to previous audio file (modulo-ized).
        """
        self.load_sound_by_index((self.current_file_index - 1) % len(self.wavfiles))

    def closeEvent(self, event):
        """
        Upon closing the window, current transcription bubbles are stored in memory,
        and all transcriptions in memory are then saved either as a textgrid, or as json.
        """
        self.audioplayer.pause()
        self.save()
        event.accept()

    @measure
    def save(self):
        self.transcriptions[self.current_file_index] = [(b.relative_x * self.audioplayer.duration_ms, b.toPlainText())
                                                        for b in self.transcription_panel.get_annotations()]
        for_json = {str(file): transcription for file, transcription in zip(self.wavfiles, self.transcriptions)}

        if self.save_as_textgrid_tier:
            io.write_to_textgrids(self.transcriptions,
                                  self.wavfiles,
                                  self.stored_durations,
                                  self.save_as_textgrid_tier)
        else:
            io.write_to_json(for_json, to_file=self.save_as_json)
        return for_json

    def __str__(self):
        return "ToneSwiperWindow"


def main():
    """
    Starts the PyQt6 app and main window, and calls upon various ui_helpers for intercepting tab/shift+tab,
    mouse movements, mute some log messages, and sets up F1 for help window.
    """
    args = ui_helpers.parse_args()

    ui_helpers.setup_logging(verbose=args.verbose, measure=args.measure)

    measurer.info(json.dumps({"action": "main", "args": args.file, "kwargs": {k: v for k, v in args.__dict__.items() if k != "file"}}))

    app = QApplication(sys.argv)
    app.setStyle('fusion')
    icon = ui_helpers.load_icon()

    qInstallMessageHandler(ui_helpers.custom_message_handler)

    window = ToneSwiperWindow(args.file, save_as_textgrids=args.textgrid, save_as_json=args.json)

    sys.excepthook = ui_helpers.exception_hook

    app.setWindowIcon(icon)
    window.setWindowIcon(icon)

    tab_interceptor = ui_helpers.TabInterceptor(window.transcription_panel.text_bubble_scene.handle_tabbing)
    app.installEventFilter(tab_interceptor)

    help_box = ui_helpers.HelpOverlay(window)
    QShortcut(QKeySequence("F1"), window, activated=help_box.display_panel)
    screen_geom = QApplication.primaryScreen().availableGeometry()
    help_box.move(screen_geom.right() - help_box.width(), screen_geom.top())

    window.resize(1200, 600)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
