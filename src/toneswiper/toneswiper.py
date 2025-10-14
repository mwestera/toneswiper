import sys
import os
import logging

from PyQt6.QtCore import Qt, QTimer, QObject, QEvent, qInstallMessageHandler, QPointF
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel, QGraphicsView, QGraphicsLineItem, \
    QMessageBox, QHBoxLayout
from PyQt6.QtGui import QShortcut, QKeySequence, QPainter, QKeyEvent, QPen, QColor, QCursor

from .textbubbles import TextBubbleScene, TextBubble
from . import ui_helpers
from . import io
from .ui_helpers import Keys

from . import spectogram

import soundfile
import sounddevice
import numpy as np
import pylibrb
from collections import defaultdict

logger = logging.getLogger('toneswiper')

# TODO: Update docstrings.


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

        # To extrapolate current position for a smoother moving progress bar:
        self.last_position = 0
        self.time_of_last_position = None  # for latency

        self.audiostream = None
        self.audio_in = None
        self.pos_consumed = 0
        self.pos_played = 0
        self.sample_rate = None
        self.n_channels = 1
        self.audio_data = None
        self.is_playing = False
        self.duration_ms = None
        self.will_end_in_frames = None

        self.n_frames = None
        self.sample_rate_ms = None

        # Also store a 'ghost' of the playback position, with some delay
        self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        self.decreasing_ghost_delay = False
        self.ghost_countdown = QTimer(self)
        self.ghost_countdown.setTimerType(Qt.TimerType.PreciseTimer)
        self.ghost_countdown.setInterval(self.GHOST_COUNTDOWN_PERIOD)
        self.ghost_countdown.timeout.connect(self.update_actual_ghost_delay)
        self.ghost_countdown.start()

    def load_file(self, path: str) -> None:
        """
        Loads a file and (by default) starts playing.
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

        self.setPosition(0)

        logger.debug(f'Audio loaded: {self.n_frames=} ({self.sample_rate=}; {self.n_channels})')

    def audio_callback(self, outdata, n_frames, time, status):
        if self.pos_played >= len(self.audio_data):
            outdata.fill(0)
            return

        self.stretcher.time_ratio = 1/self.PLAYBACK_RATE

        n_input_frames = n_frames  # TODO This sometimes overflows stretcher's buffer,
                                   #   but self.stretcher.get_samples_required() is too low, audio breaking...

        chunk = self.audio_data[self.pos_consumed: self.pos_consumed + n_input_frames]

        n_input_frames = len(chunk)

        self.audio_in[:,:len(chunk)] = chunk
        self.audio_in[:,len(chunk):] = 0

        self.stretcher.process(self.audio_in, final=False)
        audio_stretched = self.stretcher.retrieve(n_frames).transpose()

        logger.debug(f'{self.stretcher.get_samples_required()=}, {self.audio_in.shape=}, {audio_stretched.shape=}')

        if audio_stretched.ndim == 1:
            audio_stretched = audio_stretched[:, np.newaxis]

        n_output_frames = min(len(audio_stretched), len(outdata))
        outdata[:n_output_frames, :] = audio_stretched[:n_output_frames, :]
        if n_output_frames < len(outdata):
            outdata[n_output_frames:, :] = 0

        previous_pos_played = self.pos_played

        self.on_position_changed(previous_pos_played, time.outputBufferDacTime * 1000)

        self.pos_consumed += n_input_frames
        self.pos_played += n_output_frames * (self.PLAYBACK_RATE if self.PLAYBACK_RATE < 1.0 else 1.0)  # n_output_frames only goes BELOW, never ABOVE chunk size

        latency_ms = (time.outputBufferDacTime * 1000 - self.audiostream.time * 1000)

        logger.debug(f'Playback: {self.pos_consumed=:.0f}, {self.pos_played=:.0f} ({latency_ms=}; total {self.n_frames=})')

        if self.pos_played >= self.n_frames and not previous_pos_played >= self.n_frames:
            self.will_end_in_frames = int(self.n_frames - previous_pos_played + self.original_ms_to_frame(latency_ms) * self.PLAYBACK_RATE)
            logger.debug(f'Projecting audio end: {self.will_end_in_frames=}')

    def update_actual_ghost_delay(self):
        if self.is_playing:
            if self.will_end_in_frames is not None:
                if self.will_end_in_frames > 0:
                    self.will_end_in_frames -= self.GHOST_COUNTDOWN_PERIOD * self.PLAYBACK_RATE * self.sample_rate_ms
                if self.will_end_in_frames <= 0:
                    self.setPosition(self.n_frames)
                    self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
                    self.decreasing_ghost_delay = True
                    self.will_end_in_frames = None
                    logger.debug(f'Starting ghost catchup: {self.actual_ghost_delay=}')
            if self.decreasing_ghost_delay:
                self.actual_ghost_delay = int(self.actual_ghost_delay - self.original_ms_to_frame(self.GHOST_COUNTDOWN_PERIOD) * self.PLAYBACK_RATE)
                if self.actual_ghost_delay <= 0:
                    logger.debug(f'Ghost caught up!')
                    self.decreasing_ghost_delay = False
                    self.pause()

    def decrease_ghost_delay(self):
        self.GHOST_DELAY_FRAMES = max(0, self.GHOST_DELAY_FRAMES - self.GHOST_DELAY_DELTA_FRAMES)
        if not self.decreasing_ghost_delay:
            self.actual_ghost_delay = self.GHOST_DELAY_FRAMES

    def increase_ghost_delay(self):
        self.GHOST_DELAY_FRAMES = min(self.GHOST_DELAY_FRAMES + self.GHOST_DELAY_DELTA_FRAMES, self.n_frames)
        if not self.decreasing_ghost_delay:
            self.actual_ghost_delay = self.GHOST_DELAY_FRAMES

    def toggle_play_pause(self) -> None:
        """
        For use by play/pause hotkey.
        """
        if self.is_playing:
            self.pause()
        else:
            self.time_of_last_position = None
            if self.pos_consumed == self.n_frames:
                if self.actual_ghost_delay <= 0:
                    self.setPosition(0)
                    self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
                    self.decreasing_ghost_delay = False
            self.is_playing = True
            self.audiostream.start()

    def pause(self):
        if self.audiostream:
            pos = self.estimate_current_position()
            self.audiostream.stop()
            self.setPosition(pos)
            self.is_playing = False

    def skipforward(self):
        self.seek_relative(self.SEEK_STEP_FRAMES)

    def skipbackward(self):
        self.seek_relative(-self.SEEK_STEP_FRAMES)

    def skiphome(self):
        self.will_end_in_frames = None
        self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        self.decreasing_ghost_delay = False
        self.setPosition(0)

    def skipend(self):
        self.will_end_in_frames = None
        self.actual_ghost_delay = 0
        self.decreasing_ghost_delay = False
        self.setPosition(self.n_frames)
        self.pause()

    def seek_relative(self, delta_frames: int) -> None:
        """
        Skips sound player ahead by delta_frames (or back, if negative).
        """
        is_playing = self.is_playing
        self.pause()
        self.will_end_in_frames = None
        self.decreasing_ghost_delay = False
        self.actual_ghost_delay = self.GHOST_DELAY_FRAMES
        self.setPosition(self.estimate_current_position() + delta_frames)
        if is_playing:
            self.toggle_play_pause()

    def setPosition_ms(self, pos_ms):
        self.setPosition(self.original_ms_to_frame(pos_ms))

    def setPosition(self, frame):
        logger.debug(f'setPosition {frame}')
        self.pos_consumed = self.pos_played = int(min(max(0, frame), self.n_frames))
        self.stretcher.reset()
        self.on_position_changed(self.pos_consumed, self.audiostream.time * 1000)

    def speedup(self):
        if self.PLAYBACK_RATE >= self.MAX_PLAYBACK_RATE:
            return
        was_playing = self.is_playing
        self.pos_consumed = self.estimate_current_position()
        self.pause()
        self.PLAYBACK_RATE = min(self.MAX_PLAYBACK_RATE, self.PLAYBACK_RATE + self.PLAYBACK_RATE_DELTA)
        self.stretcher.reset()
        if was_playing:
            self.toggle_play_pause()

    def slowdown(self):
        if self.PLAYBACK_RATE <= self.MIN_PLAYBACK_RATE:
            return
        was_playing = self.is_playing
        self.pos_consumed = self.estimate_current_position()
        self.pause()
        self.PLAYBACK_RATE = max(self.MIN_PLAYBACK_RATE, self.PLAYBACK_RATE - self.PLAYBACK_RATE_DELTA)
        self.stretcher.reset()
        if was_playing:
            self.toggle_play_pause()

    def estimate_current_position(self) -> int:
        """
        Estimates current position from last position and time_of_last_position.
        (Because at least some audio back-ends update position only once every 50-100ms.)
        """
        if self.is_playing and self.time_of_last_position is not None:
            delta = self.audiostream.time * 1000 - self.time_of_last_position
        else:
            delta = 0
        estimated_position = self.last_position + self.original_ms_to_frame(delta) * self.PLAYBACK_RATE
        estimated_position = int(max(min(estimated_position, self.n_frames), 0))
        return estimated_position

    def original_ms_to_frame(self, ms):
        return int(ms * self.sample_rate_ms)

    def original_frame_to_ms(self, frame):
        return int(frame / (self.sample_rate_ms))

    def on_position_changed(self, pos_frame: float, dac_time) -> None:
        """
        Keeping position and timing data for extrapolating, as done in self.best_current_position.
        """
        self.last_position = pos_frame
        self.time_of_last_position = dac_time


class TranscriptionPanel(QGraphicsView):
    """
    Completing the PyQt graphics hierarchy of a view, a scene, and the text bubbles it contains.
    Handles resizeEvent (passed on to scene.resizeEventFromView), and keypresses to the containing
    TextBubble objects (when focused).
    """

    PX_PER_S = 180  # TODO Make zoomable?
    PADDING = 30  # TODO Not quite what I need... some text bubbles fall off the edge.

    def __init__(self, audioplayer):
        """
        Sets a new TextBubbleScene as its viewed scene; sets its width to a specified
        proportion of the view.
        """
        self.text_bubble_scene = TextBubbleScene()
        super().__init__(self.text_bubble_scene)
        self.setViewportMargins(self.PADDING, 10, self.PADDING, 10)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.audioplayer = audioplayer

        self.spectogram = None  # created upon load_file

        self.progress_line = QGraphicsLineItem()
        self.progress_line.setPen(QPen(QColor(255, 255, 255, 150), 3))
        self.scene().addItem(self.progress_line)

        self.transcription_line = QGraphicsLineItem()
        self.transcription_line.setPen(QPen(QColor(100, 255, 255, 150), 3))
        self.scene().addItem(self.transcription_line)

        self.cursor_line = QGraphicsLineItem()
        self.cursor_line.setPen(QPen(QColor(255, 255, 0, 150), 1))
        self.scene().addItem(self.cursor_line)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_vertical_bars)
        self.timer.start(12)

        self.setHorizontalScrollBar(ui_helpers.InterceptingScrollBar(Qt.Orientation.Horizontal, self, self.scrollBarInterceptor))

        self._programmatic_scroll = False

    def load_file(self, path, duration):
        self.audioplayer.spectogram_is_ready = False
        width_px = int(duration * (self.PX_PER_S / 1000))
        self.scene().setSceneRect(0, 0, width_px, 400)
        if self.spectogram is not None:
            self.scene().removeItem(self.spectogram)
        self.spectogram = spectogram.make_image_cached(path, width_px, 200)
        self.spectogram.setPos(0, 0)
        self.scene().addItem(self.spectogram)
        self.progress_line.setZValue(1)
        self.transcription_line.setZValue(1)
        self.cursor_line.setZValue(2)

        # TODO: The following better handled through a signal? And what if audio not yet ready?
        if not self.audioplayer.is_playing and self.audioplayer.audiostream is not None:
            self.audioplayer.toggle_play_pause()

    def centerOn(self, pos):
        if self._programmatic_scroll:
            super().centerOn(pos)
        else:
            self.audioplayer.setPosition(self.pix_to_frame(pos))

    def scrollContentsBy(self, dx, dy):
        if self._programmatic_scroll:
            super().scrollContentsBy(dx, dy)
            self.viewport().update()
        else:
            x = self.frame_to_pix(self.audioplayer.estimate_current_position()) - dx  # minus because wrt to CONTENT
            self.audioplayer.setPosition(self.pix_to_frame(x))

    def scrollBarInterceptor(self, dx):
        x = self.frame_to_pix(self.audioplayer.estimate_current_position()) + dx
        self.audioplayer.setPosition(self.pix_to_frame(x))

    def wheelEvent(self, event):
        # if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
        delta = event.angleDelta().y()
        self.scrollContentsBy(delta, 0)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if any(item.hasFocus() for item in self.scene().items()):
            super().keyPressEvent(event)
        else:
            event.ignore()

    def textBubbles(self):
        return [item for item in self.scene().items() if isinstance(item, TextBubble)]

    def remove_all_bubbles(self):
        if len(self.textBubbles()) > 5 and QMessageBox.question(self, "Confirm Deletion", f"This will remove all {len(self.textBubbles())} annotations for this audio file. Are you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
            return

        for bubble in self.textBubbles():
            bubble.scene().removeItem(bubble)

    def remove_last_added_bubble(self):
        """
        Removes the last added bubble, which happens to be the first one in self.scene.items().
        """
        bubbles = self.textBubbles()
        if bubbles:
            last_bubble = bubbles.pop(0)
            last_bubble.scene().removeItem(last_bubble)

    def update_vertical_bars(self):
        if not self.spectogram:
            return

        position = self.frame_to_pix(self.audioplayer.estimate_current_position())
        rect = self.spectogram.boundingRect()
        x = rect.left() + position
        self.progress_line.setLine(x, rect.top(), x, rect.bottom())

        delayed_x = max(0, x - self.frame_to_pix(self.audioplayer.actual_ghost_delay))
        self.transcription_line.setLine(delayed_x, rect.top(), delayed_x, rect.bottom())

        self._programmatic_scroll = True
        self.centerOn(QPointF(x, 0))
        self._programmatic_scroll = False

        global_cursor_pos = QCursor.pos()
        local_cursor_pos = self.mapFromGlobal(global_cursor_pos)
        cursor_x = local_cursor_pos.x() - self.mapFromScene(self.sceneRect().left(), 0).x() - self.PADDING
        self.cursor_line.setLine(cursor_x, rect.top(), cursor_x, rect.bottom())

    def frame_to_pix(self, pos):
        pix_per_frame = self.PX_PER_S / self.audioplayer.sample_rate
        return int(pos * pix_per_frame)

    def pix_to_frame(self, pix):
        pix_per_frame = self.PX_PER_S / self.audioplayer.sample_rate
        return int(pix / pix_per_frame)


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
        self.save_as_textgrid_tier = save_as_textgrids
        self.save_as_json = save_as_json

        self.stored_durations = defaultdict(int)

        self.currently_pressed_keys_tracker = CurrentlyPressedKeysTracker()
        QApplication.instance().installEventFilter(self.currently_pressed_keys_tracker)

        self.transcriptions = [[] for _ in self.wavfiles]
        if self.save_as_json and os.path.exists(self.save_as_json):
            from_json = io.load_from_json(self.save_as_json)
            self.transcriptions = [from_json.get(filename, []) for filename in self.wavfiles]
            if self.save_as_textgrid_tier:
                logger.warning("Both save_as_json and save_as_textgrid_tier specified;"
                                "will only load from textgrids (but save to both).")
        if self.save_as_textgrid_tier:
            from_textgrids = io.load_from_textgrids(self.wavfiles, self.save_as_textgrid_tier)
            self.transcriptions = [from_textgrids.get(wavfile, []) for wavfile in self.wavfiles]

        self.setWindowTitle('ToneSwiper')
        central = QWidget()
        layout = QVBoxLayout(central)

        self.setCentralWidget(central)

        topwidget = QWidget()
        top_layout = QHBoxLayout(topwidget)
        layout.addWidget(topwidget)

        self.label = QLabel('', self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QApplication.font()
        font.setPointSize(14)
        self.label.setFont(font)
        top_layout.addStretch(1)
        top_layout.addWidget(self.label)

        self.speedlabel = QLabel('⏱ 1.0 ×', self)
        self.speedlabel.setAlignment(Qt.AlignmentFlag.AlignRight)
        font = QApplication.font()
        font.setPointSize(14)
        self.speedlabel.setFont(font)
        top_layout.addStretch(1)
        top_layout.addWidget(self.speedlabel)

        self.audioplayer = AudioPlayer()  # getposition_interval_ms=AudioViewer.REFRESH_PERIOD

        self.transcription_panel = TranscriptionPanel(self.audioplayer)
        # AudioViewer(self.audioplayer, self, width_for_plot=0.8)
        layout.addWidget(self.transcription_panel)

        self.current_file_index = None
        self.transcription_loaded = False
        self.load_sound_by_index(0)

        # For registering ToDI transcription key sequences
        self.current_key_sequence = []
        self.current_key_sequence_time = None


    def load_sound_by_index(self, idx: int) -> None:
        """
        For an index to a sound file, this function stores current annotations in memory, loads and plays
        the requested sound file, and (re)loads the corresponding audioviewer (spectogram) and transcription
        panels (the latter only once the audio's duration is known).
        """
        self.keys_currently_pressed = set()    # to serve as refresh, sometimes it gets stuck...

        if idx == self.current_file_index:
            return

        if self.current_file_index is not None:  # i.e., if it's not first file being loaded, first save the current annotations
            if self.transcription_loaded:  # in case of next/prev before it gets fully loaded
                self.transcriptions[self.current_file_index] = [(b.relative_x * self.audioplayer.duration_ms, b.toPlainText()) for b in self.transcription_panel.textBubbles()]
                for item in self.transcription_panel.textBubbles():
                    item.scene().removeItem(item)

        self.transcription_loaded = False

        self.current_file_index = idx % len(self.wavfiles)
        path = self.wavfiles[self.current_file_index]
        tier_info = f', tier \'{self.save_as_textgrid_tier}\'' if self.save_as_textgrid_tier else ''

        self.label.setText(f"📁 {path} ({self.current_file_index + 1}/{len(self.wavfiles)}){tier_info}")

        self.audioplayer.pause()
        self.audioplayer.load_file(path)
        self.duration_known_so_load_transcription(self.audioplayer.duration_ms)
        self.stored_durations[path] = self.audioplayer.duration_ms

    def duration_known_so_load_transcription(self, duration):
        """
        Called once upon loading a new sound file, just to determine when the audioplayer is ready
        to determine the current file's duration -- needed for placing the annotation bubbles.
        """
        if duration > 0:
            for time, text in self.transcriptions[self.current_file_index]:
                self.transcription_panel.text_bubble_scene.new_item_relx(time / self.audioplayer.duration_ms, text)
            self.transcription_panel.load_file(self.wavfiles[self.current_file_index], duration=duration)
        self.transcription_loaded = True

    def keyPressEvent(self, event):
        """
        Handles most keyboard inputs, as defined in the Keys class, for controlling the audioplayer and
        for making the annotations.
        """
        if event.isAutoRepeat():
            return

        key = event.key()

        if key in Keys.PAUSE:
            self.audioplayer.toggle_play_pause()
            return
        elif key in Keys.FORWARD:
            self.audioplayer.skipforward()
        elif key in Keys.BACKWARD:
            self.audioplayer.skipbackward()
        elif key in Keys.MOREDELAY:
            self.audioplayer.increase_ghost_delay()
        elif key in Keys.LESSDELAY:
            self.audioplayer.decrease_ghost_delay()
        elif key in Keys.SLOWER:
            self.audioplayer.slowdown()
            self.speedlabel.setText(f'⏱ {self.audioplayer.PLAYBACK_RATE:.1f} ×')
        elif key in Keys.FASTER:
            self.audioplayer.speedup()
            self.speedlabel.setText(f'⏱ {self.audioplayer.PLAYBACK_RATE:.1f} ×')
        elif key in Keys.NEXT or (key == Qt.Key.Key_Right and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.next()
        elif key in Keys.PREVIOUS or (key == Qt.Key.Key_Left and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.prev()
        elif key in Keys.HOME:
            self.audioplayer.skiphome()
        elif key in Keys.END:
            self.audioplayer.skipend()
        elif key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.transcription_panel.remove_last_added_bubble()
        elif key == Qt.Key.Key_X and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.transcription_panel.remove_all_bubbles()
        elif key in Keys.TODI_KEYS:
            self.current_key_sequence.append(key)
            if Qt.Key.Key_Control in Keys.DOWNSTEP and key != Qt.Key.Key_Control and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.current_key_sequence.append(Qt.Key.Key_Control)
            elif Qt.Key.Key_Shift in Keys.UNCERTAIN and key != Qt.Key.Key_Shift and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.current_key_sequence.append(Qt.Key.Key_Shift)
            if key not in Keys.DOWNSTEP | Keys.UNCERTAIN:
                self.current_key_sequence_time = self.audioplayer.original_frame_to_ms(self.audioplayer.estimate_current_position() - self.audioplayer.actual_ghost_delay)
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
                self.transcription_panel.text_bubble_scene.new_item_relx(self.current_key_sequence_time / self.audioplayer.duration_ms, transcription)

        self.current_key_sequence = []
        self.current_key_sequence_time = None

    def next(self):
        """
        Go to next audio file (modulo-ized).
        """
        self.load_sound_by_index((self.current_file_index + 1) % len(self.wavfiles))

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
        self.transcriptions[self.current_file_index] = [(b.relative_x * self.audioplayer.duration_ms, b.toPlainText()) for b in self.transcription_panel.textBubbles()]

        self.audioplayer.pause()

        if self.save_as_textgrid_tier:
            io.write_to_textgrids(self.transcriptions,
                                  self.wavfiles,
                                  self.stored_durations,
                                  self.save_as_textgrid_tier)
        else:
            io.write_to_json(self.wavfiles, self.transcriptions, to_file=self.save_as_json)

        event.accept()


def main():
    """
    Starts the PyQt6 app and main window, and calls upon various ui_helpers for intercepting tab/shift+tab,
    mouse movements, mute some log messages, and sets up F1 for help window.
    """
    logging.basicConfig(format='%(name)s: %(message)s', level=logging.INFO)

    args = ui_helpers.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    app = QApplication(sys.argv)
    app.setStyle('fusion')
    icon = ui_helpers.load_icon()

    qInstallMessageHandler(ui_helpers.custom_message_handler)

    window = ToneSwiperWindow(args.file, save_as_textgrids=args.textgrid, save_as_json=args.json)
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
