import sys
import os
import logging
import functools

from PyQt6.QtCore import Qt, QUrl, QTimer, QElapsedTimer, QObject, QEvent, qInstallMessageHandler, QPointF
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QShortcut, QKeySequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import parselmouth

from .textbubbles import TextBubbleSceneView
from . import ui_helpers
from . import io
from .ui_helpers import Keys
from collections import deque


# TODO: Progress bar not correctly shown first 100-or-so ms.

class AudioPlayer(QMediaPlayer):
    """
    Wraps QMediaPlayer, mainly to facilitate displaying a more smoothly moving progress bar (because at least
    some audio back-ends may update position only once every 50-100ms), by virtue of self.estimate_current_position,
    as well as a delayed progress bar, by virtue of self.get_delayed_position
    """

    DELAY_MS = 500
    DELAY_REFRESH_PERIOD_MS = 10

    # UI things
    SEEK_STEP_MS = 800
    DELAY_DELTA_MS = 100  # increment size when adjusting delay

    def __init__(self, getposition_interval_ms=None):
        """
        Instantiates the audio player, connects it to the audio output, and sets up
        bookkeeping attributes for estimation of current position.

        If getposition_interval_ms is specified, will sync the queue with calls to estimate_current_position.
        """
        super().__init__()
        self.audio_output = QAudioOutput()  # apparently needed
        self.setAudioOutput(self.audio_output)

        # To 'extrapolate' current position for a smoother moving progress bar:
        self.elapsedtimer = QElapsedTimer()
        self.elapsedtimer.start()
        self.last_position = 0
        self.time_of_last_position = 0
        self.positionChanged.connect(self.on_position_changed)

        # Queue to store previous positions.
        self.sync_queue_with_getposition = getposition_interval_ms is not None
        if self.sync_queue_with_getposition:
            self.DELAY_REFRESH_PERIOD_MS = getposition_interval_ms
        self.position_queue_maxlength = self.DELAY_MS // self.DELAY_REFRESH_PERIOD_MS
        self.position_queue: deque[int] = deque()

        if not self.sync_queue_with_getposition:  # set up its own timer
            self.queue_refresh_timer = QTimer(self)
            self.queue_refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.queue_refresh_timer.setInterval(self.DELAY_REFRESH_PERIOD_MS)
            self.queue_refresh_timer.timeout.connect(self.update_position_queue)
            self.queue_refresh_timer.start()

        self.update_position = False  # only set to True once position has been updated for the first time

    def load_file(self, path: str, autoplay=True) -> None:
        """
        Loads a file and (by default) starts playing.
        """
        self.setSource(QUrl.fromLocalFile(path))
        self.position_queue.clear()
        if autoplay:
            QTimer.singleShot(150, self.play)
            self.position_queue.appendleft(0)
            self.last_position = None
            self.update_position = False

    def toggle_play_pause(self) -> None:
        """
        For use by play/pause hotkey.
        """
        if self.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.time_of_last_position = None
            self.play()

    def seek_relative(self, delta_ms: int) -> None:
        """
        Skips sound player ahead by delta_ms (or back, if negative), capped between 0 and duration.
        """
        newpos = min(max(0, self.position() + delta_ms), self.duration())
        self.setPosition(newpos)
        self.position_queue.clear()
        self.position_queue.appendleft(newpos)
        self.last_position = None  # in order for estimate_current_position to start afresh

    def estimate_current_position(self) -> int:
        """
        Estimates current position from last position and time_of_last_position.
        (Because at least some audio back-ends update position only once every 50-100ms.)
        """
        if not self.update_position:
            if self.sync_queue_with_getposition:
                self.update_position_queue(0)
            return 0

        if self.last_position is None:  # e.g., at the start or if position was recently changed by seeking
            self.on_position_changed(self.position())
        if self.playbackState() == self.PlaybackState.PlayingState and self.time_of_last_position is not None:
            delta = self.elapsedtimer.elapsed() - self.time_of_last_position
        else:
            delta = 0
        estimated_position = self.last_position + delta

        if self.sync_queue_with_getposition:
            self.update_position_queue(estimated_position)

        return estimated_position

    def get_delayed_position(self) -> int | None:
        """

        """
        if self.position_queue and self.DELAY_MS != 0:
            return self.position_queue[-1]
        else:
            return None

    def update_position_queue(self, pos=None):

        while len(self.position_queue) >= self.position_queue_maxlength:
            self.position_queue.pop()
        self.position_queue.appendleft(pos if pos is not None else self.estimate_current_position())

    def on_position_changed(self, ms: float) -> None:
        """
        Keeping position and timing data for extrapolating, as done in self.best_current_position.
        """
        self.last_position = ms
        self.time_of_last_position = self.elapsedtimer.elapsed()
        self.update_position = True  # now it's at least somewhat accurate

    def change_delay(self, delta: int):
        self.DELAY_MS = max(self.DELAY_MS + delta, 0)
        self.position_queue_maxlength = max(self.DELAY_MS // self.DELAY_REFRESH_PERIOD_MS, 1)


class AudioViewer(QWidget):
    """
    Panel showing a Praat spectogram of the current sound, with a pitch track overlaid,
    a moving progress bar depending on the audioplayer position, and a moving bar corresponding
    to the cursor's x position in the window.
    """

    FRAMERATE = 60  # fps
    REFRESH_PERIOD = 1000 // FRAMERATE

    def __init__(self, player: AudioPlayer, parent=None, width_for_plot: float = 1.0):
        """
        The audioplayer matters for displaying a moving progress bar based on the audioplayer's (estimated) position.
        Width_for_plot [0,1] matters because it will affect the width of the textbubbles panel.
        Parent is passed into super().
        """
        super().__init__(parent)
        self.fig = Figure(figsize=(8,3))
        self.width_for_plot = width_for_plot

        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # To be instantiated once first audio file is loaded
        self.ax = None
        self.ax2 = None
        self.progress_line = None
        self.progress_line_delayed = None
        self.cursor_line = None
        self.duration = None
        self.background = None
        self.last_drawn_position = None
        self.last_drawn_position_delayed = None

        self.player = player

        self.update_timer = QTimer(self)
        self.update_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.update_timer.setInterval(self.REFRESH_PERIOD)
        self.update_timer.timeout.connect(self.update_progress)
        self.update_timer.start()

        # For caching the plot background for more efficient drawing (blitting):
        self.canvas.mpl_connect('draw_event', self._on_draw)

    def _on_draw(self, event) -> None:
        """
        Handler for the 'draw_event' to recache the background.
        Will run on the first draw and any subsequent window resizes.
        """
        if self.canvas.get_renderer():
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def load_file(self, path: str) -> None:
        """
        Loads an audio file, processes it with praat (parselmouth) to extract spectogram and pitch,
        uses matplotlib to create the corresponding overlaid plots, and initiates two vertical lines:
        progress bar and cursor x-position bar.
        """

        self.fig.clear()

        # TODO: Cache the actual background, not only the data.
        # self.background = self.cached_backgrounds.get(path)

        self.ax = self.fig.add_subplot(111)
        pitch, spec, xmin, xmax = self.make_spectogram_cached(path)
        self.duration = xmax

        self.draw_spectrogram(spec, ax=self.ax)
        self.ax2 = self.ax.twinx()
        self.draw_pitch(pitch, ax=self.ax2)
        self.ax.set_xlim(xmin, xmax)

        rectangle = ((1.0-self.width_for_plot)/2, 0.1, self.width_for_plot, 0.8)
        self.ax.set_position(rectangle)
        self.ax2.set_position(rectangle)

        self.progress_line = self.ax.axvline(0, color=(0.4, 0.4, 1.0), linewidth=2, animated=True)
        self.progress_line_delayed = self.ax.axvline(0, color=(0.7, 0.7, 1.0), linewidth=1, animated=True)
        self.cursor_line = self.ax.axvline(x=0, color="white", alpha=.6, linewidth=1)
        self.canvas.draw()

        self.last_drawn_position = None
        self.last_drawn_position_delayed = None


    def update_progress(self) -> None:
        """
        Called by a timer with FRAMERATE, to update the moving progress bar, with some jitter avoidance.
        """
        if self.player.duration():
            pos = self.player.estimate_current_position()
            fraction = pos / self.player.duration()

            if (pos_delayed := self.player.get_delayed_position()) is not None:
                fraction_delayed = pos_delayed / self.player.duration()
            else:
                fraction_delayed = None

            self.set_progress(fraction, fraction_delayed)

    def set_progress(self, fraction: float, fraction_delayed: float = None) -> None:
        """
        Redraws the spectogram plot with the vertical progress bar at the given fraction of the x-axis,
        using blitting for efficiency. Called by update_progress. If fraction_delayed is provided, a secondary
        line will be drawn.
        """
        if self.background is None or self.progress_line is None:
            return

        x = fraction * self.duration
        if self.last_drawn_position is not None and (self.last_drawn_position - .1) < x < self.last_drawn_position:
            x = self.last_drawn_position  # avoid jitter:

        self.progress_line.set_xdata([x])
        self.last_drawn_position = x

        if fraction_delayed is not None:
            x_delayed = fraction_delayed * self.duration
            if self.last_drawn_position_delayed is not None and (self.last_drawn_position_delayed - .1) < x_delayed < self.last_drawn_position_delayed:
                x_delayed = self.last_drawn_position_delayed  # avoid jitter:
            self.progress_line_delayed.set_xdata([x_delayed])
            self.last_drawn_position_delayed = x_delayed

        if self.cursor_line is not None:
            self.ax.draw_artist(self.cursor_line)

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.cursor_line)
        self.ax.draw_artist(self.progress_line)
        if fraction_delayed is not None:
            self.ax.draw_artist(self.progress_line_delayed)
        self.canvas.blit(self.ax.bbox)
        QApplication.processEvents()

    def update_cursor_line(self, global_pos: QPointF) -> None:
        """
        Updates the cursor line for the plot. Gets a global position, to be called from outside the class
        (in this case a global CursorMonitor instance). Redraw not actually done here; only in set_progress.
        """
        if self.background is None or self.cursor_line is None:
            return
        local_pos = self.canvas.mapFromGlobal(global_pos)
        xdata, _ = self.ax.transData.inverted().transform((local_pos.x(), local_pos.y()))
        self.cursor_line.set_xdata([xdata])

    @staticmethod
    @functools.cache
    def make_spectogram_cached(path):
        """
        Wrapper around parselmouth spectogram and pitch extraction, to be able to
        cache it (per .wav file path).
        """
        snd = parselmouth.Sound(str(path))
        pitch = snd.to_pitch(None)
        pre = snd.copy()
        pre.pre_emphasize()
        spec = pre.to_spectrogram(window_length=0.03, maximum_frequency=8000)
        return pitch, spec, snd.xmin, snd.xmax

    @staticmethod
    def draw_spectrogram(spec, ax, dynamic_range=70):
        """
        From parselmouth spectogram to a matplotlib plot.
        """
        data = 10 * np.log10(np.maximum(spec.values, 1e-10))
        vmax = data.max()
        vmin = vmax - dynamic_range

        X, Y = spec.x_grid(), spec.y_grid()
        sg_db = 10 * np.log10(spec.values)
        ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
        ax.axis(ymin=spec.ymin, ymax=spec.ymax)

        ax.imshow(data, origin='lower', aspect='auto', cmap='gray', extent=[spec.xmin, spec.xmax, 0, spec.ymax], vmin=vmin, vmax=vmax)
        ax.set_ylabel('Frequency (Hz)')

    @staticmethod
    def draw_pitch(pitch, ax):
        """
        From parselmouth pitch track to a matplotlib plot.
        """
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        times = pitch.xs()
        ax.plot(times, pitch_values, color='cyan')
        ax.set_ylabel('Pitch (Hz)')


class CurrentlyPressedKeysTracker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pressed_keys = set()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            self.pressed_keys.add(event.key())
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self.pressed_keys.discard(event.key())
        return False  # don't block the event


class ToneSwiperWindow(QMainWindow):
    """
    Main window of the app, wrapping an AudioPlayer, Audioviewer and TextBubbleSceneView.
    Handles most keyboard controls for transcription and audioplayer.
    """

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

        self.currently_pressed_keys_tracker = CurrentlyPressedKeysTracker()

        self.transcriptions = [[] for _ in self.wavfiles]
        if self.save_as_json and os.path.exists(self.save_as_json):
            from_json = io.load_from_json(self.save_as_json)
            self.transcriptions = [from_json.get(filename, []) for filename in self.wavfiles]
            if self.save_as_textgrid_tier:
                logging.warning("Both save_as_json and save_as_textgrid_tier specified;"
                                "will only load from textgrids (but save to both).")
        if self.save_as_textgrid_tier:
            from_textgrids = io.load_from_textgrids(self.wavfiles, self.save_as_textgrid_tier)
            self.transcriptions = [from_textgrids.get(wavfile, []) for wavfile in self.wavfiles]

        self.setWindowTitle('ToneSwiper')
        central = QWidget()
        layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        self.label = QLabel('', self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QApplication.font()
        font.setPointSize(14)
        self.label.setFont(font)
        layout.addWidget(self.label)

        self.audioplayer = AudioPlayer()  # getposition_interval_ms=AudioViewer.REFRESH_PERIOD
        self.audioviewer = AudioViewer(self.audioplayer, self, width_for_plot=0.8)
        layout.addWidget(self.audioviewer)
        self.transcription_panel = TextBubbleSceneView(proportion_width=self.audioviewer.width_for_plot)
        layout.addWidget(self.transcription_panel)

        self.current_file_index = None
        self.load_sound_by_index(0)
        self.transcription_loaded = False

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

        if self.current_file_index is not None:  # i.e., if it's not first file, first save the current annotations
            if self.transcription_loaded:
                self.transcriptions[self.current_file_index] = [(b.relative_x * self.audioplayer.duration(), b.toPlainText()) for b in self.transcription_panel.textBubbles()]
                for item in self.transcription_panel.textBubbles():
                    item.scene().removeItem(item)

        self.transcription_loaded = False

        self.current_file_index = idx % len(self.wavfiles)
        path = self.wavfiles[self.current_file_index]
        self.label.setText(f"File {self.current_file_index + 1}/{len(self.wavfiles)}: {path}")

        self.audioviewer.load_file(path)
        self.audioplayer.stop()
        self.audioplayer.load_file(path)
        # Audioplayer may take a while to know the duration, which in turn affects the placement of annotations:
        self.audioplayer.durationChanged.connect(self.duration_known_so_load_transcription)

    def duration_known_so_load_transcription(self, duration):
        """
        Called once upon loading a new sound file, just to determine when the audioplayer is ready
        to determine the current file's duration -- needed for placing the annotation bubbles.
        """
        if duration > 0:
            for time, text in self.transcriptions[self.current_file_index]:
                self.transcription_panel.text_bubble_scene.new_item_relx(time / self.audioplayer.duration(), text)
        self.audioplayer.durationChanged.disconnect()
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
            self.audioplayer.seek_relative(self.audioplayer.SEEK_STEP_MS)
        elif key in Keys.BACKWARD:
            self.audioplayer.seek_relative(-self.audioplayer.SEEK_STEP_MS)
        elif key in Keys.MOREDELAY:
            self.audioplayer.change_delay(self.audioplayer.DELAY_DELTA_MS)
        elif key in Keys.LESSDELAY:
            self.audioplayer.change_delay(-self.audioplayer.DELAY_DELTA_MS)
        elif key in Keys.SLOWER:
            self.audioplayer.setPlaybackRate(max(self.audioplayer.playbackRate() - 0.1, 0.5))
        elif key in Keys.FASTER:
            self.audioplayer.setPlaybackRate(min(self.audioplayer.playbackRate() + 0.1, 2.0))
        elif key in Keys.NEXT or (key == Qt.Key.Key_Right and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.next()
        elif key in Keys.PREVIOUS or (key == Qt.Key.Key_Left and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.prev()
        elif key in Keys.FIRST:
            self.load_sound_by_index(0)
        elif key in Keys.LAST:
            self.load_sound_by_index(len(self.wavfiles) - 1)

        if key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.transcription_panel.remove_all_bubbles()
            else:
                self.transcription_panel.remove_last_added_bubble()
        elif key in Keys.TODI_KEYS:
            self.current_key_sequence.append(key)
            if key not in Keys.DOWNSTEP:
                self.current_key_sequence_time = self.audioplayer.get_delayed_position()
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
                logging.warning(e)
            else:
                self.transcription_panel.text_bubble_scene.new_item_relx(self.current_key_sequence_time / self.audioplayer.duration(), transcription)

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
        self.transcriptions[self.current_file_index] = [(b.relative_x * self.audioplayer.duration(), b.toPlainText()) for b in self.transcription_panel.textBubbles()]

        self.audioplayer.stop()

        if self.save_as_textgrid_tier:
            io.write_to_textgrids(self.transcriptions,
                                  [wavfile.replace('.wav', '.TextGrid') for wavfile in self.wavfiles],
                                  self.audioplayer.duration(),
                                  self.save_as_textgrid_tier)
        else:
            io.write_to_json(self.wavfiles, self.transcriptions, to_file=self.save_as_json)

        event.accept()


def main():
    """
    Starts the PyQt6 app and main window, and calls upon various ui_helpers for intercepting tab/shift+tab,
    mouse movements, mute some log messages, and sets up F1 for help window.
    """

    args = ui_helpers.parse_args()

    app = QApplication(sys.argv)
    app.setStyle('fusion')
    icon = ui_helpers.load_icon()

    qInstallMessageHandler(ui_helpers.custom_message_handler)

    window = ToneSwiperWindow(args.files, save_as_textgrids=args.textgrid, save_as_json=args.json)
    app.setWindowIcon(icon)
    window.setWindowIcon(icon)

    tab_interceptor = ui_helpers.TabInterceptor(window.transcription_panel.text_bubble_scene.handle_tabbing)
    app.installEventFilter(tab_interceptor)
    cursor_monitor = ui_helpers.CursorMonitor(window.audioviewer.update_cursor_line)
    app.installEventFilter(cursor_monitor)

    help_box = ui_helpers.HelpOverlay(window)
    QShortcut(QKeySequence("F1"), window, activated=help_box.display_panel)
    screen_geom = QApplication.primaryScreen().availableGeometry()
    help_box.move(screen_geom.right() - help_box.width(), screen_geom.top())

    window.resize(1200, 600)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
