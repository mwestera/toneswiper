import sys
import os
import logging

from PyQt6.QtCore import Qt, QUrl, QTimer, QElapsedTimer, QObject, QEvent, qInstallMessageHandler, QPointF
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel, QGraphicsView, QGraphicsLineItem, QMessageBox
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QShortcut, QKeySequence, QPainter, QKeyEvent, QPen, QColor, QCursor

from .textbubbles import TextBubbleScene, TextBubble
from . import ui_helpers
from . import io
from .ui_helpers import Keys

from . import spectogram


# TODO: Progress bar not correctly shown first 100-or-so ms.
# TODO: Update docstrings.
# TODO: Try make slowdown without pitch change.

class AudioPlayer(QMediaPlayer):
    """
    Wraps QMediaPlayer, mainly to facilitate displaying a more smoothly moving progress bar (because at least
    some audio back-ends may update position only once every 50-100ms), by virtue of self.estimate_current_position,
    as well as a delayed progress bar, by virtue of self.get_delayed_position
    """

    # UI things
    SEEK_STEP_MS = 800
    GHOST_DELAY_DELTA_MS = 200

    GHOST_DELAY_MS = 600
    GHOST_COUNTDOWN_PERIOD = 3

    def __init__(self):
        """
        Instantiates the audio player, connects it to the audio output, and sets up
        bookkeeping attributes for estimation of current position.

        If getposition_interval_ms is specified, will sync the queue with calls to estimate_current_position.
        """
        super().__init__()
        self.audio_output = QAudioOutput()  # apparently needed
        self.setAudioOutput(self.audio_output)

        # To extrapolate current position for a smoother moving progress bar:
        self.elapsedtimer = QElapsedTimer()
        self.elapsedtimer.start()
        self.last_position = 0
        self.time_of_last_position = 0
        self.positionChanged.connect(self.on_position_changed)

        self.position_has_been_updated = False  # some things cannot happen until the first update

        # Also store a 'ghost' of the playback position, with some delay
        self.actual_ghost_delay = self.GHOST_DELAY_MS
        self.ghost_countdown = QTimer(self)
        self.ghost_countdown.setTimerType(Qt.TimerType.PreciseTimer)
        self.ghost_countdown.setInterval(self.GHOST_COUNTDOWN_PERIOD)
        self.ghost_countdown.timeout.connect(self.update_actual_ghost_delay)
        self.ghost_countdown.start()

    def load_file(self, path: str, autoplay=True) -> None:
        """
        Loads a file and (by default) starts playing.
        """
        self.setSource(QUrl.fromLocalFile(path))
        if autoplay:
            QTimer.singleShot(150, self.play)
            self.last_position = None
            self.position_has_been_updated = False

    def update_actual_ghost_delay(self):
        if self.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
            self.actual_ghost_delay = max(0, self.actual_ghost_delay - self.GHOST_COUNTDOWN_PERIOD)
        else:
            self.actual_ghost_delay = self.GHOST_DELAY_MS

    def decrease_ghost_delay(self):
        self.GHOST_DELAY_MS = max(0, self.GHOST_DELAY_MS - self.GHOST_DELAY_DELTA_MS)

    def increase_ghost_delay(self):
        self.GHOST_DELAY_MS = min(self.GHOST_DELAY_MS + self.GHOST_DELAY_DELTA_MS, self.duration())

    def toggle_play_pause(self) -> None:
        """
        For use by play/pause hotkey.
        """
        if self.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.time_of_last_position = None
            self.play()

    def skipforward(self):
        self.seek_relative(self.SEEK_STEP_MS)

    def skipbackward(self):
        self.seek_relative(-self.SEEK_STEP_MS)

    def skiphome(self):
        self.setPosition(0)
        self.last_position = None

    def skipend(self):
        self.setPosition(self.duration())
        self.last_position = None

    def seek_relative(self, delta_ms: int) -> None:
        """
        Skips sound player ahead by delta_ms (or back, if negative), capped between 0 and duration.
        """
        newpos = min(max(0, self.position() + delta_ms), self.duration())
        self.setPosition(newpos)
        self.last_position = None  # in order for estimate_current_position to start afresh

    def estimate_current_position(self) -> int:
        """
        Estimates current position from last position and time_of_last_position.
        (Because at least some audio back-ends update position only once every 50-100ms.)
        """
        if not self.position_has_been_updated:
            return 0

        if self.last_position is None:  # e.g., at the start or if position was recently changed by seeking
            self.on_position_changed(self.position())
        if self.playbackState() == self.PlaybackState.PlayingState and self.time_of_last_position is not None:
            delta = self.elapsedtimer.elapsed() - self.time_of_last_position
        else:
            delta = 0
        estimated_position = self.last_position + delta

        return estimated_position

    def get_delayed_ghost_position(self):
        return self.actual_ghost_delay

    def on_position_changed(self, ms: float) -> None:
        """
        Keeping position and timing data for extrapolating, as done in self.best_current_position.
        """
        self.last_position = ms
        self.time_of_last_position = self.elapsedtimer.elapsed()
        self.position_has_been_updated = True  # now it's at least somewhat accurate


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
        self.timer.start(30)  # update ~33 FPS

        self._programmatic_scroll = False

    def load_file(self, path, duration):
        width_px = int(duration * (self.PX_PER_S / 1000))
        self.scene().setSceneRect(0, 0, width_px, 400)
        self.scene().removeItem(self.spectogram)
        self.spectogram = spectogram.make_image_cached(path, width_px, 200)
        self.spectogram.setPos(0, 0)
        self.scene().addItem(self.spectogram)
        self.progress_line.setZValue(1)
        self.transcription_line.setZValue(1)
        self.cursor_line.setZValue(2)

    def centerOn(self, pos):
        if self._programmatic_scroll:
            super().centerOn(pos)
        else:
            self.audioplayer.setPosition(self.pix_to_ms(pos))

    def scrollContentsBy(self, dx, dy):
        if self._programmatic_scroll:
            super().scrollContentsBy(dx, dy)
        else:
            x = self.ms_to_pix(self.audioplayer.estimate_current_position()) + -dx
            self.audioplayer.setPosition(self.pix_to_ms(x))

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

        position = self.ms_to_pix(self.audioplayer.estimate_current_position())
        rect = self.spectogram.boundingRect()
        x = rect.left() + position
        self.progress_line.setLine(x, rect.top(), x, rect.bottom())

        delayed_x = max(0, x - self.ms_to_pix(self.audioplayer.get_delayed_ghost_position()))
        self.transcription_line.setLine(delayed_x, rect.top(), delayed_x, rect.bottom())

        self._programmatic_scroll = True
        self.centerOn(QPointF(x, 0))
        self._programmatic_scroll = False

        global_cursor_pos = QCursor.pos()
        local_cursor_pos = self.mapFromGlobal(global_cursor_pos)
        cursor_x = local_cursor_pos.x() - self.mapFromScene(self.sceneRect().left(), 0).x() - self.PADDING
        self.cursor_line.setLine(cursor_x, rect.top(), cursor_x, rect.bottom())


    def ms_to_pix(self, pos_ms):
        pix_per_ms = self.PX_PER_S / 1000
        return int(pos_ms * pix_per_ms)

    def pix_to_ms(self, pix):
        pix_per_ms = self.PX_PER_S / 1000
        return int(pix / pix_per_ms)


class CurrentlyPressedKeysTracker(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pressed_keys = set()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
            self.pressed_keys.add(event.key())
        elif event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            self.pressed_keys.discard(event.key())
        return False  # don't filter out, for processing further down


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
        QApplication.instance().installEventFilter(self.currently_pressed_keys_tracker)

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

        self.transcription_panel = TranscriptionPanel(self.audioplayer)
        # AudioViewer(self.audioplayer, self, width_for_plot=0.8)
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
            self.transcription_panel.load_file(self.wavfiles[self.current_file_index], duration=duration)
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
            self.audioplayer.skipforward()
        elif key in Keys.BACKWARD:
            self.audioplayer.skipbackward()
        elif key in Keys.MOREDELAY:
            self.audioplayer.increase_ghost_delay()
        elif key in Keys.LESSDELAY:
            self.audioplayer.decrease_ghost_delay()
        elif key in Keys.SLOWER:
            self.audioplayer.setPlaybackRate(max(self.audioplayer.playbackRate() - 0.1, 0.5))
        elif key in Keys.FASTER:
            self.audioplayer.setPlaybackRate(min(self.audioplayer.playbackRate() + 0.1, 2.0))
        elif key in Keys.NEXT or (key == Qt.Key.Key_Right and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.next()
        elif key in Keys.PREVIOUS or (key == Qt.Key.Key_Left and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.prev()
        elif key in Keys.HOME:
            self.audioplayer.skiphome()
        elif key in Keys.END:
            self.audioplayer.skipend()

        if key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.transcription_panel.remove_last_added_bubble()
        elif key == Qt.Key.Key_X and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.transcription_panel.remove_all_bubbles()
        elif key in Keys.TODI_KEYS:
            self.current_key_sequence.append(key)
            if key not in Keys.DOWNSTEP:
                self.current_key_sequence_time = self.audioplayer.estimate_current_position() - self.audioplayer.get_delayed_ghost_position()
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

    help_box = ui_helpers.HelpOverlay(window)
    QShortcut(QKeySequence("F1"), window, activated=help_box.display_panel)
    screen_geom = QApplication.primaryScreen().availableGeometry()
    help_box.move(screen_geom.right() - help_box.width(), screen_geom.top())

    window.resize(1200, 600)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
