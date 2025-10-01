#!/usr/bin/env python3
"""
Minimal PyQt6 audio player with a Praat-style spectrogram + moving progress bar.
Keyboard controls: Space=play/pause, >/< = next/prev, ]/[ = seek 5s.
"""
import sys
import os
import argparse
import logging
import functools

from PyQt6.QtCore import Qt, QUrl, QTimer, QElapsedTimer, QObject, QEvent, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QShortcut, QKeySequence

import parselmouth

from .textbubbles import TextBubbleSceneView, TabInterceptor
from . import resources
from . import io

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


todi_keys = {
    'high': Qt.Key.Key_Up,
    'low': Qt.Key.Key_Down,
    'left': Qt.Key.Key_Left,
    'right': Qt.Key.Key_Right,
    'downstep': Qt.Key.Key_Control,
}

key_str_to_todi = {
    'LH': 'L*H',
    'HL': 'H*L',
    'HL>': 'H*L L%',
    'LH>': 'L*H H%',
    'LHL': 'L*HL',  # delay
    'HLH': 'H*LH',  # only pre-nuclear
    'H>': 'H%',
    'L>': 'L%',
    '<H': '%H',
    '<L': '%L',
    '>': '%',
    'H': 'H*',
    'L': 'L*',
}


def key_sequence_to_transcription(key_sequence):
    proto_transcription = ''
    for key in key_sequence:
        if key == todi_keys['high']:
            proto_transcription += 'H'
        elif key == todi_keys['low']:
            proto_transcription += 'L'

    if todi_keys['right'] in key_sequence:
        proto_transcription += '>'
    if todi_keys['left'] in key_sequence:
        proto_transcription = '<' + proto_transcription

    try:
        transcription = key_str_to_todi[proto_transcription]
    except KeyError as e:
        raise ValueError(f'Not a valid key sequence: {proto_transcription}')

    if todi_keys['downstep'] in key_sequence:
        transcription = transcription.replace('H*', '!H*')

    return transcription


class AudioPlayerWrapper(QMediaPlayer):
    SEEK_STEP_MS = 500

    def __init__(self):
        # Setup QMediaPlayer with audio output
        super().__init__()

        self.elapsedtimer = QElapsedTimer()
        self.elapsedtimer.start()

        self.audio_output = QAudioOutput()
        self.setAudioOutput(self.audio_output)

        self.last_position = 0
        self.time_of_last_position = 0

        # Signals
        self.positionChanged.connect(self.on_position_changed)
        # self.player.durationChanged.connect(self.on_duration_changed)

    def load_file(self, path):
        # load into player
        self.setSource(QUrl.fromLocalFile(str(path)))
        # auto-play
        QTimer.singleShot(150, self.play)

    def toggle_play_pause(self):
        if self.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.time_of_last_position = None
            self.play()

    def seek_relative(self, delta_ms: int):
        newpos = min(max(0, self.position() + delta_ms), self.duration())
        self.setPosition(newpos)
        self.last_position = None

    def determine_current_position(self):
        if self.last_position is None:
            self.on_position_changed(self.position())
        if self.playbackState() == self.PlaybackState.PlayingState and self.time_of_last_position is not None:
            delta = self.elapsedtimer.elapsed() - self.time_of_last_position
        else:
            delta = 0
        estimated_position = self.last_position + delta
        return estimated_position

    def on_position_changed(self, ms):
        self.last_position = ms
        self.time_of_last_position = self.elapsedtimer.elapsed()


class SpectogramWidget(QWidget):
    def __init__(self, player: AudioPlayerWrapper, parent=None, width_for_plot=1.0):
        super().__init__(parent)
        self.fig = Figure(figsize=(8,3))
        self.width_for_plot = width_for_plot

        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.ax2 = None
        self.progress_line = None
        self.cursor_line = None
        self.duration = 0.0
        self.background = None

        self.last_drawn_position = 0

        self.update_timer = QTimer(self)
        self.update_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.update_timer.setInterval(10)  # 30 ms ~ 33 Hz
        self.player = player
        self.update_timer.timeout.connect(self.update_progress)
        self.update_timer.start()

        # Connect the draw_event to our handler
        self.canvas.mpl_connect('draw_event', self._on_draw)

    def _on_draw(self, event):
        """Handler for the 'draw_event' to recache the background."""
        # This will run on the first draw and any subsequent resizes.
        if self.canvas.get_renderer() is None:
            return
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def load_file(self, path):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        pitch, spec, xmin, xmax = self.get_cached_spectogram(str(path))
        self.duration = xmax

        self.draw_spectrogram(spec, ax=self.ax)
        self.ax2 = self.ax.twinx()
        self.draw_pitch(pitch, ax=self.ax2)
        self.ax.set_xlim(xmin, xmax)

        rectangle = ((1.0-self.width_for_plot)/2, 0.1, self.width_for_plot, 0.8)
        self.ax.set_position(rectangle)
        self.ax2.set_position(rectangle)

        self.progress_line = self.ax.axvline(0, color=(0.4, 0.4, 1.0), alpha=.8, linewidth=2, animated=True)
        self.cursor_line = self.ax.axvline(x=0, color="white", alpha=.6, linewidth=1)
        self.canvas.draw()

    def set_progress(self, fraction: float):
        if self.background is None or self.progress_line is None:
            return
        x = fraction * self.duration
        self.progress_line.set_xdata([x])
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.progress_line)
        if self.cursor_line is not None:
            self.ax.draw_artist(self.cursor_line)
        self.canvas.blit(self.ax.bbox)
        QApplication.processEvents()

    def update_progress(self):
        if self.player.duration() > 0:
            pos = self.player.determine_current_position()
            # avoiding jitter:
            if self.last_drawn_position is not None and self.last_drawn_position - 100 < pos < self.last_drawn_position:
                return
            fraction = pos / self.player.duration()
            self.set_progress(fraction)
            self.last_drawn_position = pos

    def update_cursor_line(self, global_pos):
        if self.background is None or self.cursor_line is None:
            return
        local_pos = self.canvas.mapFromGlobal(global_pos)
        # if self.canvas.rect().contains(local_pos):
        xdata, _ = self.ax.transData.inverted().transform((local_pos.x(), local_pos.y()))
        self.cursor_line.set_xdata([xdata])
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.cursor_line)
        if self.progress_line is not None:
            self.ax.draw_artist(self.progress_line)
        self.canvas.blit(self.ax.bbox)
        QApplication.processEvents()

    def get_plot_relative_x_positions(self) -> tuple[float, float]:
        # These ratios appear consistent under window resizing...
        return self.ax.get_position().x0, self.ax.get_position().x1

    @staticmethod
    @functools.cache
    def get_cached_spectogram(path):
        snd = parselmouth.Sound(str(path))
        pitch = snd.to_pitch(None)
        pre = snd.copy()
        pre.pre_emphasize()
        spec = pre.to_spectrogram(window_length=0.03, maximum_frequency=8000)
        return pitch, spec, snd.xmin, snd.xmax

    @staticmethod
    def draw_spectrogram(spec, ax, dynamic_range=70):
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
        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values==0] = np.nan
        times = pitch.xs()
        ax.plot(times, pitch_values, color='cyan')
        ax.set_ylabel('Pitch (Hz)')


class ToneSwiperWindow(QMainWindow):

    def __init__(self, files: list[str], save_as_textgrids: str = None, save_as_json: str = None):
        super().__init__()
        self.wavfiles = files
        self.current_file_index = None
        self.save_as_textgrid_tier = save_as_textgrids
        self.save_as_json = save_as_json

        self.transcriptions = [[] for _ in self.wavfiles]

        if self.save_as_json and os.path.exists(self.save_as_json):
            from_json = io.load_from_json(self.save_as_json)
            self.transcriptions = [from_json.get(filename, []) for filename in self.wavfiles]
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

        self.player = AudioPlayerWrapper()
        self.spectogram = SpectogramWidget(self.player, self, width_for_plot=0.8)

        layout.addWidget(self.spectogram)

        self.textboxes = TextBubbleSceneView(proportion_width=self.spectogram.width_for_plot)
        layout.addWidget(self.textboxes)

        self.first_file = True
        self.load_index(0)

        self.keys_currently_pressed = set()
        self.current_key_sequence = []
        self.current_key_sequence_time = None

    def load_index(self, idx: int):
        if idx == self.current_file_index:
            return

        if not self.first_file:
            self.transcriptions[self.current_file_index] = [(b.relative_x * self.player.duration(), b.toPlainText()) for b in self.textboxes.textBubbles()]
            for item in self.textboxes.textBubbles():
                item.scene().removeItem(item)
        self.first_file = False

        self.current_file_index = idx % len(self.wavfiles)
        path = self.wavfiles[self.current_file_index]
        self.label.setText(f"File {self.current_file_index + 1}/{len(self.wavfiles)}: {path}")

        self.spectogram.load_file(path)
        self.player.stop()
        self.player.load_file(path)
        self.player.durationChanged.connect(self.duration_known_so_load_transcription)


    def duration_known_so_load_transcription(self, duration):
        if duration > 0:
            for time, text in self.transcriptions[self.current_file_index]:
                self.textboxes.text_bubble_scene.new_item_relx(time / self.player.duration(), text)
        self.player.durationChanged.disconnect()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.player.toggle_play_pause()
            return

        t = event.text()
        key = event.key()
        self.keys_currently_pressed.add(key)

        if t == '>' or t == '.':
            self.player.seek_relative(self.player.SEEK_STEP_MS)
        elif t == '<' or t == ',':
            self.player.seek_relative(-self.player.SEEK_STEP_MS)
        elif t == '-':
            self.player.setPlaybackRate(max(self.player.playbackRate() - 0.1, 0.5))
        elif t == '+' or t == '=':
            self.player.setPlaybackRate(min(self.player.playbackRate() + 0.1, 2.0))

        if (key == Qt.Key.Key_Right and event.modifiers() & Qt.KeyboardModifier.AltModifier)\
                or (key == Qt.Key.Key_PageDown) or (key == Qt.Key.Key_BracketRight):
            self.next()
        elif (key == Qt.Key.Key_Left and event.modifiers() & Qt.KeyboardModifier.AltModifier)\
                or (key == Qt.Key.Key_PageUp) or (key == Qt.Key.Key_BracketLeft):
            self.prev()
        elif key == Qt.Key.Key_Home:
            self.load_index(0)
        elif key == Qt.Key.Key_End:
            self.load_index(len(self.wavfiles) - 1)

        if key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.clear_current()
            else:
                self.undo()
        elif key in todi_keys.values():
            self.current_key_sequence.append(key)
            if key != todi_keys['downstep']:
                self.current_key_sequence_time = self.player.position()
        else:
            self.current_key_sequence = []
            self.current_key_sequence_time = None

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()
        self.keys_currently_pressed.discard(key)

        logging.info(self.current_key_sequence)

        if self.keys_currently_pressed:  # Sequence not yet completed
            return
        else:
            if self.current_key_sequence and self.current_key_sequence_time:
                try:
                    transcription = key_sequence_to_transcription(self.current_key_sequence)
                except ValueError as e:
                    logging.warning(e)
                else:
                    self.textboxes.text_bubble_scene.new_item_relx(self.current_key_sequence_time / self.player.duration(), transcription)
            self.current_key_sequence = []
            self.current_key_sequence_time = None

    def next(self):
        self.load_index((self.current_file_index + 1) % len(self.wavfiles))

    def prev(self):
        self.load_index((self.current_file_index - 1) % len(self.wavfiles))

    def clear_current(self):
        for bubble in self.textboxes.textBubbles():
            bubble.scene().removeItem(bubble)

    def undo(self):
        bubbles = self.textboxes.textBubbles()
        if bubbles:
            last_bubble = bubbles.pop(0)
            last_bubble.scene().removeItem(last_bubble)

    def closeEvent(self, event):
        self.transcriptions[self.current_file_index] = [(b.relative_x * self.player.duration(), b.toPlainText()) for b in self.textboxes.textBubbles()]

        self.player.stop()

        if self.save_as_textgrid_tier:
            io.write_to_textgrids(self.transcriptions,
                              [wavfile.replace('.wav', '.TextGrid') for wavfile in self.wavfiles],
                              self.player.duration(),
                              self.save_as_textgrid_tier)
        else:
            io.write_to_json(self.wavfiles, self.transcriptions, to_file=self.save_as_json)

        event.accept()  # or event.ignore() to cancel closing

class CursorMonitor(QObject):
    def __init__(self, cursor_handler):
        """
        cursor_handler is any function taking only a (global) position as argument.
        Specifically, this was meant for SpectogramWidget.update_cursorline
        """
        self.cursor_handler = cursor_handler
        super().__init__()

    def eventFilter(self, obj, event) -> bool:
        """
        Monitors for mousemove events, and simply passes the global position into the cursor_handler.
        """
        if event.type() == QEvent.Type.MouseMove:
            global_pos = event.globalPosition().toPoint()
            self.cursor_handler(global_pos)

        return super().eventFilter(obj, event)


def custom_message_handler(msg_type, context, message):
    if "QFFmpeg::Demuxer::unnamed" in message:
        return
    if "QFFmpeg::StreamDecoder::unnamed" in message:
        return
    if "QFFmpeg::AudioRenderer::unnamed" in message:
        return
    if "Using Qt multimedia" in message:
        return
    print(message)


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('files', nargs='+', type=str, help='One or more .wav files')
    group = argparser.add_mutually_exclusive_group()
    group.add_argument('--textgrid', type=str, nargs='?', const='ToDI', default=None,
                           help='Will save annotations to .TextGrid files corresponding in name to the original '
                                '.wav files., to a tier with the specified name (default: "ToDI"). '
                                'If such .TextGrid files already exist, will load annotations from the given tier '
                                '(if it exists) and overwrite them.')
    group.add_argument('--json', type=str, help='Will save annotations to the specified .json file; if file '
                                                 'already exists, will also load from and overwrite it.',)
    args = argparser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle('fusion')
    icon = resources.load_icon()

    qInstallMessageHandler(custom_message_handler)

    window = ToneSwiperWindow(args.files, save_as_textgrids=args.textgrid, save_as_json=args.json)
    app.setWindowIcon(icon)
    window.setWindowIcon(icon)

    tab_interceptor = TabInterceptor(window.textboxes.text_bubble_scene.handle_tabbing)
    app.installEventFilter(tab_interceptor)
    cursor_monitor = CursorMonitor(window.spectogram.update_cursor_line)
    app.installEventFilter(cursor_monitor)

    help_box = resources.HelpOverlay(window)

    def show_help():
        help_box.show()
        help_box.activateWindow()
        help_box.raise_()

    QShortcut(QKeySequence("F1"), window, activated=show_help)

    screen_geom = QApplication.primaryScreen().availableGeometry()
    x = screen_geom.right() - help_box.width()
    y = screen_geom.top()
    help_box.move(x, y)

    window.resize(1200, 600)
    window.show()
    return app.exec()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='')
    raise SystemExit(main())
