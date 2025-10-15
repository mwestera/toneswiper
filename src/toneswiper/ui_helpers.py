from PyQt6.QtCore import Qt, QByteArray, QObject, QEvent
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QScrollBar, QStyleOptionSlider, QStyle
from PyQt6.QtGui import QIcon, QPixmap

import os
import glob
import argparse

import logging
import sys
from datetime import datetime
from pathlib import Path
import functools
import json
import traceback
import inspect

logger = logging.getLogger('toneswiper')
measurer = logging.getLogger('measurer')


def setup_logging(verbose, measure):
    """
    Set up `logger` for stdout and `measurer` for a file.
    """

    logger = logging.getLogger("toneswiper")
    logger.setLevel(logging.INFO)
    if verbose:
        logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    logger.addHandler(stream_handler)
    logger.propagate = False

    measurer = logging.getLogger("measurer")
    measurer.setLevel(logging.CRITICAL + 1)  # silence
    if measure:
        measurer.setLevel(logging.INFO)
    log_dir = Path("measurements")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    measurer.addHandler(file_handler)
    measurer.propagate = False


def measure(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        result = None
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            result = e
            raise e
        finally:
            funcname = func.__name__
            params = list(inspect.signature(func).parameters.values())
            if params and params[0].name == "self":
                realparams = params[1:]
                realargs = args[1:]
                classname = type(args[0]).__name__
                funcname = f'{classname}.{funcname}'
            else:
                realparams = params
                realargs = args
            realparam_names = [str(param).split(': ')[0] for param in realparams]
            arguments_as_dict = {**dict(zip(realparam_names, realargs)), **kwargs}

            measurer.info(
                json.dumps(
                    {
                        'action': funcname,
                        'arguments': arguments_as_dict,
                        'result': result,
                    }, default=str,
                )
            )

    return decorated


class HelpOverlay(QWidget):
    """
    Defines an pop-up window with help about the keyboard controls.
    """

    text = ("<h2>Welcome to ToneSwiper!</h2>"
             "<h2>🖥 From the command-line</h2>"
             "<ul>"
             "<li>The command <code>toneswiper</code> plus one or more .wav files starts the application."
             "<li>To <b>save</b> your annotations, run it with <code>--json</code> or <code>--textgrid</code>."
             "<li>Do <code>toneswiper --help</code> for more detail about command-line options."
             "</ul>"

             "<h2>👂 ToDI commands</h2>"
             "<ul>"
             "<li>⬅:</b> left boundary (combine with ⬆/⬇ for high/low boundary)."
             "<li>➡:</b> right boundary (combine with ⬆/⬇/neither for high/low/level boundary)."
             "<li>⬆:</b> high tone; renders as H* if first tone in non-boundary sequence."
             "<li>⬇:</b> low tone; renders as L* if first tone in non-boundary sequence."
             "<li><b>Control:</b> when combined with H*, results in downstep !H*."
            "<li><b>(Back)slash (/, \\):</b> when combined with H*, makes vocative chant H*!H."
            "<li><b>Shift:</b> marks the current transcription as uncertain by appending '?'." 
             "</ul>"

             "<h2>📻 Navigation and audio commands</h2>"
             "<ul>"
             "<li>F1: Display this help"
             "<li>Alt+F4: Quit (<b>will auto-save</b>)"
             "<li>PageUp/PageDown (also Alt+⬅/➡): Next/previous sound file"
             "<li>Home/End: Go to start/end of current sound file"
             "<li>Space: Play/pause current sound file"
             "<li>Angle brackets (&gt; &lt;); also mousewheel 🖱: Fastforward/backward"
             "<li>Square brackets (], [): decrease/increase delay between audio and annotation timing."
             "<li>Plus/minus (+, -): increase/decrease playback speed (and pitch, for now 😼)"
             "</ul>"

             "<h2>📝 Editing annotations (text bubbles)</h2>"
             "<ul>"
             "<li>Double-click in the annotation pane to add an annotation."
             "<li>When editing an annotation, hit ENTER to stop editing."
             "<li>Click and drag an annotation to move it."
             "<li>When editing an annotation, hit shift+⬅/➡ to slightly move it horizontally."
             "<li>Right-click an existing annotation to delete it."
             "<li>Ctrl+Z: 'Undo', i.e., remove most recently added annotation (careful: no 'redo')."
             "<li>Ctrl+X: Remove all annotations of the current audio file; a blank slate!"
             "</ul>")

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle("ToneSwiper help")
        font = QApplication.font()
        font.setPointSize(12)
        self.setFont(font)

        layout = QVBoxLayout(self)
        label = QLabel(self.text)
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)

    @measure
    def display_panel(self):
        self.show()
        self.activateWindow()
        self.raise_()

    @measure
    def closeEvent(self, event):
        super().closeEvent(event)

    def __str__(self):
        return "HelpOverlay"

class Keys:
    PAUSE = {Qt.Key.Key_Space}
    FORWARD = {Qt.Key.Key_Greater, Qt.Key.Key_Period}
    BACKWARD = {Qt.Key.Key_Less, Qt.Key.Key_Comma}
    SLOWER = {Qt.Key.Key_Minus, Qt.Key.Key_Underscore}
    FASTER = {Qt.Key.Key_Plus, Qt.Key.Key_Equal}
    NEXT = {Qt.Key.Key_PageDown, Qt.Key.Key_BracketRight}
    PREVIOUS = {Qt.Key.Key_PageUp, Qt.Key.Key_BracketLeft}
    MOREDELAY = {Qt.Key.Key_BracketLeft}
    LESSDELAY = {Qt.Key.Key_BracketRight}
    HOME = {Qt.Key.Key_Home}
    END = {Qt.Key.Key_End}

    HIGH = {Qt.Key.Key_Up}
    LOW = {Qt.Key.Key_Down}
    LEFT = {Qt.Key.Key_Left}
    RIGHT = {Qt.Key.Key_Right}
    DOWNSTEP = {Qt.Key.Key_Control}
    CHANT = {Qt.Key.Key_Backslash, Qt.Key.Key_Slash}
    UNCERTAIN = {Qt.Key.Key_Shift}

    TODI_KEYS = HIGH | LOW | LEFT | RIGHT | DOWNSTEP | CHANT | UNCERTAIN


key_str_to_todi = {
    'LH': 'L*H',
    'HL': 'H*L',
    'HL>': 'H*L L%',
    'LH>': 'L*H H%',
    'LHL': 'L*HL',  # 'delayed' H*L
    'HLL': 'L*HL',  # ditto, but sloppily pressed
    'HLH': 'H*LH',  # fall-rise accent, only pre-nuclear
    'LHH': 'H*LH',  # ditto, but sloppily pressed
    'LHL>': 'L*HL L%',
    'HLL>': 'L*HL L%',
    'HLH>': 'H*LH H%',
    'LHH>': 'H*LH H%',
    'H>': 'H%',
    'L>': 'L%',
    '<H': '%H',
    '<L': '%L',
    '>': '%',
    'H': 'H*',
    'L': 'L*',
    'H\\': 'H*!H',  # vocative chaaaahaaant
}


@measure
def key_sequence_to_transcription(key_sequence: list[Qt.Key]):
    """
    Turns a list of PyQt keys first into a standardized 'proto-transcription', which is then via a dictionary
    mapped into a real ToDI transcription.
    Can raise a ValueError (caught higher up) if the key sequence does not define a ToDi sequence.
    """
    proto_transcription = ''
    for key in key_sequence:
        if key in Keys.HIGH:
            proto_transcription += 'H'
        elif key in Keys.LOW:
            proto_transcription += 'L'

    if any(k in Keys.CHANT for k in key_sequence):
        proto_transcription += '\\'

    if any(k in Keys.RIGHT for k in key_sequence):
        proto_transcription += '>'
    if any(k in Keys.LEFT for k in key_sequence):
        proto_transcription = '<' + proto_transcription

    try:
        transcription = key_str_to_todi[proto_transcription]
    except KeyError as e:
        raise ValueError(f'Not a valid key sequence: {" ".join(Qt.Key(k).name.removeprefix("Key_") for k in key_sequence)}')

    if any(k in Keys.DOWNSTEP for k in key_sequence):
        transcription = transcription.replace('H*', '!H*')

    if any(k in Keys.UNCERTAIN for k in key_sequence):
        transcription += '?'

    return transcription


def load_icon() -> QIcon:
    """
    Creates a QIcon() object with different resolutions, based on hex-encoded icons
    that were pre-computed using the base64 module:
    import base64; icon_base64 = base64.b64encode(open('icon.png', 'br').read()).decode("utf-8")
    """
    icon_16 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAEuAAABLgF7cRpNAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAfdJREFUOI2Vkz1vUgEUhp/7AYJA+EjbQLm0oNWqqZI2QaNCYpVFt9qxizG6OeliHLvUP+EkHUyMdjEmpg7GdCFRSWoKJgollFYHSoGWSy/cex3UarSS8s7nfXLOe84RnvEx2qSdMmi7TUyBA0hAMGWkmhXHjNxiN7VOYczAOIh3TyJSKEg4Jeronl9mi02mL+LF5rLua4rdPY5ysQ8AAx0D3SsDWB0W4jcnOHIhhCgJmIZJ8d0GS4/es7Op4jnq5PSNMIPnfLgUO8NXBliaXfnZCRA5G2QkMYQo/YhAEAXCsUGm5pK4/U60eptWVWPoUj+j1xWqX7b3upIBCuky7oCLb58qlDIbBE4NMHknxmGvneS98zy/v4jVKZN9UiIQ81HL7/wO9DHpUplV5e95fSE3Uw+TSBaR9Pwya5/XqRebuBQ7jbKK1ugQJLwm/i/lzVKNzEIWgPHpk9RXVbTtDpVcA63R+WMbXZRZyKHWWlhsMqOTkX1rugL0tkF2MQ/A2NURBPHfO+sKAFh5lcfQDVz9Do4lhnsHNKsqudcFAOK3xlGi/t4AAOn5ZSrFLeRDMtceJIjfnugNoKltXsy+ofThKwhw4nIEySoBIAtIWyKSYqB3hbQaGi/n3qJE/eiajqmBiFQVnpI7s8tOSkf39PLOIlLNhmPmO91xsArINKkeAAAAAElFTkSuQmCC"
    icon_32 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAJdAAACXQHBe+vTAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAABF9JREFUWIW9l2tsk1UYx3/n7X3t2rEy3MVujDEYsAIKGmDouKlkKqgRSDQmhhgTF2LcBy8hxmBMcMGEaEi8xC9o5JI5YzAjAV3IBJTLCGjacRuXrWwrMFZGS9d269vjhwn4+hYZZt3/2/mf5z3PL+fkPOd9BIBEigbaaoE6gZwIwkBGJFWJ6FBg84vM+EIgpNiAVKbja5IYnooQUgaJZyb33zJjJZvcFKh7VuN9RjTQtk7CZ0EuKnEGMpr8lqzYKWBiSpBap0jkWxFCY5YcIE6UCCEFRJ2iQEmC2KgncRTamPduBWVPF6SdTxBDQKlRghGkLkAIQW6xC1ehA4CheJLgyV6SCXVEAE6Pjep6L/5vOzm/O5gmQiLBaPy3bXNamLWygqlLJmKxmzVzyYRK4HiQo9t9hK/c1C1pz7dic5u51hbWzVlcJhwFVvpORzS+BqDIO4En367CZNVxDQdbDEya/yDFD+dzZJuPtr3nNJtX+UoJizbN5EZHlF7fDQAKHhnHSy2LKKoaT/BoiO+q9mnWVP45mLum8q7JtSBGqtY+xOOvzwFxx++/GKXjlyvY861MfrYQAPc0J57qPMKBAS4fC+nXume2/1DF0kkMJVQObf0DgDONXZxp7CJrgoXXTi7H5h4+wt8/OsmBD9rSrqHZgWM7/QzGhgC43hVm35YjfLN2F1+vaaTpwxZ6L1zXLeCtKWfq4lKNt/iTWdjcZs791IOaSDG3bgo5ZY60AKIBv+ylmwjDZ2ayGnHkZdHfFUFK7e1QDAoL1s5m+hNlGn8oluSHd34mfCXKtDUeVuycR9+pMFvnNDP3zXKq671c2t/L9uqW299k4yKPIgyrqN0wQIRBEgCkkini4URaWiklgRNBnA84cJfk3PYNJoWcomzaDwSIXIphz7fS8p6PyKUBug/34fRkcWjjaW723Kk3FqzYcep3YCQymAy8UL+McR6nxt/z8UECJ9Ldeb1u7YBy71C91CGVX79s1R3Roy97NbdiJPpfAABX20O07+/UeLnFLopnpy+9ow4A0LrDj5pMabyZK6aOHUA0FOPcwYDGK5yRx/jScWMDAPDnrtO6t2zaskljB9DfHaHbf1XjTV5YPKKSPioAAKeaz2vGJpuRiiWld4nOAEBnaw+xG9riNWf1DBzurLEBUJMpfLvPajxzlonl6xdicZjv8tUoAgD4ms7S36392cj1uHhu41JN2c4YgJpM0fJ5K+qg9pfNle/g+fqlzH91NmabKXMAAFfb+2j+9DAypX9FvTXl1Lz/mK5UjyoAQOexHvZu+o3B6JBubkK5m+w8uxZAQPK+X5B7KHA8yI/rm7l85prGj0cGGei/1XkJBCRFA/72MNcnX2Nkz+h9SUDZAg/emimk1BRHt/luQ42nkGxyzorv8demEFuCdChxoqMPkUY27BRQIiWyVgx3xr4mMCwfm+bUhpMcCbJpFZUrBQy35420vSGhDmRpJttz4IKEzaup/Eog5F8535IHHNcKcQAAAABJRU5ErkJggg=="
    icon_48 = "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"
    icon_256 = "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"

    icon = QIcon()
    for b64_str, size in [(icon_16, 16), (icon_32, 32), (icon_48, 48), (icon_256, 256)]:
        ba = QByteArray.fromBase64(b64_str.encode("utf-8"))
        pixmap = QPixmap()
        pixmap.loadFromData(ba)
        icon.addPixmap(pixmap)
    return icon


class TabInterceptor(QObject):
    """
    Apparently tabs are handled by the main window, and passed down for certain default behaviors.
    Overriding these will make tab/shift-tab more friendly.
    """

    def __init__(self, tab_handler):
        """
        Argument tab_handler is any function taking only a boolean 'backward' as argument, to differentiate
        between tab and shift-tab.

        Specifically, this was meant to be used with TextBubbleScene.handleTabbing.
        """
        self.tab_handler = tab_handler
        super().__init__()

    def eventFilter(self, obj, event) -> bool:
        """
        Called for any event; in case of tab/shift-tab, applies the tab_handler;
        otherwise returns False (meaning event is passed through)
        """
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Tab, Qt.Key.Key_Backtab
        ):
            self.tab_handler(backward=event.key() == Qt.Key.Key_Backtab)
            return True
        return False

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


def expand_globs(files: list[str]) -> list[str]:
    """
    To enable bash-like globbing (e.g., *.wav) in Windows cmd/powershell.
    """
    expanded = []
    for f in files:
        matches = glob.glob(f)
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(f)
    return expanded


def parse_args() -> argparse.Namespace:
    """
    Defines and applies argparser, does some error handling, and returns the resulting args.
    """
    argparser = argparse.ArgumentParser(
        description='ToneSwiper is a keyboard-centered app for efficiently transcribing Dutch intonation '
                    'according to the ToDI framework.',
        epilog='For example, navigate to a folder with .wav files, and run: '
               'toneswiper *.wav --textgrid. Happy annotating!')
    argparser.add_argument('file', nargs='+', type=str, metavar='<.wav file>', help='One or more .wav files')
    group = argparser.add_mutually_exclusive_group()
    group.add_argument('-t', '--textgrid', type=str, nargs='?', const='ToDI', metavar='tier name', default=None,
                       help='Save annotations to .TextGrid files corresponding in name to the original '
                            '.wav files., to a tier with the specified name (default: "ToDI"). '
                            'If such .TextGrid files already exist, will load annotations from the given tier '
                            '(if it exists) and overwrite them.')
    group.add_argument('-j', '--json', type=str, metavar='<JSON file>',
                       help='Save annotations to the specified JSON file; if file '
                            'already exists, will also load from and overwrite it.')
    argparser.add_argument('-v', '--verbose', action='store_true',
                           help='To display a rather unsystematic assortment of logging messages for debugging.')
    argparser.add_argument('-m', '--measure', action='store_true',
                           help='To measure annotation speed, hotkey usage, corrections, playback speed, scrolling; '
                                'will be logged to a new, timestamped log file in the current working directory for each run.')

    args = argparser.parse_args()
    if os.name == "nt":
        args.file = expand_globs(args.file)
    if any(not file.endswith('.wav') for file in args.file):
        argparser.print_usage(sys.stderr)
        logger.error('error: only .wav files are currently supported.')
        exit(1)
    for file in args.file:
        if not os.path.exists(file):
            logger.error(f'error: {file} not found.')
            exit(1)

    return args


class InterceptingScrollBar(QScrollBar):
    """
    Intercepts mousepresses on the scrollbar side buttons and forwards the delta to a custom function provided
    at construction. In this case, this is used to have the scrollbar control the audioplayer directly,
    which in turn controls the actual view scroll.
    """

    def __init__(self, orientation, parent=None, interceptor=None):
        """
        The interceptor is any function that takes the scroll delta (integer) as argument.
        """
        super().__init__(orientation, parent)
        self.interceptor = interceptor
        self._custom_step = 500

    def mousePressEvent(self, event):
        """
        Overrides QScrollBar's own mousePressEvent to call self.interceptor instead.
        Other events are passed on.
        """
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        control = self.style().hitTestComplexControl(QStyle.ComplexControl.CC_ScrollBar, opt, event.position().toPoint(), self)
        if control == QStyle.SubControl.SC_ScrollBarAddLine:
            self.interceptor(+self._custom_step)
            return
        elif control == QStyle.SubControl.SC_ScrollBarSubLine:
            self.interceptor(-self._custom_step)
            return
        super().mousePressEvent(event)


def exception_hook(exc_type, exc_value, exc_traceback):
    """
    Makes sure QApplication is properly quit (thus saving annotations).
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    print("⚠️ Something went wrong:\n")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    print("\nQuitting ToneSwiper. Your annotations have been saved, if you so requested.")
    QApplication.quit()
