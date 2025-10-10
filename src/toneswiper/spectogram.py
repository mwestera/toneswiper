import functools

from PyQt6.QtWidgets import QGraphicsPixmapItem
from PyQt6.QtGui import QImage, QPixmap

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import parselmouth
import warnings


@functools.cache
def make_image_cached(path, width, height):
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    pitch, spec, xmin, xmax = make_spectogram_cached(path)

    draw_spectrogram(spec, ax=ax)
    ax2 = ax.twinx()
    draw_pitch(pitch, ax=ax2)

    ax.set_xlim(xmin, xmax)

    rectangle = (0, 0, 1, 1)
    ax.set_position(rectangle)
    ax2.set_position(rectangle)

    canvas.draw()
    width_px, height_px = canvas.get_width_height()
    image = QImage(canvas.renderer.buffer_rgba(), width_px, height_px, QImage.Format.Format_RGBA8888)

    pixmap = QPixmap.fromImage(image)
    item = QGraphicsPixmapItem(pixmap)
    return item


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
    spec = pre.to_spectrogram(window_length=0.05, time_step=0.02, frequency_step=25, maximum_frequency=4000)
    return pitch, spec, snd.xmin, snd.xmax


def draw_spectrogram(spec, ax, dynamic_range=70):
    """
    From parselmouth spectogram to a matplotlib plot.
    """
    data = 10 * np.log10(np.maximum(spec.values, 1e-10))
    vmax = data.max()
    vmin = vmax - dynamic_range

    X, Y = spec.x_grid(), spec.y_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sg_db = 10 * np.log10(spec.values)
    ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot', alpha=.5)
    ax.axis(ymin=spec.ymin, ymax=spec.ymax)

    ax.imshow(data, origin='lower', aspect='auto', cmap='gray', extent=[spec.xmin, spec.xmax, 0, spec.ymax], vmin=vmin,
              vmax=vmax)
    ax.set_ylabel('Frequency (Hz)')


def draw_pitch(pitch, ax):
    """
    From parselmouth pitch track to a matplotlib plot.
    """
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    times = pitch.xs()
    ax.plot(times, pitch_values, color=(0, 1, 1), linewidth=2)
    # ax.set_ylim(0, 1000)
    ax.set_ylabel('Pitch (Hz)')

