# ToneSwiper

![](https://github.com/mwestera/toneswiper/blob/master/toneswiper.png)

## Transcription of Dutch Intonation (ToDI)

The [ToDI framework](https://todi.cls.ru.nl/) describes intonation in the Dutch language as a series of prosodic events, such as rising (`L*H`) and falling (`H*L`) pitch accents, and high (`H%`) and low (`L%`) boundaries. 

ToneSwiper facilitates manual transcription of intonation using this framework, by making it more efficient and more enjoyable. Specifically, it enables speedy transcription in real-time, in sync with the audio, through intuitive hotkey combinations.

Future versions of this program may support other ToDI/ToBI-like transcription frameworks, suitable for other languages.

## Installation

You can install ToneSwiper from the Python Package Index:

```bash
pip install toneswiper
```

Or use `pipx` to install it in its own virtual environment (see `pipx` [installation instructiond](https://pipx.pypa.io/latest/installation/)).

To install the latest version potentially in development, install directly from the git repository (with `pip` or `pipx`): 

```bash
pip install git+https://github.com/mwestera/toneswiper
```

## Usage

On the command-line, a typical usage would be to navigate to a folder with one or more `.wav`-files (`cd some/folder/with/wav/files`) to be transcribed, and do:

```bash
toneswiper *.wav
```

This will start the gui app to let you annotate the selected sound files. It can be almost exclusively controlled by the keyboard; press `F1` to open a help window explaining the keyboard controls.

If your folder also contains `.TextGrid` files (with names matching the `.wav` files), as exported from Praat, and/or you want to save your annotations to such files, you can do the following:

**🌩 WARNING 🌩** This will modify your `.TextGrid` files by adding a 'ToDI' tier, and/or modifying it if the tier already exists. It may also destroy your files altogether, so best do this only on a duplicate of your 'real' files.    

```bash
toneswiper *.wav --textgrid
```

You can also customize the tier to which the annotations are saved:

```bash
toneswiper *.wav --textgrid todi2
```

For more info, do:

```bash
toneswiper --help
```