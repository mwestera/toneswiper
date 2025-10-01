# ToneSwiper

![](toneswiper.png)

## Installation

Recommended way of installing as a globally available command, but with its own virtual environment, is as follows. If your system does not have `pipx` yet (for installing Python programs), you first need to install that, see [here](https://pipx.pypa.io/latest/installation/).

```bash
pipx install git+https://github.com/mwestera/toneswiper
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