# ToneSwiper

## Installation

Recommended way of installing as a globally available command, but with its own virtual environment:

```bash
pipx install git+https://github.com/mwestera/toneswiper
```

## Usage

A common approach would be to navigate to a folder with `.wav`-files to be transcribed, and do:

```bash
toneswiper *.wav
```

If your folder also contains `.TextGrid` files (with names matching the `.wav` files), and/or you want to save your annotations to such files, do:

```bash
toneswiper *.wav --textgrid
```

For more info, do:

```bash
toneswiper --help
```