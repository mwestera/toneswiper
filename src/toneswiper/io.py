import logging
import json
import tgt
import os


def load_from_json(path):
    """
    Loads annotations from json.
    """
    logging.warning(f'Loading from existing file {path}; will be modified.')
    with open(path, 'r') as file:
        from_json = json.loads(file.read())
    return from_json


def write_to_json(wavfiles, transcriptions, to_file):
    """
    Write a dictionary mapping wav filenames to transcriptions (lists of pairs).
    If to_file is None, prints to stdout.
    """
    for_json = {str(file): transcription for file, transcription in zip(wavfiles, transcriptions)}
    if to_file is not None:
        with open(to_file, 'w') as file:
            file.write(json.dumps(for_json))
    else:
        print(json.dumps(for_json))


def load_from_textgrids(wav_paths: list[str], tier: str) -> dict[str, list[tuple[float,str]]]:
    """
    For a series of .wav file paths, looks for corresponding .TextGrid files.
    Loads annotations from the provided point tier if they exist.
    Returns a dictionary from wav filenames to list of time-stamped annotations (float, str) pairs.
    """
    from_textgrids = {}
    will_be_modified = False
    for wavfile in wav_paths:
        textgrid_path = wavfile.replace('.wav', '.TextGrid')
        if not os.path.exists(textgrid_path):
            from_textgrids[wavfile] = []
            logging.warning(f'No textgrid found for {wavfile}')
            continue
        textgrid = tgt.io.read_textgrid(textgrid_path)
        if not textgrid.has_tier(tier):
            from_textgrids[wavfile] = []
            continue
        will_be_modified = True
        transcription = [(p.time * 1000, p.text) for p in textgrid.get_tier_by_name(tier).points]
        from_textgrids[wavfile] = transcription
    if will_be_modified:
        logging.warning(f'Loaded from existing textgrids; existing tier {tier} will be modified.')
    return from_textgrids


def write_to_textgrids(transcriptions, paths, duration, tier):
    for textgrid_path, transcription in zip(paths, transcriptions):
        basename = os.path.splitext(os.path.basename(textgrid_path))[0]
        if os.path.exists(textgrid_path):
            textgrid = tgt.io.read_textgrid(textgrid_path)
            if textgrid.has_tier(tier):
                textgrid.delete_tier(tier)
        else:
            textgrid = tgt.core.TextGrid(basename)
        points = [tgt.core.Point(time/1000, text) for time, text in transcription]
        textgrid.add_tier(tgt.core.PointTier(start_time=0, end_time=duration/1000,
                                             name=tier, objects=points))
        tgt.write_to_file(textgrid, textgrid_path, format='long')
