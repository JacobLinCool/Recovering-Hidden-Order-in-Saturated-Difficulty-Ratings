"""Symbolic note IDs and source difficulty IDs used by the paper."""

NOTE_TYPES = [
    "Don",  # 0
    "Ka",  # 1
    "DonBig",  # 2
    "KaBig",  # 3
    "Roll",  # 4
    "RollBig",  # 5
    "Balloon",  # 6
    "BalloonAlt",  # 7
    "EndOf",  # 8
]

NOTE_TYPE_TO_ID: dict[str, int] = {
    note_type: i for i, note_type in enumerate(NOTE_TYPES)
}
NUM_NOTE_TYPES = len(NOTE_TYPES)
PAD_TOKEN_ID = NUM_NOTE_TYPES  # 9 for padding

DIFFICULTY_TO_ID: dict[str, int] = {
    "easy": 0,
    "normal": 1,
    "hard": 2,
    "oni": 3,
    "ura": 4,
}
