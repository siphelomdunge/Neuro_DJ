import json
import tempfile
import unittest
from pathlib import Path

from mvp_dj import Track, TrackSelector, _key_score, discover_folder, load_playlist


class SelectorTests(unittest.TestCase):
    def test_same_explicit_genre_is_preferred(self):
        current = Track("a", "a.wav", "A", genre="house", bpm=124, key="8A")
        house = Track("b", "b.wav", "House", genre="house", bpm=125, key="9A")
        amapiano = Track("c", "c.wav", "Amapiano", genre="amapiano", bpm=124, key="8A")
        selector = TrackSelector([current, amapiano, house])
        selector.claim(current)
        selected = selector.choose(current)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.track_id, "b")

    def test_unknown_genre_is_not_treated_as_every_genre(self):
        current = Track("a", "a.wav", "A", genre="house", bpm=124)
        unknown = Track("b", "b.wav", "Unknown", genre="open", bpm=124)
        other = Track("c", "c.wav", "Other", genre="techno", bpm=124)
        selector = TrackSelector([current, unknown, other])
        selector.claim(current)
        self.assertEqual(selector.choose(current).track_id, "b")

    def test_variety_guard_avoids_same_artist_back_to_back(self):
        current = Track("a", "a.wav", "A", artist="DJ One", genre="house", bpm=124)
        same_artist = Track("b", "b.wav", "B", artist="DJ One", genre="house", bpm=124)
        different_artist = Track("c", "c.wav", "C", artist="DJ Two", genre="house", bpm=124)
        selector = TrackSelector([current, same_artist, different_artist])
        selector.claim(current)
        self.assertEqual(selector.choose(current).track_id, "c")

    def test_manual_order_ignores_compatibility_scores(self):
        first = Track("a", "a.wav", "A", genre="house", bpm=124)
        planned_next = Track("b", "b.wav", "Planned Next", genre="techno", bpm=140)
        compatible_but_later = Track("c", "c.wav", "Compatible Later", genre="house", bpm=124)
        selector = TrackSelector(
            [first, planned_next, compatible_but_later],
            manual_order=True,
        )
        selector.claim(first)
        self.assertEqual(selector.choose(first).track_id, "b")

    def test_camelot_adjacent_keys_score_above_clash(self):
        self.assertGreater(_key_score("8A", "9A"), _key_score("8A", "1B"))

    def test_new_live_queue_items_can_be_added(self):
        first = Track("a", "a.wav", "A", genre="house")
        selector = TrackSelector([first])
        selector.claim(first)
        added = selector.add_tracks([Track("b", "b.wav", "B", genre="house")])
        self.assertEqual(added, 1)
        self.assertEqual(selector.choose(first).track_id, "b")


class InputTests(unittest.TestCase):
    def test_playlist_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            playlist = root / "party.json"
            playlist.write_text(
                json.dumps({"tracks": [{"path": "music/song.wav", "genre": "house", "bpm": 124}]}),
                encoding="utf-8",
            )
            tracks = load_playlist(playlist)
            self.assertEqual(tracks[0].genre, "house")
            self.assertEqual(Path(tracks[0].source), (root / "music/song.wav").resolve())

    def test_folder_name_becomes_genre(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            music = root / "house"
            music.mkdir()
            (music / "song_bpm_124.wav").touch()
            tracks = discover_folder(root)
            self.assertEqual(tracks[0].genre, "house")
            self.assertEqual(tracks[0].bpm, 124.0)


if __name__ == "__main__":
    unittest.main()
