import tempfile
import unittest
from pathlib import Path

from audio.src.dataset import ASVSpoof2019LADataset


class AudioDatasetSafetyTests(unittest.TestCase):
    def test_missing_flac_raises_instead_of_using_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "ASVspoof2019_LA_train" / "flac"
            audio_dir.mkdir(parents=True)
            protocol_dir = root / "protocols"
            protocol_dir.mkdir()
            protocol_file = protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
            protocol_file.write_text(
                "speaker1 missing_audio - - bonafide\n",
                encoding="utf-8",
            )

            dataset = ASVSpoof2019LADataset(
                data_dir=root,
                protocol_dir=protocol_dir,
                split="train",
            )

            with self.assertRaisesRegex(FileNotFoundError, "missing_audio"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
