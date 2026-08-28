import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pls.features.external_v4_loader import load_external_v4


class ExternalV4LoaderTests(unittest.TestCase):
    def test_migrated_source_loads_with_provenance(self):
        root = Path("/home/pc/Code/BIO/protein")
        if not root.is_dir(): self.skipTest("external BIO/protein source is unavailable")
        module, hashes = load_external_v4(root)
        self.assertEqual(len(module.get_spatial_scalar_feature_names_v4()), 89)
        self.assertEqual(len(hashes), 7)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__": unittest.main()
