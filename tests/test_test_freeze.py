import unittest
from pathlib import Path


class PermanentTestFreezeTests(unittest.TestCase):
 def test_training_code_has_no_test_evaluation_bypass_or_output(self):
  root=Path(__file__).parents[1]/'src/pls/training'
  forbidden=('allow-test-evaluation','test_metrics.json',"loaders['test']",'evaluation_splits')
  violations=[]
  for path in root.glob('*.py'):
   text=path.read_text()
   for token in forbidden:
    if token in text:violations.append(f'{path.name}: {token}')
  self.assertEqual(violations,[])

 def test_every_configurable_trainer_hard_rejects_test(self):
  root=Path(__file__).parents[1]/'src/pls/training';violations=[]
  for path in root.glob('train_*.py'):
   text=path.read_text()
   if "evaluate_test" in text and 'test evaluation is permanently disabled' not in text:violations.append(path.name)
  self.assertEqual(violations,[])


if __name__=='__main__':unittest.main()
