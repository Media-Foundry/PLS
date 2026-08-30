import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import select_esol_partial_candidates as selector


class EsRecursiveSelectionTests(unittest.TestCase):
 def test_legacy_structure_sweep_uses_frozen_first_report(self):
  with tempfile.TemporaryDirectory() as directory:
   report=Path(directory)/'report.json';report.write_text(json.dumps({'sequence_runs':['sequence'],'structure_runs':['geometry','tree'],'reports':[{'structure_weights':[.1,.2]}]}));expected=(np.asarray([.2]),np.asarray([.3]),np.asarray([4]))
   with patch.object(selector,'selected_prediction',return_value=expected) as selected:self.assertIs(selector.frozen_prediction(report),expected);selected.assert_called_once_with(['sequence'],'geometry','tree',[.1,.2])
 def test_objectives_have_higher_is_better_orientation(self):
  truth=np.asarray([0.,1.,2.]);good=np.asarray([0.,1.,2.]);bad=np.asarray([2.,1.,0.])
  for objective in ('spearman','pearson','rmse','mae'):self.assertGreater(selector.objective_score(truth,good,objective),selector.objective_score(truth,bad,objective))
 def test_recursive_report_reconstructs_frozen_partial_candidate(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);base=root/'base.json';nested=root/'nested.json'
   base.write_text(json.dumps({'sequence_runs':['sequence'],'geometry_run':'geometry','leaf_run':'leaf','leaf_structure_weights':[1.0],'full_run':'full','full_structure_weights':[1.0],'leaf_weight':.6,'full_weight':.4}))
   nested.write_text(json.dumps({'base_report':str(base),'candidate_weights':{'candidate':.25}}))
   targets=np.asarray([.1,.9]);entities=np.asarray([11,22]);leaf=np.asarray([.2,.4]);full=np.asarray([.3,.5])
   with patch.object(selector,'selected_prediction',side_effect=[(targets,leaf,entities),(targets,full,entities)]),patch.object(selector,'load',return_value=(targets[:1],np.asarray([.8]),entities[:1])):
    observed,prediction,observed_entities=selector.frozen_prediction(nested)
   np.testing.assert_array_equal(observed,targets);np.testing.assert_array_equal(observed_entities,entities);np.testing.assert_allclose(prediction,np.asarray([.24+.25*(.8-.24),.44]))


if __name__=='__main__':unittest.main()
