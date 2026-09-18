from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_swmm import SimSwmm


def test_swmm_uses_writable_temporary_report_and_output_paths():
    simulation = MagicMock()
    simulation.__enter__.return_value = simulation
    simulation.__iter__.return_value = iter(())

    with patch('sim_swmm.Simulation', return_value=simulation) as constructor, \
         patch('sim_swmm.Nodes', return_value=[]):
        SimSwmm('/read-only/scenario.inp').swmm_execute()

    _, report_path, output_path = constructor.call_args.args
    assert Path(report_path).parent == Path(output_path).parent
    assert Path(report_path).suffix == '.rpt'
    assert Path(output_path).suffix == '.out'
    assert str(Path(report_path).parent).startswith('/tmp/')
