import numpy as np

from benchmarks.v4.phase_3_cell_region.tools.extract_xcell_features import selected_cells_from_maps


def test_selected_cells_match_reference_order_and_keep_total_count():
    inst=np.array([[3,3,0,2],[0,1,1,2],[0,1,1,0]],dtype=np.int32)
    cls=np.array([[4,4,0,2],[0,5,5,2],[0,5,5,0]],dtype=np.uint8)
    cells,ids,total=selected_cells_from_maps(inst,cls,max_cells=2)
    # The established feature extractor takes IDs [1,2] first, then sorts them by y/x.
    assert total==3
    np.testing.assert_array_equal(ids,np.array([2,1],dtype=np.int32))
    np.testing.assert_allclose(cells[:,0],np.array([0.75,0.375],dtype=np.float32))
    np.testing.assert_array_equal(cells[:,3],np.array([2,5],dtype=np.float32))


def test_selected_cells_empty():
    cells,ids,total=selected_cells_from_maps(np.zeros((3,3),np.int32),np.zeros((3,3),np.uint8),255)
    assert cells.shape==(0,4) and ids.shape==(0,) and total==0
