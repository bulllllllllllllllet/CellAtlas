import numpy as np
from benchmarks.v4.phase_3_cell_region.src.cells import collate_cells,encode_xcell_features


def run():
    a=np.arange(40,dtype=np.float32).reshape(10,4); batch=collate_cells([("a",a,10),("b",a[:2],2)],4)
    assert batch["cells"].shape==(2,4,4) and batch["cell_valid"].sum().item()==6 and batch["total_cell_count"].tolist()==[10.,2.]
    learned=np.zeros((3,68),dtype=np.float32); learned[:,0]=[.1,.2,.3]
    learned_batch=collate_cells([("learned",learned,3)],255)
    assert learned_batch["cells"].shape==(1,3,68)
    metadata=np.asarray([[.1,.2,3.,0.],[.3,.4,5.,6.]],dtype=np.float32)
    encoded=encode_xcell_features(metadata,np.zeros((2,64),dtype=np.float32))
    assert encoded.shape==(2,74) and np.array_equal(encoded[:,3:10].sum(1),np.ones(2,dtype=np.float32))
    assert encoded[0,3]==1 and encoded[1,9]==1
if __name__=="__main__": run()
