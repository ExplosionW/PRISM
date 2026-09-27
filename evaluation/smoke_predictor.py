from pathlib import Path
import sys,json,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'predictor'));from predict import load,predict
torch.set_num_threads(4);z=np.load(R/'tests/predictor_input.npz');expected=np.load(R/'tests/predictor_expected.npz')
for seed in range(3):
    net,targets=load(R/f'checkpoints/predictor/seed{seed}.pt');got=predict(net,z['sequences'].tolist(),z['esm33'],batch=4)
    assert targets==expected['targets'].tolist();np.testing.assert_allclose(got,expected['predictions'][seed],rtol=0,atol=2e-5)
    print(json.dumps({'seed':seed,'max_absolute_error':float(abs(got-expected['predictions'][seed]).max()),'passed':True}))
