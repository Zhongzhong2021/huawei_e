"""Fixed, paired placeholder and co-located missingness diagnostics on validation."""
import numpy as np
from .missing import select_positions, stable_seed
from .observation import text_state

SCENARIOS = ['clean'] + [f'{m}_{r}_{s}' for m in ['text', 'joint']
                         for r in [10,30,50] for s in ['random','scattered']]

def apply_condition(data, scenario, policy):
    if scenario not in SCENARIOS:
        raise ValueError('Condition must belong to the fixed validation protocol')
    result = {k:v.copy() for k,v in data.items()}
    selected = np.zeros_like(data['valid'])
    if scenario != 'clean':
        modality,rate,shape = scenario.split('_')
        for i,sid in enumerate(data['ids']):
            rng=np.random.default_rng(stable_seed(sid,scenario,20260926))
            selected[i,select_positions(data['valid'][i],int(rate)/100,shape,rng)]=True
        result['tokens'][selected]=100
        if modality=='joint':
            for m,name in [(1,'audio'),(2,'vision')]:
                result[name][selected]=0
                result['observed'][:,:,m][selected]=False
    # Use original presence, not the candidate policy's already-reduced mask.
    present=data.get('text_present',data['observed'][:,:,0])
    states=text_state(result['tokens'],result['valid'],present,policy)
    result['observed'][:,:,0]=states.pop('text_observed')
    result.update(states)
    return result,selected
