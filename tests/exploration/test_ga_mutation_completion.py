from types import SimpleNamespace
import numpy as np
import pytest
from ase import Atoms
from gdpx.exploration.genetic_algorithm.generation import GeneticGenerationManager
from gdpx.exploration.persist.database import GlobalOptimisationDatabase
from gdpx.exploration.population.config import PopulationConfig


def make_manager(maximum_attempts=8, **completion):
    parent=Atoms('Cu2',positions=[[0,0,0],[2,0,0]])
    parent.info={'confid':99, 'nested':{'value':1}}
    config=PopulationConfig({'periodic':False,'preserve_fragments':False,
        'builders':{'init':{'method':'random_structure_improved','composition':{'Cu':2},'box':[10,10,10]}},
        'initial':{'total_size':1,'builder_allocations':[{'builder':'init','size':1}]},
        'generation':{'total_size':3}})
    config.validate_candidate=lambda *args: None
    config._require_builders=lambda *args: pytest.fail('Mutation completion must not call builders')
    selector=SimpleNamespace(select_one=lambda *args,**kwargs: parent)
    params=dict(method='genetic_algorithm',reproduction={'size':2,'maximum_attempts':2},
        mutation={'size':0},completion={'method':'mutation','maximum_attempts':maximum_attempts,**completion})
    manager=GeneticGenerationManager(params,config,object(),selector,np.random.default_rng(3))
    manager._print=lambda *args: None
    calls=[]
    manager._reproduce=lambda *args: calls.append(1)
    return manager,parent,calls


class Mutation:
    def __init__(self, fail=()): self.calls=0; self.fail=fail
    def get_new_individual(self, parents):
        self.calls+=1
        if self.calls in self.fail: return None,'failed'
        a=parents[0]
        a.positions+=self.calls*.1
        a.info['nested']['value']=2
        a.info={'data':{'parents':[99]},'key_value_pairs':{'mutation':1}}
        return a,'mutation: rattle test'


def test_completion_fills_failed_crossover_without_builders_and_reloads(tmp_path):
    manager,parent,calls=make_manager()
    db=GlobalOptimisationDatabase(tmp_path/'candidates.db')
    mutation=Mutation(fail=(1,3))
    operators={'mobile':{'mutations':mutation}}
    result=manager._prepare_current_population(db,1,{},operators)
    assert len(result)==3 and len(calls)==2 and mutation.calls==5
    assert parent.info['nested']['value']==1 and parent.positions[0,0]==0
    assert db.get_generation_plan(1)['completion_mutation_attempts']==5
    groups=manager._get_current_candidates(db,1)
    assert len(groups['mutated'])==3 and not groups['completion']
    assert all(a.info['data']['parents']==[99] for a in groups['mutated'])
    reloaded=manager._prepare_current_population(db,1,{},operators,groups)
    assert len(reloaded)==3 and mutation.calls==5


def test_attempt_exhaustion_preserves_progress_and_resumes(tmp_path):
    manager,_,_=make_manager(maximum_attempts=3)
    db=GlobalOptimisationDatabase(tmp_path/'candidates.db')
    mutation=Mutation(fail=(2,3))
    operators={'mobile':{'mutations':mutation}}
    with pytest.raises(RuntimeError,match='Mutation completion exhausted 3'):
        manager._prepare_current_population(db,1,{},operators)
    groups=manager._get_current_candidates(db,1)
    assert len(groups['mutated'])==1
    manager.completion_max_try=5
    result=manager._prepare_current_population(db,1,{},operators,groups)
    assert len(result)==3 and mutation.calls==5
    assert len(list(db.connection.select(generation=1,relaxed=0)))==3


def test_rejects_mixed_completion_configuration():
    with pytest.raises(ValueError,match='cannot specify builder'):
        make_manager(builder_proportions=[{'builder':'init','proportion':1.}])
    with pytest.raises(ValueError,match='must be builder or mutation'):
        make_manager(method='invalid')


def test_refuses_switching_in_progress_builder_completion(tmp_path):
    manager,_,_=make_manager()
    db=GlobalOptimisationDatabase(tmp_path/'candidates.db')
    db.set_generation_plan(1,{'stage':'completion','completion_sizes':[{'builder':'init','size':3}]})
    with pytest.raises(RuntimeError,match='Cannot resume builder completion'):
        manager._prepare_current_population(db,1,{}, {})
