import os
import pickle
from dataclasses import dataclass, field
from typing import Optional
from pygpg.sk import GPGRegressor

@dataclass
class GpGomeaConfig:
    t: Optional[int] = field(default=3600)
    g: Optional[int] = field(default=-1)
    e: Optional[int] = field(default=499500)
    finetune_max_evals: Optional[int] = field(default=500)
    finetune: Optional[bool] = field(default=True)
    tour: Optional[int] = field(default=4)
    d: Optional[int] = field(default=4)
    pop: Optional[int] = field(default=1024)
    disable_ims: Optional[bool] = field(default=True)
    feat_sel: Optional[int] = field(default=20)
    no_univ_exc_leaves_fos: Optional[bool] = field(default=False)
    no_large_fos: Optional[bool] = field(default=True)
    bs: Optional[int] = field(default=100)
    fset: Optional[str] = field(default='+,-,*,/,log,sqrt,sin,cos')
    cmp: Optional[float] = field(default=0.0)
    rci: Optional[float] = field(default=0.0)
    verbose: Optional[bool] = field(default=True)
    random_state: Optional[int] = field(default=0)

    @classmethod
    def from_yaml(cls, yaml_string):
        pass

class GpGomeaRegressor:
    def __init__(self, config):
        self.config = config
        #self.model = self.get_model()
        self.converter = {'sub': lambda x, y : x - y,
                          'div': lambda x, y : x/y,
                          'mul': lambda x, y : x*y,
                          'add': lambda x, y : x + y,
                          'neg': lambda x    : -x,
                          'pow': lambda x, y : x**y,
                          'sin': lambda x    : sin(x),
                          'cos': lambda x    : cos(x),
                          'inv': lambda x: 1/x,
                          'sqrt': lambda x: x**0.5,
                          'pow3': lambda x: x**3,
                          }

    def get_model(self):
        model = GPGRegressor(
            t=self.config.t,
            g=self.config.g,
            e=self.config.e,
            finetune_max_evals=self.config.finetune_max_evals,
            finetune=self.config.finetune,
            tour=self.config.tour,
            d=self.config.d,
            pop=self.config.pop,
            disable_ims=self.config.disable_ims,
            feat_sel=self.config.feat_sel,
            no_univ_exc_leaves_fos=self.config.no_univ_exc_leaves_fos,
            no_large_fos=self.config.no_large_fos,
            bs=self.config.bs,
            fset=self.config.fset,
            cmp=self.config.cmp,
            rci=self.config.rci,
            random_state=self.config.random_state,
            verbose=self.config.verbose
            )
        return model
    
    def predict_single(self, X, y):
        self.model = self.get_model()
        self.model.fit(X, y)
        return self.model.model

    def benchmark(self, data):
        results = []
        for X,y in data:
            program = self.predict_single(X, y)
            results.append(program)

        return results

