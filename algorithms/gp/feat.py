import os
import pickle
from dataclasses import dataclass, field
from typing import Optional
from sympy import sympify, sin, cos
from feat import FeatRegressor as FR

@dataclass
class FeatConfig:
    population_size: Optional[int] = field(default=5000)
    generations: Optional[int] = field(default=20)
    stopping_criteria: Optional[float] = field(default=0.01)
    p_crossover: Optional[float] = field(default=0.7)
    p_subtree_mutation: Optional[float] = field(default=0.1)
    p_hoist_mutation: Optional[float] = field(default=0.1)
    p_point_mutation: Optional[float] = field(default=0.05)
    max_samples: Optional[float] = field(default=0.9)
    verbose: Optional[int] = field(default=1)
    parsimony_coefficient: Optional[float] = field(default=0.01)
    function_set: Optional[list] = field(default_factory=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', ])
    random_state: Optional[int] = field(default=42)

    @classmethod
    def from_yaml(cls, yaml_string):
        pass

class FeatRegressor:
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
        model = FR(
            pop_size=100,
            gens=100,
            max_time=8*60*60,  # 8 hrs
            max_depth=6,
            verbosity=2,
            batch_size=100,
            functions=['+','-','*','/','^2','^3','sqrt','sin','cos','exp','log'],
            otype='f'
        )
        
        return model
    
    def predict_single(self, X, y, sympify_expr=True):
        self.model = self.get_model()
        self.model.fit(X, y)
        if sympify_expr:
            return sympify(str(self.model._program), locals=self.converter)
        return self.model._program

    def benchmark(self, data, sympify_expr=True):
        results = []
        for X,y in data:
            program = self.predict_single(X, y, sympify_expr)
            results.append(program)

        return results

