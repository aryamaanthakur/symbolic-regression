import os
import pickle
from dataclasses import dataclass, field
from typing import Optional
from sympy import sympify, sin, cos

from gplearn.genetic import SymbolicRegressor
@dataclass
class GpLearnConfig:
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

class GpLearnRegressor:
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
        model = SymbolicRegressor(population_size=self.config.population_size,
                                  generations=self.config.generations,
                                  stopping_criteria=self.config.stopping_criteria,
                                  p_crossover=self.config.p_crossover,
                                  p_subtree_mutation=self.config.p_subtree_mutation,
                                  p_hoist_mutation=self.config.p_hoist_mutation,
                                  p_point_mutation=self.config.p_point_mutation,
                                  max_samples=self.config.max_samples,
                                  verbose=self.config.verbose,
                                  parsimony_coefficient=self.config.parsimony_coefficient,
                                  random_state=self.config.random_state,
                                  function_set=self.config.function_set)
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

