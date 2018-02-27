from Chempy.parameter import ModelParameters
a=ModelParameters()
from Chempy.cem_function import posterior_function_predictions
out=posterior_function_predictions([-2.4,-2.7,-0.6,0.5,0.45],a)
print(out)
