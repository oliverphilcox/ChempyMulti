from Chempy.parameter import ModelParameters
a=ModelParameters()
from Chempy.cem_function import posterior_function_predictions
print(posterior_function_predictions([-2.4,-2.7,-0.4,0.6,0.4],a))
#print(out)
