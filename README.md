# pyloric
Simulator of the pyloric network in the stomatogastric ganglion (STG) in cython.

The model was proposed in Prinz, Bucher, Marder, Nature Neuroscience, 2004.

### Installation 
```
git clone https://github.com/mackelab/pyloric.git
cd pyloric
pip install .
```

### Usage
```
from pyloric import simulate, create_prior, stats
prior = create_prior()
parameter_set = prior.sample((1,))
simulation_output = simulate(parameter_set.loc[0])
summary_statistics = stats(simulation_output)
```

### Features

- simulator in cython (simulation time on single core = 2 seconds for simulated time = 10 seconds).  
- bounds for reasonable parameter ranges. Prior to sample from this range.  
- extraction of features from the voltage traces.  
- simulation at higher temperatures and with custom <img src="https://render.githubusercontent.com/render/math?math=Q_{10}"> values.  
- calculation of energy consumption of the circuit.

### Units
All conductances are in mS / <img src="https://render.githubusercontent.com/render/math?math=\text{cm}^2">.  
All voltages are in mV.  
All energies are in nJ / <img src="https://render.githubusercontent.com/render/math?math=\text{cm}^2">.  
All times are in milliseconds.  

### License
MIT
