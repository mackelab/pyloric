# pyloric
Simulator of the pyloric network in the stomatogastric ganglion (STG) in python. The solver for the differential equations is implemented in cython.

The model was proposed in Prinz, Bucher, Marder, Nature Neuroscience, 2004.

### Installation 
```
git clone https://github.com/mackelab/pyloric.git
cd pyloric
pip install .
```

### Usage
```
from pyloric import simulate, create_prior, summary_stats
prior = create_prior()
parameter_set = prior.sample((1,))
simulation_output = simulate(parameter_set.loc[0])
summary_statistics = summary_stats(simulation_output)
```

### Features

- simulator in cython (simulation time on single core = 2 seconds for simulated time = 10 seconds).  
- bounds for reasonable parameter ranges. Prior to sample from this range.  
- extraction of features from the voltage traces.  
- simulation at higher temperatures and with custom <img src="https://render.githubusercontent.com/render/math?math=Q_{10}"> values.  
- calculation of energy consumption of the circuit.

### Units
All membrane conductances are in mS / <img src="https://render.githubusercontent.com/render/math?math=\text{cm}^2">.  
All synaptic conductances are in log(mS) (natural log).
All voltages are in mV.  
All energies are in nJ / <img src="https://render.githubusercontent.com/render/math?math=\text{cm}^2">.  
All times are in milliseconds.  

### Citation
If you are using this simulator, please cite the corresponding paper:
```
@article{deistler2022energy,
  title={Energy-efficient network activity from disparate circuit parameters},
  author={Deistler, Michael and Macke, Jakob H and Gon{\c{c}}alves, Pedro J},
  journal={Proceedings of the National Academy of Sciences},
  volume={119},
  number={44},
  pages={e2207632119},
  year={2022},
  publisher={National Acad Sciences}
}
```

### License
MIT
