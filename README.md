# Optimizing-for-Solar-Cells-Structure
&nbsp;&nbsp; This repository apply ML and heuristic algorithms for optimizing structures of tandem solar cells. In [Modeling](https://github.com/HKjoe/Optimizing-for-Solar-Cells-Structure/tree/master/Modeling) file, it include source code of building and training CPN model. Furthermore, ModelConfig.py is some configurations about CPN while training. CPN.py is main code for training model. 5to3.h5 is our trained model for replace simulation tools. WHUT_more_data.npy is simulation dataset, which includes 12500 sets.  
&nbsp;&nbsp; You can train you own model by run CPN.py, after configuring required parameters in ModelConfig.py. Model also can be saved by adopted .savemodel(path) attribute in CPN.py.
```python
python CPN.py
```

