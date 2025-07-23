This instruction guides you to reproduce results from LLM Drive. 
(1) Navigate through TransFuser++ GitHub repo (https://github.com/autonomousvision/carla\_garage) and download TransFuser++ dataset. Use TPpp_dataset for a simpler test. 
(2) Run 'create\_dataset.py' to generate dataset. 
(3) Run 'split.py' to split the dataset. (Optional)
(4) Run 'generate\_explanations.py' to generate explanations. This inference step has a very high demand of computing resources. 
(5) Run 'main\_distil.py', 'main_base.py', 'main_large.py' for training and evaluation. This inference step has a moderate demand of computing resources. 
