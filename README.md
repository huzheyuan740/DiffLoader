# DiffLoader
DiffLoader: Environment-Adaptive Computation Offloading via Generative Prompt-Conditional Planning
## Usage:
To train DiffLoader:
```bash
python scripts/diffloader_mec.py --model models.Tasksmeta --diffusion models.GaussianActDiffusion --loss_type statehuber --loader datasets.RTGActMecDataset
```
To train SACLoader:
```bash
python sac.py
```
To train MoELoader:
```bash
python pmoe.py
```
To train MetaLoader:
```bash
python eval_maml_policy.py --num-workers 1 --fast-lr 0.1 --max-kl 0.01 --fast-batch-size 1 --meta-batch-size 40 --num-layers 2 --hidden-size 100 --num-batches 10000 --gamma 0.99 --tau 1.0 --cg-damping 1e-5 --ls-max-steps 15 --output-folder maml-mec-dir --device cuda --env-name MEC-MetaLoader
```