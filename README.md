# MLP-NAS

* Clone using: 

```
git clone --recurse-submodules https://github.com/TomasGadea/MLP-NAS.git
cd MLP-NAS
```

* Create python3.9.13 environ (python3 version of _scar_):
```
python3 -m venv environ
source environ/bin/activate
pip install -r requirements.txt
```
* Ask @TomasGadea for config files: `config.json` and add your [wandb](https://wandb.ai/site) API key (this last thing is optional)

## Train Search

(Open a `tmux` session rooted in `MLP-NAS`)

```
sh execute.sh
```
To see all available params check [`main.py`](https://github.com/TomasGadea/MLP-NAS/blob/main/main.py).

* `--use-amp` stores `True` when added and uses `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`.
* `--wandb` stores `True` and logs into your wandb account using your API (optional).


## Retrain Fixed architecture found with Train Search

(Open a `tmux` session rooted in `MLP-NAS`)

```
sh fixed_execute.sh
```
To see all available params check [`fixed_main.py`](https://github.com/TomasGadea/MLP-NAS/blob/main/fixed_main.py).

* `--path-to-supernet` is the output path of any past experiment of `execute.sh`. Check the example in `fixed_execute.sh`.

## Output Files

Output files for **Train Search** are:

* `flops_table.txt`: string formatted table of n_params and flops of model.
* `log.csv`: metrics such as acc, F, mmc, along epochs.
* `params.json`: parameters that include all arguments called in `execute.sh` and other extra info.
* `W.pt`: Last version of the model saved in PyTorch format after all training epochs.
* `W_test.pt`: Best version of the model saved in PyTorch format after all training epochs.

Output files for **Retrain Fixed** are:

* `flops_table.txt`: string formatted table of n_params and flops of model.
* `log.csv`: metrics such as acc, mmc, along epochs.
* `params.json`: parameters that include all arguments called in `fixed_execute.sh` and other extra info.
* `W.pt`: Last version of the model saved in PyTorch format after all training epochs.

Retrain Fixed files are stored in `out/retrain/` dir, unlike Train Search that are directly into `out/` directoy. They can be modified however using the `--output` arg if desired.
