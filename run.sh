rm -rf results/*

python3 -m dna.train --steps 1000 --n-layers 2 --d-model 64 --pos rope --results results/dna_train_exp_layers.jsonl
python3 -m dna.train --steps 1000 --n-layers 4 --d-model 64 --pos rope --results results/dna_train_exp_layers.jsonl
python3 -m dna.train --steps 1000 --n-layers 8 --d-model 64 --pos rope --results results/dna_train_exp_layers.jsonl

python3 -m dna.plot results/dna_train_exp_layers.jsonl 

python3 -m dna.train --steps 1000 --n-layers 4 --d-model 32 --pos rope --results results/dna_train_exp_d.jsonl
python3 -m dna.train --steps 1000 --n-layers 4 --d-model 64 --pos rope --results results/dna_train_exp_d.jsonl
python3 -m dna.train --steps 1000 --n-layers 4 --d-model 128 --pos rope --results results/dna_train_exp_d.jsonl


python3 -m dna.plot results/dna_train_exp_d.jsonl 