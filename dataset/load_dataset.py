from datasets import load_dataset

ds = load_dataset("LLM4Code/SATBench")
print(ds)
print(ds["train"][0].keys())
print(ds["train"][0]["scenario"])
print(ds["train"][0]["conditions"])
print(ds["train"][0]["question"])
print(ds["train"][0]["satisfiable"])