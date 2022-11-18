import sys 
from transformers import AutoModel

checkpoint = sys.argv[1]
hub_name = sys.argv[2]

print(f"Pushing model from {checkpoint} to {hub_name}")

model = AutoModel.from_pretrained(checkpoint)

model.push_to_hub(hub_name)