from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/GEAR-SONIC"

encoder = hf_hub_download(repo_id=REPO_ID, filename="model_encoder.onnx")
decoder = hf_hub_download(repo_id=REPO_ID, filename="model_decoder.onnx")
config  = hf_hub_download(repo_id=REPO_ID, filename="observation_config.yaml")
planner = hf_hub_download(repo_id=REPO_ID, filename="planner_sonic.onnx")

print("Policy encoder :", encoder)
print("Policy decoder :", decoder)
print("Obs config     :", config)
print("Planner        :", planner)
