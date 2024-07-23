# debug information about LLM
model_name = "CohereForAI/c4ai-command-r-plus" # @param {type: "string"}

model_config = AutoConfig.from_pretrained(model_name)

hidden_layers = model_config.num_hidden_layers
hidden_size = model_config.hidden_size
attention_heads = model_config.num_attention_heads

#Number of parameters in the model (in billions)
nb_billion_parameters = 104 # @param {type:"number"}
print("Number of parameters in the model (n): "+str(nb_billion_parameters)+"B")

#Precision of the parameters in the model
bitwidth_model = 16 # @param {type:"integer"}
print("Bitwidth of the model's parameters (p): "+str(bitwidth_model)+"-bit")

#Precision of the parameters in the optimizer
bitwidth_optimizer = 32 # @param {type:"integer"}
print("Bitwidth of the optimizer's parameters (o): "+str(bitwidth_optimizer)+"-bit")

#The maximum number of tokens in a sequence
seqlen = 512 # @param {type:"integer"}
print("Sequence length (s): "+str(seqlen))

#The batch size
batch_size = 8 # @param {type:"integer"}
print("Batch size (b): "+str(batch_size))

def estimate_consumption():
  #34 sbh + 5as²b
  return round((34*seqlen*batch_size*hidden_size + 5*attention_heads*seqlen*seqlen*batch_size)*2/(1024**3),2)

def estimate_optimizer_size():
  return round((2*nb_billion_parameters*bitwidth_optimizer/8*(1000**3))/(1024**3),2)

def estimate_model_size():
  return round(nb_billion_parameters*bitwidth_model/8*(1000**3)/(1024**3),2)

activation_consumption = estimate_consumption()
model_consumption = estimate_model_size()
optimizer_consumption = estimate_optimizer_size()

print("Memory consumption of the model: "+str(model_consumption)+" GB\n")

print("Memory consumption of the optimizer: "+str(optimizer_consumption)+" GB")
print("Memory consumption of activations for fine-tuning: "+str(activation_consumption*hidden_layers)+" GB")
print("Total memory consumption for fine-tuning: "+str(model_consumption+optimizer_consumption+activation_consumption*hidden_layers)+" GB\n")

print("Memory consumption of activations for inference: "+str(activation_consumption)+" GB")
print("Total memory consumption for inference: "+str(model_consumption+activation_consumption)+" GB")

print("Model: "+str(model_name))
print("Hidden Layers (L): "+str(hidden_layers))
print("Hidden Size (h): "+str(hidden_size))
print("Attention Heads (a): "+str(attention_heads))
