from diffusers import DiffusionPipeline

# Optionally set non-default scheduler
# from diffusers import EulerDiscreteScheduler
# pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)

# move generator object to GPU
pipeline.to("cuda")

image = pipeline("An image of a squirrel in Picasso style").images[0]
image.save('location_of_file.png')
