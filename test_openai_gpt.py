# # This code is for v1 of the openai package: pypi.org/project/openai
# from openai import OpenAI
# import os
# os.environ["OPENAI_API_KEY"] = "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"
#
# client = OpenAI()
# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "user",
#       "content": ""
#     }
#   ],
#   temperature=1,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
print(text.shape)
with torch.no_grad():
    # image_features = model.encode_image(image)
    # print(image_features.shape)
    text_features = model.encode_text(text)
    print(text_features.shape)
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]