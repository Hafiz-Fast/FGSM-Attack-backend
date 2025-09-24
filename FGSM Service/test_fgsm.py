import requests, base64, io
from PIL import Image

url = "http://127.0.0.1:8000/attack"
files = {'file': open('Images/pacman.png','rb')}
data = {'epsilon': '0.1'}
r = requests.post(url, files=files, data=data)
res = r.json()
print("Clean:", res['clean_prediction'])
print("Adv:", res['adversarial_prediction'])
print("Attack success:", res['attack_success'])
b64 = res['adversarial_image_base64']
img = Image.open(io.BytesIO(base64.b64decode(b64)))

target_size = (280, 280)
resample_method = Image.NEAREST

resized = img.resize(target_size, resample=resample_method)
resized.save("Images/adv_out3_upscaled.png")

# img.save("adv_out2.jpg")