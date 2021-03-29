import json
from PIL import Image, ImageDraw, ImageFont

items = {
    "Step1.png": """ 
        [[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 'DOWN' 'BOMB' 'DOWN' -1 -1 -1 -1 -1 -1 'BOMB' -1 -1 -1 'DOWN' 'DOWN' -1]
         [-1 'BOMB' -1 -1 -1 'DOWN' -1 -1 -1 'WAIT' -1 -1 -1 'DOWN' -1 'WAIT' -1]
         [-1 -1 -1 -1 'BOMB' -1 'WAIT' 'DOWN' 'LEFT' -1 -1 -1 -1 'UP' -1 -1 -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 'LEFT' -1 -1 -1 -1 -1 -1 -1]
         [-1 -1 -1 'BOMB' -1 -1 -1 'DOWN' 'WAIT' -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 -1 -1 'UP' -1 'UP' -1 'BOMB' -1 'DOWN' -1 -1 -1 'UP' -1 'LEFT' -1]
         [-1 'WAIT' 'BOMB' -1 'BOMB' -1 'BOMB' -1 -1 'BOMB' -1 -1 -1 -1 -1 -1 -1]
         [-1 'WAIT' -1 'BOMB' -1 -1 -1 -1 -1 -1 -1 'DOWN' -1 -1 -1 -1 -1]
         [-1 -1 -1 'BOMB' 'LEFT' -1 'RIGHT' -1 -1 -1 'LEFT' -1 -1 -1 -1 -1 -1]
         [-1 'WAIT' -1 'UP' -1 -1 -1 'DOWN' -1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 'BOMB' -1 -1 'BOMB' -1 -1 -1 -1 -1 -1 -1 -1 'UP' 'LEFT' 'LEFT' -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 'DOWN' -1 'DOWN' -1 -1 -1 -1 -1]
         [-1 'BOMB' -1 'BOMB' -1 'WAIT' 'WAIT' 'WAIT' -1 -1 -1 -1 'WAIT' -1 -1 -1 -1]
         [-1 'UP' -1 -1 -1 -1 -1 -1 -1 -1 -1 'UP' -1 -1 -1 'DOWN' -1]
         [-1 'RIGHT' 'WAIT' -1 'UP' -1 'UP' -1 'UP' -1 'WAIT' -1 -1 -1 'LEFT' 'UP' -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]]
         """,
    "Step2.png": """ 
        [[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 'RIGHT' 'WAIT' 'DOWN' -1 -1 -1 -1 -1 -1 'BOMB' -1 -1 -1 'RIGHT' 'DOWN' -1]
         [-1 'BOMB' -1 -1 -1 'DOWN' -1 -1 -1 'UP' -1 -1 -1 'DOWN' -1 'BOMB' -1]
         [-1 -1 -1 -1 'LEFT' -1 'LEFT' 'DOWN' 'LEFT' -1 -1 -1 -1 'UP' -1 -1 -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 'LEFT' -1 -1 -1 -1 -1 -1 -1]
         [-1 -1 -1 'BOMB' -1 -1 -1 'BOMB' 'WAIT' -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 -1 -1 'UP' -1 'BOMB' -1 'BOMB' -1 'DOWN' -1 -1 -1 'DOWN' -1 'DOWN' -1]
         [-1 'DOWN' 'BOMB' -1 'BOMB' -1 'BOMB' -1 -1 'UP' -1 -1 -1 -1 -1 -1 -1]
         [-1 'LEFT' -1 'DOWN' -1 -1 -1 -1 -1 -1 -1 'WAIT' -1 -1 -1 -1 -1]
         [-1 -1 -1 'BOMB' 'LEFT' -1 'RIGHT' -1 -1 -1 'UP' -1 -1 -1 -1 -1 -1]
         [-1 'WAIT' -1 'UP' -1 -1 -1 'UP' -1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 'BOMB' -1 -1 'WAIT' -1 -1 -1 -1 -1 -1 -1 -1 'BOMB' 'LEFT' 'LEFT' -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 'DOWN' -1 'UP' -1 -1 -1 -1 -1]
         [-1 'BOMB' -1 'WAIT' -1 'WAIT' 'WAIT' 'LEFT' -1 -1 -1 -1 'LEFT' -1 -1 -1 -1]
         [-1 'UP' -1 -1 -1 -1 -1 -1 -1 -1 -1 'UP' -1 -1 -1 'DOWN' -1]
         [-1 'UP' 'WAIT' -1 'UP' -1 'UP' -1 'RIGHT' -1 'LEFT' -1 -1 -1 'BOMB' 'WAIT' -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]]
         """,
    "Step56.png": """ 
        [[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 'RIGHT' 'RIGHT' 'DOWN' 'RIGHT' 'DOWN' 'RIGHT' 'RIGHT' 'DOWN' 'DOWN' 'LEFT' 'LEFT' 'LEFT' 'DOWN' 'LEFT' 'LEFT' -1]
         [-1 'DOWN' -1 'DOWN' -1 'DOWN' -1 -1 -1 'DOWN' -1 -1 -1 'DOWN' -1 'DOWN' -1]
         [-1 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'UP' 'LEFT' 'LEFT' 'LEFT' 'LEFT' 'LEFT' 'LEFT' -1]
         [-1 'RIGHT' -1 'RIGHT' -1 -1 -1 -1 -1 'UP' -1 -1 -1 'DOWN' -1 'DOWN' -1]
         [-1 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'BOMB' 'BOMB' -1 -1 -1 -1 'BOMB' -1 'DOWN' -1]
         [-1 'WAIT' -1 'UP' -1 'UP' -1 'BOMB' -1 'DOWN' -1 -1 -1 'DOWN' -1 'DOWN' -1]
         [-1 'UP' 'RIGHT' 'UP' 'LEFT' -1 'LEFT' -1 -1 'BOMB' -1 -1 -1 'BOMB' -1 'DOWN' -1]
         [-1 'UP' -1 'UP' -1 -1 -1 -1 -1 -1 -1 'LEFT' -1 'UP' -1 'UP' -1]
         [-1 'UP' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'LEFT' -1 'UP' 'LEFT' -1 -1 'UP' -1 'WAIT' -1]
         [-1 'UP' -1 'UP' -1 -1 -1 'UP' -1 'UP' -1 -1 -1 'UP' -1 -1 -1]
         [-1 'RIGHT' 'RIGHT' 'UP' 'RIGHT' -1 -1 'UP' 'BOMB' 'UP' 'UP' 'LEFT' 'LEFT' 'UP' 'LEFT' 'WAIT' -1]
         [-1 'DOWN' -1 'DOWN' -1 'DOWN' -1 -1 -1 'DOWN' -1 'UP' -1 'UP' -1 'WAIT' -1]
         [-1 'WAIT' 'WAIT' 'RIGHT' 'RIGHT' 'RIGHT' 'DOWN' 'BOMB' -1 'BOMB' -1 'BOMB' 'LEFT' 'LEFT' -1 'WAIT' -1]
         [-1 'DOWN' -1 'WAIT' -1 'WAIT' -1 -1 -1 'UP' -1 'UP' -1 'UP' -1 'UP' -1]
         [-1 'UP' 'BOMB' 'LEFT' 'WAIT' 'RIGHT' 'RIGHT' 'LEFT' 'RIGHT' 'WAIT' 'WAIT' 'WAIT' 'WAIT' 'LEFT' 'WAIT' 'WAIT' -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]]
         """,
    "Step93.png": """ 
        [[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 'RIGHT' 'LEFT' 'RIGHT' 'RIGHT' 'LEFT' 'RIGHT' 'DOWN' 'RIGHT' 'DOWN' 'LEFT' 'DOWN' 'LEFT' 'DOWN' 'LEFT' 'DOWN' -1]
         [-1 'UP' -1 'UP' -1 'DOWN' -1 -1 -1 'DOWN' -1 'UP' -1 'DOWN' -1 'DOWN' -1]
         [-1 'DOWN' 'LEFT' 'DOWN' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'LEFT' 'DOWN' 'DOWN' 'DOWN' 'LEFT' 'DOWN' 'LEFT' 'DOWN' -1]
         [-1 'DOWN' -1 'DOWN' -1 'DOWN' -1 -1 -1 'UP' -1 'UP' -1 'DOWN' -1 'DOWN' -1]
         [-1 'UP' 'LEFT' 'LEFT' 'WAIT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'DOWN' 'LEFT' 'DOWN' 'DOWN' 'DOWN' 'UP' 'DOWN' -1]
         [-1 'BOMB' -1 'WAIT' -1 'RIGHT' -1 'LEFT' -1 'DOWN' -1 'LEFT' -1 'DOWN' -1 'DOWN' -1]
         [-1 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'RIGHT' 'LEFT' 'LEFT' 'LEFT' 'RIGHT' 'BOMB' -1 'BOMB' -1]
         [-1 'WAIT' -1 'DOWN' -1 'RIGHT' -1 -1 -1 'DOWN' -1 'DOWN' -1 'BOMB' -1 'DOWN' -1]
         [-1 'DOWN' 'RIGHT' 'DOWN' 'RIGHT' 'DOWN' 'RIGHT' 'BOMB' 'BOMB' 'LEFT' 'LEFT' 'RIGHT' 'RIGHT' 'RIGHT' 'LEFT' 'LEFT' -1]
         [-1 'DOWN' -1 'DOWN' -1 'DOWN' -1 'BOMB' -1 'DOWN' -1 'DOWN' -1 'UP' -1 -1 -1]
         [-1 'UP' 'RIGHT' 'DOWN' 'RIGHT' 'BOMB' 'RIGHT' 'BOMB' 'BOMB' 'UP' 'UP' 'UP' 'UP' 'UP' 'UP' 'LEFT' -1]
         [-1 'DOWN' -1 'DOWN' -1 'UP' -1 'UP' -1 'UP' -1 'UP' -1 'UP' -1 'UP' -1]
         [-1 'DOWN' 'UP' 'WAIT' 'RIGHT' 'RIGHT' 'RIGHT' 'UP' 'UP' 'UP' 'UP' 'UP' 'WAIT' 'UP' -1 'UP' -1]
         [-1 'RIGHT' -1 'LEFT' -1 'UP' -1 'UP' -1 'UP' -1 'LEFT' -1 'UP' -1 'UP' -1]
         [-1 'DOWN' 'RIGHT' 'RIGHT' 'LEFT' 'RIGHT' 'LEFT' 'UP' 'UP' 'UP' 'UP' 'LEFT' 'BOMB' 'LEFT' 'LEFT' 'UP' -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]]
         """,
}

# Website: https://www.google.com/get/noto/
# Source: https://noto-website-2.storage.googleapis.com/pkgs/NotoSansMono-hinted.zip
fontsize = 10
font = ImageFont.truetype("NotoSansMono-ExtraBold.ttf", fontsize)

for filename, data in items.items():

    lines = []
    for line in data.strip().split("\n"):
        line = line.strip()
        line = line.replace(" ", ", ")
        line = line.replace("'", '"')
        if line:
            lines.append(line)
    data = ",".join(lines)
    data = json.loads(data)

    image = Image.open(filename)
    draw = ImageDraw.Draw(image)

    left = 48
    top = 54
    width = 30
    height = 30

    for x in range(17):
        for y in range(17):
            text = data[y][x]
            if not isinstance(text, str):
                continue
            if len(text) < 4:
                text = " " + text
            position = x * width + left, y * height + top
            color = 255, 255, 255
            draw.text(position, text, color, font=font)

    image = image.crop((45, 45, 555, 555))
    image.save("_" + filename)
