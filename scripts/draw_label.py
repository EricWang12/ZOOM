from PIL import Image, ImageFont, ImageDraw
import os 
import numpy as np


def generate_single_grid(G, s, ds, weight, image_num = 20):

    for i in np.linspace(0, weight, image_num):


        pass



def draw_label(img, text):

    width, height = img.width, img.height

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./times-ro.ttf", 65)

    x, y = 5,6
    edge = 5

    w, h = font.getsize(text)
    h += 23
    x = width - w - edge - 2
    draw.rounded_rectangle((x-edge, y-edge, x + w +edge, y + h +edge), fill="White", outline="black", width=2, radius=7)
    draw.text((x, y+23), text, fill=(0, 0, 0), font=font)
    return img


def read_text(input_file, images):

    with open(input_file) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    i = 0
    def processNumber(line):
        line = line[line.find(':')+2:]
        return float(line)
    ret = {}
    images = sorted(images)
    for line_index in range(len(lines)):
        line = lines[line_index]
        if f"Sample Index {images[i]}" in line:
            gt = processNumber(lines[line_index+1])
            attacked = processNumber(lines[line_index+3])

            ret[images[i]] = (gt, attacked)
            i += 1
            if i >= len(images):
                break
    return ret
    
if __name__ == "__main__":

    img_dir = "/media/exx/8TB1/jinqil/stylegan/output/output-frames/afhqdog/Dog/multiple"
    outdir = "./output/dog-new"

    os.makedirs(outdir, exist_ok=True)
    goodones = [18, 26, 134 , 40, 59 , 65, 76  ,78 , 94 , 95  ,112 , 126   ,133 , 172 ,179 ,201]
    d = read_text("./log/afhqdog-2022-10-12/Dog-multiple-0.1-100-5.txt", goodones)

    for img in d:
        img_ori = Image.open(f"{img_dir}/{img}--original.png")
        img_99 = Image.open(f"{img_dir}/{img}-0099.png")
        draw_label(img_ori, f"{d[img][0]:.2f}").save(f"{outdir}/{img}-original.png")
        draw_label(img_99, f"{d[img][1]:.2f}").save(f"{outdir}/{img}-z99.png")
