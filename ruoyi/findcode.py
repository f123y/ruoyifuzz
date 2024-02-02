import requests
import os

import onnxruntime
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import common


def vec2Text(vec):
    vec = torch.argmax(vec, dim=1)  # 把为1的取出来
    text = ''
    for i in vec:
        text += common.captcha_array[i]
    return text


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def ma():
    path = "img_test/00.jpg"
    onnxFile = 'mathcode.onnx'

    img = Image.open(path)
    trans = transforms.Compose([
        transforms.Resize((60, 160)),
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])
    img_tensor = trans(img)
    img_tensor = img_tensor.reshape(1, 3, 60, 160)  # 1张图片 1 灰色
    ort_session = onnxruntime.InferenceSession(onnxFile)

    modelInputName = ort_session.get_inputs()[0].name
    # onnx 网络输出
    onnx_out = ort_session.run(None, {modelInputName: to_numpy(img_tensor)})
    onnx_out = torch.tensor(np.array(onnx_out))
    onnx_out = onnx_out.view(-1, common.captcha_array.__len__())
    a=eval(vec2Text(onnx_out).replace('=？', '').replace('÷', '/').replace('×', '*'))
    os.remove("img_test/00.jpg")
    return a;



