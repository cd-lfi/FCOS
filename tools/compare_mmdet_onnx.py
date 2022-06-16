from deployment.pytorch2onnx import pytorch2onnx

from argparse import ArgumentParser
import os
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot, init_detector
from mmdet.models import build_detector
from mmdet.core.export import build_model_from_cfg, preprocess_example_input

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default="/home/novabot/detection/lfidetection/onnx/model.onnx", help='Path to output onnx file')
    parser.add_argument("--img-path", default="/home/novabot/detection/lfidetection/data/coco/val2017/")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    args = parser.parse_args()
    return args

def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config

def main(args):
    img_list=os.listdir(args.img_path)
    cfg=mmcv.Config.fromfile(args.config)
    model = build_model_from_cfg(args.config, args.checkpoint)
    test_img=args.img_path+random.sample(img_list,1)[0]
    input_img=args.img_path+random.sample(img_list,1)[0]
    compare_img=args.img_path+random.sample(img_list,1)[0]
    normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)
    input_shape = (1, 3) + tuple(args.shape)
    if not os.path.isfile(args.out_file):
        pytorch2onnx(model,input_img,input_shape,opset_version=11,normalize_cfg=normalize_cfg,show=False,output_file=args.out_file,verify=True,test_img=test_img,do_simplify=False,dynamic_export=True,skip_postprocess=False)
    ort_session = ort.InferenceSession(args.out_file,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    output_name = ort_session.get_outputs()[0].name
    input_name=ort_session.get_inputs()[0].name
    resizer=transforms.Resize(args.shape)
    to_tensor = transforms.ToTensor()
    ref_img=to_tensor(resizer(Image.open(compare_img)))
    onnx_detections = ort_session.run([output_name], {input_name: np.expand_dims(np.asarray(ref_img),axis=0)})[0]

    mmdet_model = init_detector(args.config, args.checkpoint, device=args.device)
    # result=inference_detector(mmdet_model, compare_img)
    result=mmdet_model(np.expand_dims(np.asarray(ref_img),axis=0))
    print(onnx_detections)
    print(result)

if __name__=='__main__':
    args = parse_args()
    main(args)




    