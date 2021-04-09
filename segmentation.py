import os
import torch
import sys
import numpy as np
import cv2
import timeit
from model import td4_psp18, td2_psp50, pspnet
from dataloader import preprocessor
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


torch.backends.cudnn.benchmark = True
torch.cuda.cudnn_enabled = True

def on_image(msg):
    on_image.last_image = msg
on_image.last_image = None



if __name__ == "__main__":
    prepr = preprocessor(in_size=(449, 577))
    i = 0

    rospy.init_node('segmentation_node')

    MODEL = rospy.get_param('~model', 'td2-psp50')
    GPU = rospy.get_param('~gpu', '0')
    TOPIC_IMAGE = rospy.get_param('~topic_image', 'image_raw')
    TOPIC_SEMANTIC = rospy.get_param('~topic_semantic', 'semantic')
    TOPIC_SEMANTIC_COLOR = rospy.get_param('~topic_semantic_color', 'semantic_color')
    RATE = rospy.get_param('~rate', 3.0)

    sub_image = rospy.Subscriber(TOPIC_IMAGE, Image, on_image)
    pub_semantic = rospy.Publisher(TOPIC_SEMANTIC, Image, queue_size = 1)
    pub_semantic_color = rospy.Publisher(TOPIC_SEMANTIC_COLOR, Image, queue_size = 1)

    rate = rospy.Rate(RATE)

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL=='td4-psp18':
        path_num = 4
        model = td4_psp18.td4_psp18(nclass=40,path_num=path_num,model_path=args._td4_psp18_path)

    elif MODEL=='td2-psp50':
        path_num = 2
        model = td2_psp50.td2_psp50(nclass=40,path_num=path_num,model_path=args._td2_psp50_path)

    elif MODEL=='psp101':
        path_num = 1
        model = pspnet.pspnet(nclass=40,model_path=args._psp101_path)

    model.eval()
    model.to(device)


    with torch.no_grad():
        while not rospy.is_shutdown():
            rate.sleep()

            if on_image.last_image is None:
                continue

            image = on_image.last_image

            image = prepr.load_frame(image)
                #for i, (image, img_name, folder, ori_size) in enumerate(vid_seq.data):

            image = image.to(device)

            torch.cuda.synchronize()
            start_time = timeit.default_timer()

            output = model(image, pos_id=i)

            torch.cuda.synchronize()
            elapsed_time = timeit.default_timer() - start_time


            pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)

            pred = pred.astype(np.int8)
            #pred = cv2.resize(pred, (ori_size[0]//4,ori_size[1]//4), interpolation=cv2.INTER_NEAREST)
            decoded = prepr.decode_segmap(pred)

            header = on_image.last_image.header
            #semantic = model.infer([cv_bridge.imgmsg_to_cv2(on_image.last_image)])[0]

            if pub_semantic.get_num_connections() > 0:
                m = CvBridge.cv2_to_imgmsg(decoded.astype(np.uint8), encoding='mono8')
                m.header.stamp.secs = header.stamp.secs
                m.header.stamp.nsecs = header.stamp.nsecs
                pub_semantic.publish(m)

            if pub_semantic_color.get_num_connections() > 0:
                m = CvBridge.cv2_to_imgmsg(model.color_map[decoded.astype(np.uint8)], encoding='rgb8')
                m.header.stamp.secs = header.stamp.secs
                m.header.stamp.nsecs = header.stamp.nsecs
                pub_semantic_color.publish(m)

            if i == path_num:
                i=0
            else:
                i+=1