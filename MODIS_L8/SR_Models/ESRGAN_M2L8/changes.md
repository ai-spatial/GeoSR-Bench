
1. move 3 python files: inference_esrgan, train, test to BasicSR folder
2. rrdbnet_arch, BasicSR/__init__
3. utils/logger
4. train_ESRGAN_x4_debug
5. to read in tif: paired_image_dataset, img_util (more clear see from paired_image_dataset)
6. to make the dataloader correct: transform.py (changed cv2.clip() function)
7. to make generator support 20 scaling facter: modify rrdbnet_arch.py (also need to add scale:20 parameter to the yml file under network_g)
8. to make perceptual loss work, only use RGB bands to feed in vgg19, modification on esrgan_model.py
9. shape 480 is not compatible with old discriminator, in discriminator_arch.py, add VGGStyleDiscriminator_s20
10. sr_model.py change save_image, img_util.py add imwrite_tif, also add new function in utils._init_

to make the model support 4 channels, only need to modify the the yml file