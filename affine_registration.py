import os
import glob
import ants
import numpy as np
import SimpleITK as sitk


train_path = r'D:/Transmorph_Frame/Data/training_validation_datas_5/train_nii/'
train_tumor = r'D:/Transmorph_Frame/Data/training_validation_datas_5/train_seg_nii/'
validation_path = r'D:/Transmorph_Frame/Data/training_validation_datas_5/validation_nii/'
validation_tumor = r'D:/Transmorph_Frame/Data/training_validation_datas_5/validation_seg_nii/'


validation_ct_imgs = glob.glob(validation_path + "*CT*.nii.gz")
validation_mr_imgs = glob.glob(validation_path + "*MR*.nii.gz")
validation_ct_tumors = glob.glob(validation_tumor + "*CT*.nii.gz")
validation_mr_tumors = glob.glob(validation_tumor + "*MR*.nii.gz")
validation_zipped_imgs = zip(validation_ct_imgs, validation_mr_imgs, validation_ct_tumors, validation_mr_tumors)

train_ct_imgs = glob.glob(train_path + "*CT*.nii.gz")
train_mr_imgs = glob.glob(train_path + "*MR*.nii.gz")
train_ct_tumors = glob.glob(train_tumor + "*CT*.nii.gz")
train_mr_tumors = glob.glob(train_tumor + "*MR*.nii.gz")
train_zipped_imgs = zip(train_ct_imgs, train_mr_imgs, train_ct_tumors, train_mr_tumors)
# ants图片的读取
for fixed, moving, fixed_tumor, moving_tumor in train_zipped_imgs:
    f_img = ants.image_read(fixed)
    m_img = ants.image_read(moving)
    f_tumor = ants.image_read(fixed_tumor)
    m_tumor = ants.image_read(moving_tumor)
    '''
    ants.registration()函数的返回值是一个字典：
        warpedmovout: 配准到fixed图像后的moving图像 
        warpedfixout: 配准到moving图像后的fixed图像 
        fwdtransforms: 从moving到fixed的形变场 
        invtransforms: 从fixed到moving的形变场

    type_of_transform参数的取值可以为:
        Rigid:刚体
        QuickRigid
        Affine:仿射配准，即刚体+缩放
        AffineFast
        ElasticSyN:仿射配准+可变形配准,以MI为优化准则,以elastic为正则项
        SyN:仿射配准+可变形配准,以MI为优化准则
        SyNCC:仿射配准+可变形配准,以CC为优化准则
    '''
    # 图像配准
    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='Rigid', reg_iterations=(160, 80, 40))
    # 将形变场作用于moving图像，得到配准后的图像，interpolator也可以选择"nearestNeighbor"等
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                    interpolator="linear")
    warped_tumor = ants.apply_transforms(fixed=f_tumor, moving=m_tumor, transformlist=mytx['fwdtransforms'],
                                         interpolator='nearestNeighbor')


    # 将配准后图像的direction/origin/spacing和原图保持一致
    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)
    warped_tumor.set_direction(f_tumor.direction)
    warped_tumor.set_origin(f_tumor.origin)
    warped_tumor.set_spacing(f_tumor.spacing)

    img_name = moving.replace(".nii.gz", '_warped.nii.gz')
    tumor_name = moving_tumor.replace(".nii.gz", '_warped.nii.gz')
    # 图像的保存
    ants.image_write(warped_img, img_name)
    ants.image_write(warped_tumor, tumor_name)
    print('running')
print('end')



# # 将antsimage转化为numpy数组
# warped_img_arr = warped_img.numpy(single_components=False)
# # 从numpy数组得到antsimage
# img = ants.from_numpy(warped_img_arr, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False)
# # 生成图像的雅克比行列式
# jac = ants.create_jacobian_determinant_image(domain_image=f_img, tx=mytx["fwdtransforms"][0], do_log=False, geom=False)
# ants.image_write(jac, "./result/jac.nii.gz")
# # 生成带网格的moving图像，实测效果不好

# m_grid = ants.create_warped_grid(m_img)
# m_grid = ants.create_warped_grid(m_grid, grid_directions=(False, False), transform=mytx['fwdtransforms'],
#                                  fixed_reference_image=f_img)
# ants.image_write(m_grid, "./result/m_grid.nii.gz")

