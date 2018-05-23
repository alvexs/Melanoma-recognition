import cv2
import os



# directory_image = os.path.normpath('C:\\Users\\Alexander\\Documents\\unet_arch\\validation\\melanoma_img_color')
# directory_mask = os.path.normpath('C:\\Users\\Alexander\\Documents\\unet_arch\\validation\\melanoma_mask')
directory_result_full = os.path.normpath('/Users/alexivannikov/recognizeMelanoma/recognizeApp/media')
directory_result = os.path.normpath('/Users/alexivannikov/recognizeMelanoma/recognizeApp/static')


def create_samples(image_path, mask_path):
    i=0
    x=0
    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]
    mask_name = mask_path.split('/')[-1]
    mask_name = mask_name.split('.')[0]
    for y in range(0,1):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        mask_gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(mask_gray,127,255,0)
        im3, contours, hierarchy = cv2.findContours(thresh, 1, 2)


        mask_out=cv2.subtract(mask,image)
        mask_out=cv2.subtract(mask, mask_out)


        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        try:
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            cnt = contours[0]
            #a, b, h, w = cv2.boundingRect(cnt)
            a, b, h, w = cv2.boundingRect(biggest_contour)
            if w < 10 and h < 10:
                print(str(image_name)+'size')
                pass
            #a, b, h, w = cv2.boundingRect(cnt)
            mask_out = mask_out[b:b+w, a:a+h]
            image = image[b:b+w, a:a+h]
            if not os.path.exists(directory_result):
                os.mkdir(directory_result)
            if not os.path.exists(directory_result_full):
                os.mkdir(directory_result_full)
            result_path = '%s/%s(result).jpg'%(directory_result, image_name)
            cv2.imwrite('%s/%s(result).jpg'%(directory_result_full, image_name), image)
            cv2.imwrite(result_path, mask_out)
        except:
            print(image_name)
            return None
        return ('%s(result).jpg'%image_name)
