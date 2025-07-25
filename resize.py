import os
import cv2
import tifffile

def divide_by_two(path):
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    name, ext = filename.split('.')

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]

    # divide by 2
    image_s2 = cv2.resize(image, (w//2, h//2))
    filename = os.path.join(dirname, f'{name}_s2.{ext}')
    cv2.imwrite(filename, image_s2, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    # divide by 4
    image_s4 = cv2.resize(image_s2, (w//4, h//4))
    filename = os.path.join(dirname, f'{name}_s4.{ext}')
    cv2.imwrite(filename, image_s4, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    # divide by 8
    image_s8 = cv2.resize(image_s4, (w//8, h//8))
    filename = os.path.join(dirname, f'{name}_s8.{ext}')
    cv2.imwrite(filename, image_s8, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    # divide by 16
    image_s16 = cv2.resize(image_s8, (w//16, h//16))
    filename = os.path.join(dirname, f'{name}_s16.{ext}')
    cv2.imwrite(filename, image_s16, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

if __name__ == '__main__':
    # path = './evol2/02000.tif'
    # divide_by_two(path)

    # x0, y0 = 4296, 2232
    # data = tifffile.imread('./evol2/02000.tif')
    # tifffile.imwrite(f'./evol2/02000_x{x0}_y{y0}.tif', data[y0:y0+500, x0:x0+500])
    # tifffile.imwrite(f'./evol2/02000_x{x0}_y{y0+500}.tif', data[y0+500:y0+1000, x0:x0+500])
    # tifffile.imwrite(f'./evol2/02000_x{x0+500}_y{y0}.tif', data[y0:y0+500, x0+500:x0+1000])
    # tifffile.imwrite(f'./evol2/02000_x{x0+500}_y{y0+500}.tif', data[y0+500:y0+1000, x0+500:x0+1000])

    x0, y0 = 4296//2, 2232//2
    data = tifffile.imread('./evol2/02000_s2.tif')
    tifffile.imwrite(f'./evol2/02000_s2_x{x0}_y{y0}.tif', data[y0:y0+500, x0:x0+500])

    # y0, x0, h, w = 924, 1632, 500, 500
    # data = tifffile.imread('./evol2/02000.tif')
    # data = data[y0:y0+h, x0:x0+w]

    # tifffile.imwrite(f'./evol3/02000_x{x0}_y{y0}.tif', data)

