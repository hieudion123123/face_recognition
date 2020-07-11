from annoy import AnnoyIndex
import numpy as np
from imutils import paths
from tqdm import tqdm
# Sô lượng cây Annoy xây dựng = 100
NUMBER_OF_TREES = 100
f = 128
# Trả về một index mới với f là số chiều của vector đó với metric là "angular"
# mectric là khoảng cách đại số giữa 2 vector trong khôn gian
t = AnnoyIndex(f, 'angular')
# Load các dữ liệu trong tập ảnh trong dữ liệu ban đầu
imagePaths = list(paths.list_images('path_of_you'))
# Hàm dùng để mã hóa khuôn mặt trong ảnh thành vector trong không gian euclidean 128 chiều
def image_encoding(imagePath):
    # Dùng thư viện face_recognition để load các ảnh thay cho OpenCV
    img = face_recognition.load_image_file(imagePath)
    # Dùng hàm face_recognition.face_location để xác dịnh khuôn mặt trong ảnh (trả về vị trí top,bot,lelf,right) của khuôn mặt trong ảnh)
    img_ = face_recognition.face_locations(img)
    top, right, bottom, left = [ v for v in img_[0] ]
    face = img[top:bottom, left:right]
    # Mã hóa ảnh thành các vector 128 chiều
    img_emb = face_recognition.face_encodings(face)[0]
    return img_emb


for i, imagePath in tqdm(enumerate(imagePaths)):
    # Gọi hàm mã hóa ảnh
    img_emb = image_encoding(imagePath)
    # Thêm item i với vector v
    t.add_item(i, img_emb)
# Xây dựng các cây Annoy
t.build(NUMBER_OF_TREES) # 100 cây
t.save('images.ann')