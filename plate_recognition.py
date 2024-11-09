from keras.models import load_model
from PIL import Image
import numpy as np
import os,shutil,time,PIL,cv2,glob

def emptydir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
        time.sleep(2)
    os.mkdir(dirname)

def dirResize(src,dst):
    myfiles = glob.glob(src + '\\*.jpg')
    emptydir(dst)
    print(src + ' 資料夾: ')
    print( '開始轉換圖形尺寸! ')
    for f in myfiles:
        fname = f.split("\\")[-1]
        img = Image.open(f)
        img_new = img.resize((300,225),PIL.Image.LANCZOS)
        img_new.save(dst + '/' + fname)
    print('圖片轉換完成! ')

def plateRecog(ldirname,detector_path,model_path):
    label = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P',\
         'Q','R','S','T','U','V','W','X','Y','Z']

    myfiles = glob.glob(ldirname +"\\*.jpg")

    result = []
    for imgname in myfiles:    
        dirname = ldirname + "\\recogdata"
        emptydir(dirname)
        img = cv2.imread(imgname)

        detector = cv2.CascadeClassifier(detector_path)
        signs = detector.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4,minSize=(20,20))
        model = load_model(model_path)

        #擷取車牌
        if len(signs) > 0:
            for(x,y,w,h) in signs:
                image1 = Image.open(imgname)
                image2 = image1.crop((x,y,x+w,y+h))
                image3 = image2.resize((140,40),PIL.Image.LANCZOS)
                image3.save( 'tem.jpg')
                image4 = cv2.imread( 'tem.jpg')
                img_gray = cv2.cvtColor(image4,cv2.COLOR_RGB2BGR)
                _, img_thre = cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY)
                cv2.imwrite('tem.jpg', img_thre)

        #分割車牌
        image = cv2.imread('tem.jpg')

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

        contours1 = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours1[0]
        letter_image_regions = []

        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            letter_image_regions.append((x,y,w,h))
        letter_image_regions = sorted(letter_image_regions,key=lambda x:x[0])

        #存檔

        i=1
        for letter_bounding_box in letter_image_regions:
            x,y,w,h = letter_bounding_box
            if w>=5 and h>30 and h<40:
                letter_image = gray[y:y+h,x:x+w]
                letter_image = cv2.resize(letter_image,(18,38))
                cv2.imwrite(dirname+'\\{}.jpg'.format(i),letter_image)
                i += 1 

        datan = 0
        for fname in os.listdir(dirname):
            if os.path.isfile(os.path.join(dirname,fname)):
                datan +=1
        tem_data =[]
        for index in range(1,(datan+1)):
            tem_data.append((np.array(Image.open(dirname+"\\"+str(index)+".jpg")))/255.0)

        real_data = np.stack(tem_data)
        real_data1 = np.expand_dims(real_data,axis=3)
        prediction = model.predict(real_data1)
        tem_result =[]
        for i in range(len(prediction)):
            tem_result.append(label[int(np.argmax(prediction[i],axis=-1))])
        result.append(tem_result)
    for i in range (len(result)):
        print(''.join(result[i]))
    shutil.rmtree(dirname)

def main():
    detector_path = 'machineLearning\\haar_carplate.xml'
    model_path = "machineLearning\\carplate_model.hdf5"
    predictPlate_path = "machineLearning\\Chapter12\\predictPlate_sr"

    dirResize(predictPlate_path,predictPlate_path+"\\..\\predictPlate")
    plateRecog(predictPlate_path+"\\..\\predictPlate",detector_path,model_path)

if __name__=="__main__":
    main()