#Crop_FullSize.py
#STEP_1

import cv2, pathlib
ROI_L, ROI_R = LEFT_COORD, RIGHT_COORD    					
ROI_T, ROI_B = TOP_COORD, BOTTOM_COORD  
SRC = pathlib.Path(r"FULLIMG_INPUT")  			 
DST = pathlib.Path(r"CROP_OUTPUT")     		 
DST.mkdir(exist_ok=True)
def fixed_crop(img):
    h, w = img.shape[:2]
    x0, x1 = int(w*ROI_L), int(w*ROI_R)
    y0, y1 = int(h*ROI_T), int(h*ROI_B)
    return img[y0:y1, x0:x1]
for p in SRC.glob("*.jpg"):
    im  = cv2.imread(str(p))
    out = fixed_crop(im)
    cv2.imwrite(str(DST / p.name), out)
    print(f"{p.name:>12}  ->  {out.shape[1]}×{out.shape[0]} px")
