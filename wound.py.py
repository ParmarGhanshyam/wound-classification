import cv2
import numpy as np

img = cv2.imread('/minisample.jpeg')

cv2.imshow('img',img)



hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])

lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30,255,255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([350,55,100])



mask_red = cv2.inRange(hsv, lower_red, upper_red)

mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

mask_black = cv2.inRange(hsv, lower_black, upper_black)



res_red = cv2.bitwise_and(img,img, mask= mask_red)

res_yellow = cv2.bitwise_and(img,img, mask= mask_yellow)

res_black = cv2.bitwise_and(img,img, mask= mask_black)



kernel = np.ones((7,7),np.uint8)

mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)


mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)



contours_red, hierarchy_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_for_red = cv2.drawContours(res_red, contours_red, -1, (255,255,0), 1)

try : 
    cnt_red = contours_red[0]

    epsilon_red = 0.1 * cv2.arcLength(cnt_red,True)
    approx_red = cv2.approxPolyDP(cnt_red,epsilon_red,True)

    print("approx_red",approx_red)


    cv2.imshow("Output_red", output_for_red)

    ratio_red = cv2.countNonZero(mask_red)/(img.size)

    colorPercent = (ratio_red * 100)

    print('Red pixel percentage:', np.round(colorPercent, 2)*3)


    cv2.imshow("images", np.hstack([img, output_for_red]))

except:
    print("No Red colur in the picture")




contours_yellow, hierarchy_yellow = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_for_yellow = cv2.drawContours(res_yellow, contours_yellow, -1, (255,255,0), 1)

try : 
    cnt_yellow = contours_yellow[0]

    epsilon = 0.1 * cv2.arcLength(cnt_yellow,True)
    approx = cv2.approxPolyDP(cnt_yellow,epsilon,True)

    print("approx",approx)


    cv2.imshow("Output_for_yellow", output_for_yellow)
    ratio_yellow = cv2.countNonZero(mask_yellow)/(img.size)

    colorPercent = (ratio_yellow * 100)

    print('yellow pixel percentage:', np.round(colorPercent, 2)*3)

    cv2.imshow("images", np.hstack([img, output_for_yellow]))

except:
    print("No yellow colour in the picture")




contours_black, hierarchy_black = cv2.findContours(mask_black.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_for_black = cv2.drawContours(res_black, contours_black, -1, (255,255,0), 1)

try : 
    
    cnt_black = contours_black[0]

    epsilon = 0.1 * cv2.arcLength(cnt_black,True)
    approx = cv2.approxPolyDP(cnt_black,epsilon,True)

    print("approx",approx)


    cv2.imshow("Output_for_black", output_for_black)
    ratio_black = cv2.countNonZero(mask_black)/(img.size)

    colorPercent = (ratio_black * 100)

    print('Black pixel percentage:', np.round(colorPercent, 2)*3)

    cv2.imshow("images", np.hstack([img, output_for_black]))

except:
    print("No black colour in the picture")



cv2.waitKey(0)

cv2.destroyAllWindows()

