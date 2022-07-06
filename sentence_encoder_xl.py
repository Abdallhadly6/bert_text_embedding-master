import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import xlwt
from xlwt import Workbook
  
# Workbook is created
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')
sen1 = ["go to عليكم يا اخي كيف حالك"]
sen2 = ["go to عليكم يا اخي كيف حالك"]
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

#print((embed(sen2)))
print(tf.keras.backend.get_value(embed(sen2)))
print("-------------------------------------------")
print(tf.keras.backend.get_value(embed(sen2))[0][511])
f = open("vectors.txt","a")
for j in range(10):
    for i in range(512):
        sheet1.write(i, j, str(tf.keras.backend.get_value(embed(sen2))[0][i]))    
wb.save('xlwt example2.xls')
