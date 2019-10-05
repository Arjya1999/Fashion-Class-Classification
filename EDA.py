# Remember the 10 classes decoding is as follows:
# 0 => T-shirt/top
# 1 => Trouser
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle boot
import pandas as pd
fashion_train_df=pd.read_csv('fashion-mnist_train.csv',sep=',')
fashion_test_df=pd.read_csv('fashion-mnist_test.csv',sep=',')
fashion_train_df.head()
fashion_train_df.tail()
fashion_test_df.head()
fashion_test_df.tail()
fashion_train_df.shape
fashion_test_df.shape