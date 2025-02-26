# 순환 신경망 (Recurrent Neural Network)

# Part 1 - 데이터 전처리

# 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "c:/Windows/Fonts/malgun.ttf"  # Windows의 경우
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 훈련 세트 가져오기
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# 특성 스케일링
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 60 타임스텝과 1 출력으로 데이터 구조 생성
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# 리셰이핑
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - RNN 구축

# Keras 라이브러리 및 패키지 임포트
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# RNN 초기화
regressor = Sequential()

# 첫 번째 LSTM 레이어와 드롭아웃 정규화 추가
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 두 번째 LSTM 레이어와 드롭아웃 정규화 추가
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# 출력 레이어 추가
regressor.add(Dense(units = 1))

# RNN 컴파일
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 훈련 세트에 RNN 피팅
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)

# Part 3 - 예측 및 결과 시각화

# 2017년 실제 주가 가져오기
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# 2017년 예측 주가 가져오기
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# 결과 시각화
plt.plot(real_stock_price, color = 'red', label = '실제 구글 주가')
plt.plot(predicted_stock_price, color = 'blue', label = '예측된 구글 주가')
plt.title('구글 주가 예측')
plt.xlabel('시간')
plt.ylabel('구글 주가')
plt.legend()
plt.show()
