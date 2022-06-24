# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

from typing import Dict,Optional, Tuple

import flwr as fl
import tensorflow as tf

import logging
import time

from keras.utils.np_utils import to_categorical

import numpy as np

import health_dataset as dataset

import wandb
from datetime import datetime, timedelta 
import os
import boto3

import requests, json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


 # 날짜를 폴더로 설정
# global today_str, yesterday_str
# today= datetime.today()
# today_str = today.strftime('%Y-%m-%d')
# today_time = today.strftime('%Y-%m-%d %H-%M-%S')
# yesterday= today-timedelta(days=1)
# yesterday_str = yesterday.strftime('%Y-%m-%d')


# FL 하이퍼파라미터 설정
global num_rounds, epochs, batch_size, val_steps
num_rounds = 1
local_epochs = 1
batch_size = 32
val_steps = 5

# # server 상태 확인을 위함
# class FL_Server(BaseModel):
#     FLSeReady: bool = True

# app = FastAPI()

# FL_Se = FL_Server()

# @app.get("/FL_ST")
# def fl_server_status():
    # return FL_Se

# 참고: https://loosie.tistory.com/210, https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
# aws session 연결
def aws_session(region_name='ap-northeast-2'):
    return boto3.session.Session(aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
                                aws_secret_access_key=os.environ.get('ACCESS_SECRET_KEY'),
                                region_name=region_name)

# s3에 global model upload
def upload_model_to_bucket(global_model):
    bucket_name = os.environ.get('BUCKET_NAME')
    global today_str, latest_gl_model_v, next_gl_model
    
    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'/model/model_V{latest_gl_model_v}.h5',
        Key=global_model,
    )
    
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{global_model}"

    return s3_url

# s3에 저장되어 있는 latest global model download
def model_download():
    bucket_name = os.environ.get('BUCKET_NAME')
    print('bucket_name: ', bucket_name)
    global latest_gl_model_v, next_gl_model
    
    session = aws_session()
    s3_resource = session.client('s3')
    bucket_list = s3_resource.list_objects(Bucket=bucket_name)
    content_list = bucket_list['Contents']

    # s3 bucket 내 global model 파일 조회
    # 00시에 처음 FL 학습 돌린 모델은 전날에 수행한 global 모델 사용
    file_list=[]

    for content in content_list:
        key = content['Key']
        file_list.append(key)
    
    # model = s3_resource.download_file(bucket_name,'model_V%s.h5'%latest_gl_model_v, '/model/model_V%s.h5'%latest_gl_model_v)

    print('global model name: ', f'model_V{latest_gl_model_v}.h5')
    if 'model_V%s.h5'%latest_gl_model_v in file_list:
       model = s3_resource.download_file(bucket_name,f'model_V{latest_gl_model_v}.h5', f'/model/model_V{latest_gl_model_v}.h5')
       return model

    else:
       pass

def fl_server_start(model):

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        # tfa.metrics.F1Score(name='f1_score', num_classes=5),
        tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    logging.getLogger('flower')
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit > fraction_eval이여야 함
        fraction_fit=0.3, # 클라이언트 학습 참여 비율
        fraction_eval=0.2, # 클라이언트 평가 참여 비율
        min_fit_clients=1, # 최소 학습 참여 수
        min_eval_clients=1, # 최소 평가 참여 수
        min_available_clients=1, # 클라이언트 연결 필요 수
        eval_fn=get_eval_fn(model), # 모델 평가 결과
        on_fit_config_fn=fit_config, # batchsize, epoch 수
        on_evaluate_config_fn=evaluate_config, # val_step
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": num_rounds}, strategy=strategy)

def main() -> None:

    global num_rounds, latest_gl_model_v
    global today_str, yesterday_str

    print('')
    print('latest_gl_model_v', latest_gl_model_v)
    print('')

    
    if os.path.isfile('/model/model_V%s.h5'%latest_gl_model_v):
        print('load model')
        model = tf.keras.models.load_model('/model/model_V%s.h5'%latest_gl_model_v)
        fl_server_start(model)

    else:
        print('basic model making')
        
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            # tfa.metrics.F1Score(name='f1_score', num_classes=5),
            tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
        ]

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                16, activation='relu',
                input_shape=(6,)),
                tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='sigmoid'),
        ])
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

        fl_server_start(model)
        

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    df, p_list = dataset.data_load()

    # Use the last 5k training examples as a validation set
    x_val, y_val = df.iloc[:10000,1:7], df.loc[:9999,'label']

    # y(label) one-hot encoding
    y_val = to_categorical(np.array(y_val))

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        
        loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
        # loss, accuracy, precision, recall, f1_score, auc, auprc = model.evaluate(x_val, y_val)

        global next_gl_model, res

        # model save
        model.save("/model/model_V%s.h5"%next_gl_model)

        # wandb에 log upload
        wandb.log({'loss':loss,"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc})
        # wandb.log({'loss':loss,"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score,"auc": auc})

        
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "auprc": auprc}

        # loss, accuracy, precision, recall, auc, f1_score, auprc = model.evaluate(x_val, y_val)
        # return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "f1_score": f1_score, "auprc": auprc}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """

    global batch_size, local_epochs

    config = {
        "batch_size": batch_size,
        # "local_epochs": 1 if rnd < 2 else 2,
        "local_epochs": local_epochs,
        "num_rounds": num_rounds,
    }

    # wandb log upload
    # wandb.config.update({"local_epochs": local_epochs, "batch_size": batch_size},allow_val_change=True)

    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if rnd < 4 else 10
    global val_steps

    # wandb log upload
    wandb.config.update({"val_steps": val_steps},allow_val_change=True)
    
    return {"val_steps": val_steps}


if __name__ == "__main__":

    # uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)

    today= datetime.today()
    today_time = today.strftime('%Y-%m-%d %H-%M-%S')

    # server_status 주소
    inform_SE: str = 'http://10.152.183.18:8000/FLSe/'

    # server_status 확인 => 전 global model 버전
    server_res = requests.get(inform_SE + 'info')
    latest_gl_model_v = int(server_res.json()['Server_Status']['GL_Model_V'])
    
    # 다음 global model 버전
    next_gl_model = latest_gl_model_v + 1

    inform_Payload = {
            # 형식
            'S3_bucket': 'fl-flower-model', # 버킷명
            'S3_key': 'model_V%s.h5'%latest_gl_model_v,  # 모델 가중치 파일 이름
            'play_datetime': today_time, # server 수행 시간
            'FLSeReady': True, # server 준비 상태 on
            'Model_V' : latest_gl_model_v # GL 모델 버전
        }

    while True:
        # server_status => FL server ready
        r = requests.put(inform_SE+'FLSeUpdate', data=json.dumps(inform_Payload))
        if r.status_code == 200:
            break
        else:
            print(r.content)
        time.sleep(5)
    
    # wandb login and init
    wandb.login(key=os.environ.get('WB_KEY'))
    # wandb.init(entity='ccl-fl', project='health_flower', name='health_acc_loss v2')
    wandb.init(entity='ccl-fl', project='server_flower', name= 'server_V%s'%next_gl_model, dir='/',  \
        config={"num_rounds": num_rounds,"local_epochs": local_epochs, "batch_size": batch_size,"val_steps": val_steps, "today_datetime": today_time,
        "Model_V": next_gl_model})

    
    # s3에서 latest global model 가져오기
    model_download()
    
    try:
        # Flower Server 실행
        main()

        # s3 버킷에 global model upload
        upload_model_to_bucket("model_V%s.h5" %next_gl_model)
        
    finally:
        print('server close')
      
        # server_status에 model 버전 수정 update request
        res = requests.put(inform_SE + 'FLRoundFin', params={'FLSeReady': 'false'})
        if res.status_code == 200:
            print('global model version upgrade')
            print('global model version: ', res.json()['GL_Model_V'])

        # wandb 종료
        wandb.finish()
