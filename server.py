# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

from typing import Dict,Optional, Tuple

import flwr as fl
import tensorflow as tf
import tensorflow_addons as tfa

from keras.utils.np_utils import to_categorical

import numpy as np

import health_dataset as dataset

import wandb

# wandb login and init
wandb.login(key='6266dbc809b57000d78fb8b163179a0a3d6eeb37')
wandb.init(entity='ccl-fl', project='flower',config={"epochs": 4, "batch_size": 32,"val_steps": 4})

def main() -> None:
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

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            16, activation='relu',
            input_shape=(6,)), # 학습 고려 컬럼 6개(bpm, spo2, rr, sbp, dbp,mbp)
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit > fraction_eval이여야 함
        fraction_fit=0.3, # 클라이언트 학습 참여 비율
        fraction_eval=0.2, # 클라이언트 평가 참여 비율
        min_fit_clients=10, # 최소 학습 참여 수
        min_eval_clients=10, # 최소 평가 참여 수
        min_available_clients=10, # 클라이언트 연결 필요 수
        eval_fn=get_eval_fn(model), # 모델 평가 결과
        on_fit_config_fn=fit_config, # batchsize, epoch 수
        on_evaluate_config_fn=evaluate_config, # val_step
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 8}, strategy=strategy)



def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    df, p_list = dataset.data_load()

    # Use the last 5k training examples as a validation set
    x_val, y_val = df.iloc[:5000,1:7], df.loc[:4999,'label']

    # y(label) one-hot encoding
    y_val = to_categorical(np.array(y_val))

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        # loss, accuracy, precision, recall, auc, prc = model.evaluate(x_val, y_val)
        # return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "prc": prc}
        loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
        
        # wandb에 log upload
        wandb.log({'loss':loss,"accuracy": accuracy})
        
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "auprc": auprc}

        # loss, accuracy, precision, recall, auc, f1_score, auprc = model.evaluate(x_val, y_val)
        # return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "f1_score": f1_score, "auprc": auprc}


    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    
    batch_size =32
    local_epochs = 10

    config = {
        "batch_size": batch_size,
        # "local_epochs": 1 if rnd < 2 else 2,
        "local_epochs": local_epochs,

    }

    # wandb log upload
    wandb.config.update({"local_epochs": local_epochs, "batch_size": batch_size},allow_val_change=True)

    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if rnd < 4 else 10
    val_steps = 5

    # wandb log upload
    wandb.config.update({"val_steps": val_steps},allow_val_change=True)
    
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()

    # wandb 종료
    wandb.finish()