import os
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from datetime import datetime

from catboost import CatBoostClassifier
from sqlalchemy import create_engine

from schema import PostGet, Response
import hashlib

salt = 'my_salt'
percent = 50

app = FastAPI()


def get_model_path(path: str) -> str:
    MODEL_PATH = path
    return MODEL_PATH


def load_models(mod: str):
    model_path = get_model_path(mod)
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://somedata"
        "postgres.somedata"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


# загрузка фичей, которые будут использованы при построении таблицы для предсказаний модели
def load_features():
    table_name = "lyudmila_anoshkina_features_lesson_22"
    query = f"SELECT * FROM {table_name}"
    return batch_load_sql(query)


# загрузка таблицы с информацией о постах
def load_post_text():
    table_name = "post_text_df"
    query = f"SELECT * FROM {table_name}"
    return batch_load_sql(query)


# проверка на существование юзера в таблице фичей
def find_user(user_id: int):
    if user_id in features['user_id'].values:
        return 1
    else:
        return 0


# по user_id пользователя разделение на группу (тестовя или контрольная)
def get_exp_group(user_id: int):
    value_str = str(user_id) + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    if value_num % 100 > percent:
        return 'test'
    else:
        return 'control'


# генерация таблицы для предсказания модели
def get_pred_table(group: str, user_id: int, day_of_week: int, hour: int, month=0):
    col_user = ['user_id', 'gender', 'age', 'country', 'city', 'exp_group', 'os', 'source']  # фичи для юзера

    # фичи для постов
    if group == 'control':
        col_post = list(features.columns.drop(col_user + ['emb_min', 'emb_mean']))
    elif group == 'test':
        col_post = list(features.columns.drop(col_user))

    user = features[features['user_id'] == user_id][col_user]
    pred_table = features[col_post].dropna()
    pred_table = pd.concat((user.reset_index(), pred_table.reset_index()), axis=1).drop('index', axis=1)
    for col in col_user:
        pred_table[col] = pred_table[col].fillna(user[col].values[0])

    pred_table[['user_id', 'gender', 'age', 'exp_group']] = pred_table[
        ['user_id', 'gender', 'age', 'exp_group']].astype('int')
    pred_table['day_of_week'] = day_of_week
    pred_table['hour'] = hour

    if group == 'test':
        pred_table['month'] = month

    return pred_table


# построение рекомендаций
def get_posts(table: pd.DataFrame, model):
    predict = model.predict_proba(table.drop(['post_id', 'user_id'], axis=1))[:,
              1]  # выполнение предсказаний (прогнозирование вероятности), выбор вероятности принадл. к классу 1 (т.е. вероятность того, что пользователь поставит лайк)
    p = pd.concat((table.reset_index().drop(['index'], axis=1), pd.DataFrame(predict, columns=['predict'])), axis=1)
    post_ids = p.sort_values('predict', ascending=False).head(5)['post_id']  # сортировка по вероятности

    posts = []
    for p in post_ids:
        post = post_text[post_text['post_id'] == int(p)]
        posts.append(
            {'id': int(p),
             'text': post['text'].values[0],
             'topic': post['topic'].values[0]}
        )
    return posts


# загружаем модели
model_control = load_models('model_control')
model_test = load_models('model_test')

# загружаем необходимые данные
features = load_features()
post_text = load_post_text()


@app.get("/post/recommendations/", response_model=Response)
def getRecom(id: int, time: datetime = 0, limit: int = 10) -> Response:
    start_time = datetime.now()

    if time != 0:
        hour = time.hour
        day_of_week = time.weekday()
        month_ = time.month
    else:
        hour = 0
        day_of_week = 0
        month_ = 0

    if find_user(id) == 0:
        raise HTTPException(404, "user not found")
    else:
        exp_group = get_exp_group(id)

        if exp_group == 'control':
            features_table = get_pred_table('control', id, day_of_week, hour)
            posts = get_posts(features_table, model_control)
        elif exp_group == 'test':
            features_table = get_pred_table('test', id, day_of_week, hour, month_)
            posts = get_posts(features_table, model_test)
        else:
            raise ValueError('unknown group')

        resp = {
            'exp_group': exp_group,
            'recommendations': posts
        }
        print(datetime.now() - start_time)
        return resp
