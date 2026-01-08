import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from starlette.middleware.cors import CORSMiddleware


class SearchAlgorithm:
    LINEAR = "linear"
    BINARY = "binary"


class DictionaryElement(BaseModel):
    key: str
    definition: str
    index: int


class SearchRun(BaseModel):
    result: DictionaryElement | None
    history: deque[DictionaryElement]
    duration_ns: int
    comparisons: int


DICTIONARY_CSV_PATH: str = "./data/dict.csv"
dictionary_elements: list[DictionaryElement] = list()


@asynccontextmanager
async def lifespan(app: FastAPI):
    dict_df: pd.DataFrame = pd.read_csv(DICTIONARY_CSV_PATH, skip_blank_lines=True)
    dict_df = dict_df.dropna(axis=0)
    for index, row in dict_df.iterrows():
        dictionary_elements.append(DictionaryElement(
            key=row["word"],
            definition=row["definition"],
            index=index
        ))
    dictionary_elements.sort(key=lambda el: el.key, reverse=False)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def linear_dictionary_search(dictionary: list[DictionaryElement], key: str) -> SearchRun:
    t0: float = time.time()
    comparisons: int = 0
    history: deque[DictionaryElement] = deque([], maxlen=10)
    for el in dictionary:
        history.append(el)
        comparisons += 1
        if el.key == key:
            return SearchRun(result=el, history=history, duration_ns=int((time.time() - t0) * 1000 * 1000),
                             comparisons=comparisons)
    return SearchRun(result=None, history=history, duration_ns=int((time.time() - t0) * 1000 * 1000),
                     comparisons=comparisons)


def binary_dictionary_search(dictionary: list[DictionaryElement], key: str, history_maxlen: int = 10) -> SearchRun:
    t0: float = time.time()
    history: deque[DictionaryElement] = deque([], maxlen=history_maxlen)
    comparisons: int = 0
    i = 0
    j = len(dictionary) - 1
    while i <= j:
        # total de j - i + 1 elemente in [i, j]
        # daca este numar impar => vreau half_index sa fie i + ((j - i + 2) / 2 - 1) = (i + j)/2 = (i + j) // 2
        # daca este numar par => vreau half_index sa fie i + ((j - i + 1) / 2 - 1) = (i + j - 1) / 2 = (i + j) // 2
        half_index: int = (i + j) // 2
        history.append(dictionary[half_index])
        comparisons += 1
        if key < dictionary[half_index].key:
            j = half_index - 1
        elif key > dictionary[half_index].key:
            i = half_index + 1
        else:
            return SearchRun(result=dictionary[half_index], history=history,
                             duration_ns=int((time.time() - t0) * 1000 * 1000),
                             comparisons=comparisons)
    return SearchRun(result=None, history=history,
                     duration_ns=int((time.time() - t0) * 1000 * 1000),
                     comparisons=comparisons)


@app.get("/definition")
def get_definition(
        word: str,
        algorithm: str
):
    if algorithm == SearchAlgorithm.BINARY:
        return binary_dictionary_search(dictionary=dictionary_elements, key=word, history_maxlen=32)
    elif algorithm == SearchAlgorithm.LINEAR:
        return linear_dictionary_search(dictionary=dictionary_elements, key=word)
    else:
        raise HTTPException(status_code=422, detail=f"Unknown algorithm '{algorithm}'")
