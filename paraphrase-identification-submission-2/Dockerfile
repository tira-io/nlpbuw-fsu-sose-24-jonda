FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip install transformers pandas scikit-learn torch tira datasets gdown

RUN gdown --folder https://drive.google.com/drive/folders/1uj5rsxUj2gJiYrikHb-IM1FXLMEsuIzy?usp=sharing --output /model

ADD run.py /code/run.py
ADD train.py /code/train.py

ENTRYPOINT [ "/code/run.py" ]
