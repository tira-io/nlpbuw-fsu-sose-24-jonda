FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip install transformers pandas scikit-learn torch tira datasets

RUN wget https://drive.google.com/drive/folders/1uj5rsxUj2gJiYrikHb-IM1FXLMEsuIzy?usp=drive_link -o /code/model

ADD run.py /code/run.py
ADD train.py /code/train.py

ENTRYPOINT [ "/code/run.py" ]
