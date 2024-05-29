FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip install transformers pandas scikit-learn torch tira datasets

ADD run.py /code/run.py
ADD train.py /code/train.py
ADD entrypoint.sh /code/entrypoint.sh

ENTRYPOINT [ "/code/entrypoint.sh" ]