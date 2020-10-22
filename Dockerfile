FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /var/model_app
COPY . .
RUN pip install -r requirements.txt
RUN python setup.py
EXPOSE 5000
CMD python web_app.py