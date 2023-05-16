FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN dnf install -y libgl1-mesa-glx

COPY . .

EXPOSE 8080

CMD [ "python", "app.py" ]
