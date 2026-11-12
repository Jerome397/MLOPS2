FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit==1.67.0 requests==2.34.2
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
