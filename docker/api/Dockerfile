# Start from a light-weight Python base
FROM python:3.11-slim

# Prevent Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Create a working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install your dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app/

# EXPOSE command is optional for documentation. 

EXPOSE 8000
# Alternatively, if you want to run FastAPI by default with uvicorn, 
# comment out the line above and uncomment the line below:
CMD ["python", "main.py"]
