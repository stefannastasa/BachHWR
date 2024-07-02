# Use an official Python runtime as a parent image
FROM python:3.9.19-slim-bullseye
# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./
ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libc-dev \
	pkg-config \
        libhdf5-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .
# Make port 27018 available to the world outside this container
EXPOSE 27018

# Define environment variable
ENV NAME FastAPIApp

# Run app.py when the container launches
CMD ["python3", "main.py"]
