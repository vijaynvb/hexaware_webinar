# Use the official lightweight Python image based on Alpine Linux
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8081

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8081", "--server.address", "0.0.0.0"]
