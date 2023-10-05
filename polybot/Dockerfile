# Use the official Python image as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and copy the requirements file to the container
RUN pip install --upgrade pip

# Install the required Python packages from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port (if needed)
EXPOSE 8443

# Specify the command to run your application
CMD ["python3", "app.py"]
