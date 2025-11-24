# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Jupyter runs on
EXPOSE 8888

# Command to run Jupyter Notebook
# --ip=0.0.0.0 allows external access (required for Docker)
# --allow-root allows running as root inside the container
# --no-browser prevents trying to open a browser in the headless container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]