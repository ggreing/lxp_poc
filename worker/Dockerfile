# Use a specific Python version for reproducibility
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utility modules
COPY __init__.py config.py db.py minio_utils.py rabbitmq.py rag.py vector_utils.py /app/

# Copy worker-specific modules
COPY AI_Assist/ /app/AI_Assist/
COPY AI_Simulation_Training/ /app/AI_Simulation_Training/
COPY Galaxy_Coach/ /app/Galaxy_Coach/

# Set the default command to run the Galaxy_Coach worker
# This can be overridden at runtime to run other workers
# e.g., docker run <image> python -m AI_Assist.main
# The module path is relative to the WORKDIR /app
CMD ["python", "-m", "Galaxy_Coach.main"]
