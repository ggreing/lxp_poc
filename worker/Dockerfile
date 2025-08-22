# Use a specific Python version for reproducibility
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project context
# This makes shared files like course_data.csv available
COPY . .

# Set the default command to run the Galaxy_Coach worker
# This can be overridden at runtime to run other workers
# e.g., docker run <image> python -m worker.AI_Assist.main
CMD ["python", "-m", "worker.Galaxy_Coach.main"]
