# Use a lightweight Python image.
FROM python:3.12-slim-bookworm

# Set the working folder inside the container.
WORKDIR /app

# Copy dependencies first.
COPY requirements.txt .

# Install Python packages.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files.
COPY . .

# Document the FastAPI port.
EXPOSE 8000

# Start the FastAPI server.
CMD ["python", "-m", "uvicorn", "app.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]