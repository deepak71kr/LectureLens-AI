# 1. Use a lightweight Python environment
FROM python:3.11-slim

# 2. Set up a non-root user (Hugging Face security requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. Set the working directory
WORKDIR /app

# 4. Copy requirements and install them
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy the rest of your application code
COPY --chown=user . /app

# 6. Run the Streamlit app on port 7860 (Hugging Face default)
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]