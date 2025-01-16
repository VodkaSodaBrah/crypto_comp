# Use Miniconda as the base image
FROM continuumio/miniconda3:4.9.2

# Set the working directory inside the container
WORKDIR /app

# Copy environment.yml to the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Set the default shell to use the Conda environment
SHELL ["conda", "run", "-n", "my_crypto_env", "/bin/bash", "-c"]

# Install PyTorch and related dependencies using pip
RUN conda run -n my_crypto_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Copy the entire project directory
COPY . .

# Make the run_all.sh script executable
RUN chmod +x scripts/run_all.sh

# Set the default command to execute the pipeline
CMD ["./scripts/run_all.sh"]