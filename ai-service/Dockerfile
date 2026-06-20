# Stage 1: Build Rust MRI reconstruction binary
FROM rust:1.80-slim as rust-builder
WORKDIR /build

# Copy Cargo manifests for dependency caching
COPY rust-mri/Cargo.toml rust-mri/Cargo.lock ./
# Create a dummy source file to compile and cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

# Copy the actual Rust source code and rebuild
COPY rust-mri/src ./src
RUN touch src/main.rs && cargo build --release

# Stage 2: Final Python environment
FROM python:3.11-slim

WORKDIR /app

# System libraries required by SimpleITK and OpenCV-style deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy python requirements first to leverage caching
COPY ai-service/requirements.txt .

# CPU-only PyTorch keeps the image smaller and builds faster in Docker
RUN pip install --no-cache-dir --default-timeout=1000 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    && pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy all python source files
COPY ai-service/ .

# Copy the compiled Rust binary from the builder stage
COPY --from=rust-builder /build/target/release/rust-mri /usr/local/bin/rust-mri-bin

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]