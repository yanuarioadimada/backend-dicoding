FROM node:18

# Set working directory
WORKDIR /app

# Copy seluruh kode aplikasi ke dalam container
COPY . .

# Install dependencies
RUN npm install

# Jalankan aplikasi
CMD ["node", "index.js"]
