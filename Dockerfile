# Dockerfile

# Используем официальный образ Python как базовый
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл requirements.txt в рабочую директорию
COPY requirements.txt .

# Устанавливаем все зависимости из requirements.txt
# Устанавливаем необходимые системные зависимости для scikit-learn и pandas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libatlas-base-dev \
        && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем весь остальной код проекта в рабочую директорию контейнера
COPY . .

# Указываем команду, которая будет выполняться по умолчанию при запуске контейнера
# Мы будем переопределять ее, когда будем "входить" в контейнер
CMD ["bash"]