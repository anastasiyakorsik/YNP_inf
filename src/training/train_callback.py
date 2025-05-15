import os
import re

def extract_latest_experiment_logs(log_dir='.'):
    # Находим все файлы, начинающиеся на "experiment_logs"
    log_files = [f for f in os.listdir(log_dir) if f.startswith("experiment_logs") and f.endswith(".txt")]
    if not log_files:
        raise FileNotFoundError("Не найдено файлов, начинающихся на 'experiment_logs'")

    # Сортируем по времени изменения (или алфавиту — если нужно)
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    latest_file = os.path.join(log_dir, log_files[0])
    print(f"Читаем файл: {latest_file}")

    # Извлекаем логи по эпохам
    return extract_epoch_logs(latest_file)

def extract_epoch_logs(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    epoch_logs = {}
    for line in lines:
        # Ищем "Epoch N (...)" в начале строки
        match = re.match(r"(Epoch\s+(\d+)\s+\(\d+/\d+\))", line)
        if match:
            epoch_num = int(match.group(2))
            if epoch_num not in epoch_logs:
                epoch_logs[epoch_num] = []
            epoch_logs[epoch_num].append(line.strip())

    # Соединяем список строк в один текст для каждой эпохи
    for epoch in epoch_logs:
        epoch_logs[epoch] = "\n".join(epoch_logs[epoch])

    return epoch_logs


