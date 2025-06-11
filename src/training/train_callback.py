import os
import re
from super_gradients.training.utils.callbacks import PhaseCallback, PhaseContext, Phase


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

class EpochProgressToContainer(PhaseCallback):
    """
    PhaseCallback that calls ContainerStatus.post_progress()
    at the end of every training epoch.
    """
    def __init__(self, cs, max_epochs):
        super().__init__(phase=Phase.TRAIN_EPOCH_END)
        self.container_status = cs
        self.max_epochs = max_epochs

    def __call__(self, context: PhaseContext):
        epoch_num = context.epoch + 1
        progress = round(epoch_num / self.max_epochs * 100, 2)
        payload = {
            "stage": f"Процесс обучения",
            "progress": progress,
            "train_msg": f"Эпоха {epoch_num}, {context.metrics if hasattr(context, 'metrics') else None}, {float(context.loss) if hasattr(context, 'loss') else None}"
            # "metrics": context.metrics if hasattr(context, "metrics") else None,
            # "loss": float(context.loss) if hasattr(context, "loss") else None,
        }
        if self.container_status is not None:
            self.container_status.post_progress(payload)