import os
import glob
import re
import matplotlib.pyplot as plt
import sys


def parse_log_file(filepath):
    print(f"Parsing: {filepath}")

    # Структуры данных
    ssl_history = {'epochs': [], 'loss': []}
    clf_folds = []

    current_fold = {'epochs': [], 'loss': [], 'f1': []}
    last_clf_epoch = -1

    regex_ssl = re.compile(r'SSL Ep\s+(\d+)\s+\|\s+Loss:\s+([\d\.]+)')

    regex_clf_loss = re.compile(r'(?<!SSL )Ep\s+(\d+).*?Loss:\s+([\d\.]+)')

    regex_clf_f1 = re.compile(r'Val F1:\s+([\d\.]+)')

    regex_epoch_generic = re.compile(r'(?<!SSL )Ep\s+(\d+)')

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f" Error reading {filepath}: {e}")
        return None, None

    for line in lines:
        match_ssl = regex_ssl.search(line)
        if match_ssl:
            ep = int(match_ssl.group(1))
            loss = float(match_ssl.group(2))
            ssl_history['epochs'].append(ep)
            ssl_history['loss'].append(loss)
            continue

        match_ep = regex_epoch_generic.search(line)
        if match_ep:
            ep = int(match_ep.group(1))

            if ep < last_clf_epoch:
                if len(current_fold['epochs']) > 0:
                    clf_folds.append(current_fold)
                current_fold = {'epochs': [], 'loss': [], 'f1': []}

            last_clf_epoch = ep

            if not current_fold['epochs'] or current_fold['epochs'][-1] != ep:
                current_fold['epochs'].append(ep)
                current_fold['loss'].append(None)
                current_fold['f1'].append(None)

            idx = current_fold['epochs'].index(ep)

            # Ищем Loss
            match_loss = regex_clf_loss.search(line)
            if match_loss:
                current_fold['loss'][idx] = float(match_loss.group(2))

            # Ищем F1
            match_f1 = regex_clf_f1.search(line)
            if match_f1:
                current_fold['f1'][idx] = float(match_f1.group(1))

    # Добавляем последний фолд
    if len(current_fold['epochs']) > 0:
        clf_folds.append(current_fold)

    return ssl_history, clf_folds


def plot_curves(logs_data):
    if not logs_data:
        print(" No valid data found to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.title("Training Loss (SSL & Classification)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    has_data = False
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for filename, (ssl, folds) in logs_data.items():
        # Plot SSL
        if ssl and len(ssl['epochs']) > 0:
            plt.plot(ssl['epochs'], ssl['loss'], label=f"{filename} (SSL)", linestyle='--', linewidth=2)
            has_data = True

        for i, fold in enumerate(folds):
            valid_pts = [(e, l) for e, l in zip(fold['epochs'], fold['loss']) if l is not None]
            if valid_pts:
                ep, val = zip(*valid_pts)
                label = f"{filename} (Fold {i + 1})" if len(folds) > 1 else f"{filename} (Clf)"
                plt.plot(ep, val, label=label, alpha=0.8)
                has_data = True

    if has_data:
        plt.legend()
        plt.savefig('loss_curve.png', dpi=150)
        print("Saved 'loss_curve.png'")
    else:
        print("No Loss data found to plot.")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.title("Validation F1-Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True, linestyle='--', alpha=0.6)

    has_data = False

    for filename, (_, folds) in logs_data.items():
        for i, fold in enumerate(folds):
            valid_pts = [(e, f) for e, f in zip(fold['epochs'], fold['f1']) if f is not None]
            if valid_pts:
                ep, val = zip(*valid_pts)
                label = f"{filename} (Fold {i + 1})" if len(folds) > 1 else f"{filename}"
                plt.plot(ep, val, marker='.', label=label)
                # Аннотация максимума
                max_f1 = max(val)
                max_ep = ep[val.index(max_f1)]
                plt.annotate(f'{max_f1:.4f}', xy=(max_ep, max_f1), xytext=(max_ep, max_f1 + 0.005),
                             arrowprops=dict(facecolor='black', arrowstyle='-'), fontsize=8)
                has_data = True

    if has_data:
        plt.legend(loc='lower right')
        plt.savefig('f1_curve.png', dpi=150)
        print("Saved 'f1_curve.png'")
    else:
        print("No F1 data found to plot.")
    plt.close()


def main():
    # Ищем все txt файлы, похожие на логи
    log_files = glob.glob("*log*.txt")
    # Добавляем стандартные имена, если их нет в glob
    defaults = ['training_log.txt', 'godzilla_log.txt']
    for d in defaults:
        if os.path.exists(d) and d not in log_files:
            log_files.append(d)

    if not log_files:
        print(" No log files found in current directory!")
        return

    print(f"Found logs: {log_files}")

    all_data = {}
    for log_file in log_files:
        ssl, folds = parse_log_file(log_file)
        if ssl or folds:
            all_data[log_file] = (ssl, folds)

    plot_curves(all_data)


if __name__ == "__main__":
    main()