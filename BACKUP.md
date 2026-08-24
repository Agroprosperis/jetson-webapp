# Резервне копіювання Tilletia

Документ описує рекомендовану схему бекапу для розгортання Tilletia на одному хості
(Ubuntu 24.04 + NVIDIA, Docker Compose під systemd).

Базове припущення цього документа: **відеофайли не є критичними для бекапу**. Вони становлять
основний обсяг даних, тому їх виключення робить бекап компактним і дозволяє тримати
офсайт-копію навіть на слабкому каналі.

---

## 1. Де живуть дані

Уся персистентна інформація зберігається на хості поза контейнером — у двох коренях:

| Хост-шлях | Шлях у контейнері | Вміст |
|---|---|---|
| `/var/lib/tilletia-app/runs` | `/app/runs` | БД SQLite, секрет підпису токенів, fallback-файли налаштувань |
| `/var/lib/tilletia-app/output_hq` | `/app/output_hq` | Результати аналізів, по одній теці на `run_id` |
| `/var/lib/tilletia-app/model` | `/app/model` | Ваги моделей: `ul/` (Ultralytics), `rf/` (Roboflow RF-DETR) |
| `/var/lib/tilletia-app/credentials` | — | TPM-зашифрований мастер-ключ (`roboflow-master-key.cred`) |
| `/etc/tilletia-app` | частково монтується | `mediamtx.yml`, `localtime`, `timezone` |

Змінні `TILLETIA_DATA_ROOT`, `TILLETIA_CONFIG_ROOT` можуть перевизначати ці шляхи
(див. `deploy/docker-compose.yml`, `deploy/start.sh`). У standalone-режимі (`./start.sh` без
встановлення сервісу) корінь даних — `${REPO_DIR}/data`.

`/run/tilletia/secrets` — це ramfs із розшифрованим ключем. **Бекапити не потрібно і не можна**:
вміст відтворюється на кожному старті сервісу.

`/upload` у контейнері (завантажені початкові відео) **не змонтований на хост** — ці файли вже
є тимчасовими і зникають при перезапуску контейнера. Бекапити нічого.

---

## 2. Пріоритети: що бекапимо, а що ні

### Критично — втрата означає втрату роботи

1. **`/var/lib/tilletia-app/runs/tilletia.sqlite3`** — єдина база даних застосунку.
   Таблиці: `users`, `user_roles`, `refresh_tokens`, `results` (власники результатів),
   `model_owners`, `dashboard_settings`, `roboflow_settings`, `smtp_settings`.
2. **`/var/lib/tilletia-app/output_hq/<run_id>/`, без відео** — власне результати аналізів:
   - `<run_id>.csv` — підрахунки за класами та S-значення;
   - `manual_counts.json` — ручні підрахунки оператора;
   - `*.jpg`, `*.jpeg`, `*.png` — доказові кадри та снапшоти.
3. **`/var/lib/tilletia-app/runs/roboflow.json`, `smtp.json`** — існують лише якщо адміністратор
   свідомо обрав plaintext-fallback замість TPM. Містять **секрети у відкритому вигляді** —
   отже, бекап обов'язково має бути зашифрованим.
4. **`/var/lib/tilletia-app/credentials/roboflow-master-key.cred`** — без нього зашифровані
   токени в SQLite неможливо розшифрувати. Див. розділ 6 про обмеження TPM.

### Бажано — відновлюване, але з витратами часу

5. **`/var/lib/tilletia-app/model/ul/*.pt`, `model/rf/*`** — ваги моделей. Технічно їх можна
   заново завантажити з Roboflow, але це вимагає доступу до акаунту й точного дотримання
   конвенції імен (суфікс `-fp16` обов'язковий, інакше застосунок не знайде модель).
6. **`/etc/tilletia-app/mediamtx.yml`** — відтворюється з `config/mediamtx.yml` у репозиторії,
   але якщо його правили руками, копія потрібна.
7. **`/var/lib/tilletia-app/runs/tilletia.token`** — секрет підпису токенів. Якщо втратити,
   він згенерується автоматично, ціна — розлогінення всіх користувачів.

### Не бекапимо

8. **Відео**: `<run_id>.mkv`, `<run_id>-raw.mkv`, а також `*.mp4`, `*.avi`, `*.mov`.
9. **Скомпільовані рушії `*.engine`** — привʼязані до конкретного GPU й версії TensorRT.
   Після відновлення їх однаково потрібно перекомпілювати з `.pt`, тож у бекапі вони — марна вага.
10. **Systemd-юніти, Docker-образ, код** — відтворюються з git-репозиторію
    (`deploy/install-autostart.sh` перевстановлює юніти й перебудовує образ).

### Свідомий компроміс через виключення відео

Після відновлення сторінка результатів покаже CSV, підрахунки й доказові зображення, але
**відтворення та завантаження відео для відновлених прогонів буде недоступним**. Якщо для
окремих еталонних прогонів відео все ж потрібне — виносьте їх вручну в окрему довготривалу
копію, а не змінюйте загальне правило.

---

## 3. Консистентність SQLite — головний технічний нюанс

База працює у стандартному режимі журналювання (не WAL). Звичайне `cp` живого файлу під
навантаженням може дати **пошкоджену копію**. Потрібно використовувати онлайн-бекап SQLite,
який коректно взаємодіє з паралельними записами:

```bash
sudo apt install -y sqlite3
sqlite3 /var/lib/tilletia-app/runs/tilletia.sqlite3 \
  ".backup '/var/lib/tilletia-app/runs/tilletia-backup.sqlite3'"
```

Далі у бекап потрапляє саме `tilletia-backup.sqlite3`, а живий файл і його журнали
(`-journal`, `-wal`, `-shm`) виключаються.

Якщо ставити `sqlite3` на хост не хочеться, той самий результат дає Python у контейнері:

```bash
docker exec tilletia-app python3 -c "import sqlite3; s=sqlite3.connect('/app/runs/tilletia.sqlite3'); d=sqlite3.connect('/app/runs/tilletia-backup.sqlite3'); s.backup(d); d.close(); s.close()"
```

### Узгодженість БД і файлів результатів

Таблиця `results` зіставляє `run_id` із власником, а самі дані лежать у теці `output_hq/<run_id>/`.
Якщо ці дві частини зняти в різні моменти, можливий розсинхрон. Практичні правила:

- Запускайте бекап у вікні, коли аналіз не виконується (типовий прогін обмежений
  `max_run_minutes`, за замовчуванням 30 хвилин, — нічне вікно безпечне).
- **Спочатку знімайте снапшот БД, потім файли.** Тоді прогін, створений під час бекапу,
  у гіршому разі дасть теку без запису про власника (адміністратор це побачить і виправить).
  Зворотний порядок дав би запис у БД без файлів — це виглядає як зламаний результат.

---

## 4. Рекомендований інструмент: restic

Профіль даних тут — один невеликий БД-файл плюс велика кількість дрібних незмінних файлів.
Для такого набору `restic` підходить найкраще:

- шифрування репозиторію за замовчуванням (обов'язково, бо в бекапі можуть бути plaintext-токени);
- дедуплікація — повторні бекапи майже незмінного `output_hq` майже не займають місця;
- прості `--exclude` правила для відео;
- вбудована ротація (`forget --prune`) і перевірка цілісності (`check`);
- підтримка локального диска, SFTP, S3/B2 одним інструментом.

`borgbackup` — рівноцінна альтернатива, якщо він уже використовується у вашій інфраструктурі.

### Ініціалізація

```bash
sudo apt install -y restic

# Пароль репозиторію. Зберігається локально для автоматизації,
# АЛЕ обов'язково має бути продубльований у зовнішньому менеджері паролів.
sudo install -d -m 0700 /etc/tilletia-app
openssl rand -base64 32 | sudo tee /etc/tilletia-app/restic-password > /dev/null
sudo chmod 0600 /etc/tilletia-app/restic-password

# Приклад для локального/USB/NAS-призначення
echo '/mnt/backup/tilletia' | sudo tee /etc/tilletia-app/restic-repo > /dev/null
sudo chmod 0600 /etc/tilletia-app/restic-repo

sudo restic --repository-file /etc/tilletia-app/restic-repo \
  --password-file /etc/tilletia-app/restic-password init
```

### Файл виключень

`/etc/tilletia-app/backup-exclude.txt`:

```
# Відео — не критичне для відновлення
*.mkv
*.mp4
*.avi
*.mov

# TensorRT-рушії прив'язані до GPU та версії TensorRT — перекомпілювати після відновлення
*.engine

# Живий файл БД: у бекап іде консистентний снапшот tilletia-backup.sqlite3
/var/lib/tilletia-app/runs/tilletia.sqlite3
/var/lib/tilletia-app/runs/tilletia.sqlite3-journal
/var/lib/tilletia-app/runs/tilletia.sqlite3-wal
/var/lib/tilletia-app/runs/tilletia.sqlite3-shm

# Пароль репозиторію всередині самого репозиторію користі не має
/etc/tilletia-app/restic-password
```

### Скрипт бекапу

`/usr/local/sbin/tilletia-backup.sh` (режим `0700`, власник `root`):

```bash
#!/usr/bin/env bash
set -euo pipefail

export RESTIC_REPOSITORY_FILE=/etc/tilletia-app/restic-repo
export RESTIC_PASSWORD_FILE=/etc/tilletia-app/restic-password

DATA_ROOT="${TILLETIA_DATA_ROOT:-/var/lib/tilletia-app}"
CONFIG_ROOT="${TILLETIA_CONFIG_ROOT:-/etc/tilletia-app}"
DB_LIVE="$DATA_ROOT/runs/tilletia.sqlite3"
DB_SNAPSHOT="$DATA_ROOT/runs/tilletia-backup.sqlite3"

# 1. Консистентний снапшот БД без зупинки сервісу
if [[ -f "$DB_LIVE" ]]; then
  rm -f "$DB_SNAPSHOT"
  sqlite3 "$DB_LIVE" ".backup '$DB_SNAPSHOT'"
  sqlite3 "$DB_SNAPSHOT" 'PRAGMA integrity_check;' | grep -qx ok
  chmod 0600 "$DB_SNAPSHOT"
fi

# 2. Файли (відео та рушії відсікаються файлом виключень)
restic backup \
  --exclude-file "$CONFIG_ROOT/backup-exclude.txt" \
  --tag tilletia \
  "$DATA_ROOT/runs" \
  "$DATA_ROOT/output_hq" \
  "$DATA_ROOT/model" \
  "$DATA_ROOT/credentials" \
  "$CONFIG_ROOT"

# 3. Ротація
restic forget --tag tilletia \
  --keep-daily 7 --keep-weekly 4 --keep-monthly 6 \
  --prune
```

### Розклад (systemd timer)

`/etc/systemd/system/tilletia-backup.service`:

```ini
[Unit]
Description=Tilletia backup
After=tilletia-app.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/tilletia-backup.sh
```

`/etc/systemd/system/tilletia-backup.timer`:

```ini
[Unit]
Description=Nightly Tilletia backup

[Timer]
OnCalendar=*-*-* 02:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tilletia-backup.timer
sudo systemctl list-timers tilletia-backup.timer
```

`Persistent=true` важливий: якщо пристрій був вимкнений у час запуску, бекап виконається
після наступного завантаження.

---

## 5. Розклад, зберігання і ретенція

- **Частота:** щодня вночі. Дані додаються прогонами, тому добова гранулярність достатня.
  Якщо аналізи виконуються інтенсивно — двічі на добу.
- **Ретенція:** 7 щоденних + 4 щотижневих + 6 щомісячних. Завдяки дедуплікації та відсутності
  відео це недорого.
- **Правило 3-2-1:** мінімум дві копії на різних носіях і одна поза локацією.
  Реалістична конфігурація для edge-пристрою:
  1. локальний зовнішній диск або NAS (швидке відновлення);
  2. офсайт — S3/Backblaze B2/SFTP. Саме виключення відео робить цей пункт практичним.

Для другого призначення додайте у скрипт ще один виклик `restic backup` з іншим
`RESTIC_REPOSITORY_FILE` або скористайтеся `restic copy` між репозиторіями.

Перед вибором обсягу сховища виміряйте реальні дані на пристрої:

```bash
sudo du -sh /var/lib/tilletia-app/*
sudo du -sh --exclude='*.mkv' --exclude='*.mp4' /var/lib/tilletia-app/output_hq
```

Якщо різниця між цими числами велика — виключення відео дає основну економію. Якщо доказових
зображень надто багато, окремо перевірте, чи потрібні в бекапі відлагоджувальні кадри
(`tracker_*.jpg`) — їх можна додати до виключень.

---

## 6. Секрети та обмеження TPM

Це найважливіший нюанс усього плану відновлення.

Мастер-ключ `/var/lib/tilletia-app/credentials/roboflow-master-key.cred` зашифрований
через `systemd-creds --with-key=tpm2` **без PCR-політики**. Наслідки:

- звичайні перезапуски сервісу, оновлення ОС і прошивки, а також перевстановлення застосунку
  ключ **не ламають**;
- відновлення на **тому самому хості** (наприклад, після відновлення диска) — працює;
- перенесення на **інше залізо**, очищення або заміна TPM робить `.cred`
  **непридатним назавжди**. Зашифровані значення в `roboflow_settings` і `smtp_settings`
  розшифрувати буде неможливо.

Тому бекап `.cred` є необхідним, але **недостатнім**. Обов'язково додатково зберігайте
у зовнішньому менеджері паролів:

- Roboflow API token і назву проєкту;
- SMTP host / port / username / password / sender;
- пароль restic-репозиторію;
- пароль адміністратора застосунку.

Без пароля restic бекап неможливо відновити взагалі — тримати його виключно на пристрої,
який ви бекапите, означає не мати бекапу.

---

## 7. Процедура відновлення

### 7.1 Той самий хост (відновлення після збою диска / ОС)

```bash
# 1. Зупинити стек
sudo systemctl stop tilletia-app.service

# 2. Відновити дані (від root — щоб зберегти власників і права)
sudo restic --repository-file /etc/tilletia-app/restic-repo \
  --password-file /etc/tilletia-app/restic-password \
  restore latest --target /

# 3. Ввести в дію снапшот БД
sudo mv /var/lib/tilletia-app/runs/tilletia-backup.sqlite3 \
        /var/lib/tilletia-app/runs/tilletia.sqlite3
sudo chmod 0600 /var/lib/tilletia-app/runs/tilletia.sqlite3
sudo sqlite3 /var/lib/tilletia-app/runs/tilletia.sqlite3 'PRAGMA integrity_check;'

# 4. Перевірити права на секрети
sudo chmod 0700 /var/lib/tilletia-app/credentials
sudo chmod 0600 /var/lib/tilletia-app/credentials/roboflow-master-key.cred

# 5. Перевстановити сервіс і перебудувати образ із git-репозиторію
cd ${REPO_DIR}/deploy
sudo ./install-autostart.sh
```

Після старту: перекомпілювати TensorRT-рушії з `.pt` через `Model Catalog`
(рушії в бекап не входять), потім вибрати модель на дашборді.

### 7.2 Нове залізо

Кроки 1–5 ті самі, але додатково:

- ключ TPM не перенесеться — на сторінці Settings **заново введіть Roboflow API token
  і пароль SMTP** зі свого менеджера паролів;
- переконайтеся, що новий пристрій має TPM 2.0, інакше застосунок запропонує лише
  пам'ять або plaintext-fallback;
- перевірте назви відеопристроїв (`/dev/video*`) — налаштування камери в
  `dashboard_settings` можуть посилатися на інший індекс.

### 7.3 Часткове відновлення одного результату

```bash
sudo restic --repository-file /etc/tilletia-app/restic-repo \
  --password-file /etc/tilletia-app/restic-password \
  restore latest --target /tmp/restore \
  --include '/var/lib/tilletia-app/output_hq/<run_id>'
```

---

## 8. Перевірка бекапів

Бекап, який ніколи не відновлювали, бекапом не є.

```bash
# Щотижнево: цілісність метаданих репозиторію
sudo restic --repository-file /etc/tilletia-app/restic-repo \
  --password-file /etc/tilletia-app/restic-password check

# Щомісячно: перевірка частини реальних даних
sudo restic ... check --read-data-subset=10%

# Щомісячно: тестове відновлення в тимчасову теку + перевірка БД
sudo restic ... restore latest --target /tmp/tilletia-restore-test
sqlite3 /tmp/tilletia-restore-test/var/lib/tilletia-app/runs/tilletia-backup.sqlite3 \
  'PRAGMA integrity_check; SELECT count(*) FROM users; SELECT count(*) FROM results;'
```

Раз на квартал варто виконати повний прогін відновлення на тестовій машині — це єдиний
спосіб пересвідчитися, що документ вище справді працює.

### Контроль виконання

```bash
sudo systemctl list-timers tilletia-backup.timer
sudo journalctl -u tilletia-backup.service -n 50
```

Для оповіщень про збої додайте у `tilletia-backup.service` директиву
`OnFailure=` з юнітом, який надсилає лист, або періодично перевіряйте
`restic snapshots --tag tilletia --latest 1` на свіжість дати.

---

## 9. Код і конфігурація розгортання

Код застосунку живе в git (`Agroprosperis/jetson-webapp`). Це вже друга копія, але
GitHub тут — єдина копія історії. Якщо репозиторій критичний для бізнесу, тримайте
дзеркало:

```bash
git clone --mirror https://github.com/Agroprosperis/jetson-webapp.git
```

Разом із кодом відновлюються всі systemd-юніти, `docker-compose.yml`, Ansible-плейбуки
та `config/mediamtx.yml` — тому окремо бекапити їх не потрібно, достатньо зафіксувати
коміт, на якому працює продакшн.

---

## 10. Чекліст впровадження

- [ ] Встановлено `restic` і `sqlite3` на хості
- [ ] Створено репозиторій restic на окремому носії
- [ ] Пароль restic-репозиторію збережено **поза** пристроєм
- [ ] Створено `/etc/tilletia-app/backup-exclude.txt` із виключенням відео
- [ ] Створено `/usr/local/sbin/tilletia-backup.sh` (`0700`, root)
- [ ] Увімкнено `tilletia-backup.timer`, перевірено перший успішний прогін
- [ ] Налаштовано друге (офсайт) призначення
- [ ] Roboflow token, SMTP-креденшели й пароль адміністратора внесено в менеджер паролів
- [ ] Виконано тестове відновлення та `PRAGMA integrity_check`
- [ ] Зафіксовано продакшн-коміт застосунку
