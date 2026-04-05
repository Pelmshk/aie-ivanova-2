# HW13 – токенизация текста, инференс BERT и базовый fine-tuning для классификации

## 1. Кратко: что сделано

- Использовался датасет ag_news, поскольку он не слишком большой для обычного учебного запуска, содержит понятные классы, подходит для задачи классификации текста и не требует тяжёлой предварительной очистки и долгой подготовки.
- Для sequence classification использовалась модель rubert-tiny2
- В части токенизации показано полное преобразование текста AG News в формат BERT-base-uncased. Каждый пример содержит ключи label, input_ids, token_type_ids и attention_mask. Input_ids представляют числовые идентификаторы токенов, начинающиеся с [CLS] и заканчивающиеся [SEP]. Attention_mask отмечает активные токены единицами. Labels батча соответствуют истинным меткам классов. Декодирование input_ids полностью восстанавливает исходный текст, демонстрируя корректную работу subword-токенизатора. Тексты новостей укладываются в 40-50 токенов, что идеально для sequence classification. Готово для fine-tuning модели.
- Инференс предобученной BERT-base-uncased на AG News показал низкую точность около 57 процентов со слабой уверенностью предсказаний, что демонстрирует случайный характер предсказаний без fine-tuning, близкий к базовому уровню 25 процентов для 4 классов. Готовая модель не подходит для задачи классификации новостей и требует дообучения.
- Fine-tuning на AG News дал высокую точность на test-разметке.

## 2. Среда и воспроизводимость

- Python: 3.14.0
- datasets: 4.8.4
- transformers: 5.4.0
- torch: 2.10.0+cpu
- Устройство (CPU/GPU): CPU
- Seed: 19
- Как запустить: открыть `HW13.ipynb` и выполнить Run All.

## 3. Данные и постановка задачи

- Датасет: ag_news
- Число классов: 4
- Размер `train`: 90000
- Размер `validation`: 30000
- Размер `test`: 7600
- Что именно классифицируется: классификация новостей по тематике (World, Sports, Business, Sci/Tech)
- Комментарий (2-4 предложения): Датасет содержит короткие новости (30-50 токенов), идеальные для BERT без усечения. Четыре сбалансированных класса обеспечивают стабильное обучение без переобучения на доминирующем классе. Задача средней сложности: тематика четко разделима, но требует понимания контекста. Баланс классов и достаточный объем train позволяют достичь 90%+ accuracy после fine-tuning.

## 4. Токенизация и готовый инференс

### 4.1. Токенизация

- Какая модель токенизатора использовалась: bert-base-multilingual-cased
- Какие special tokens используются: [CLS] (начало последовательности, ID=101), [SEP] (конец предложения, ID=102), [PAD] (заполнение, ID=0)
- Показан ли пример `padding` / `truncation`: Да, показан пример `padding`
- Что показал разбор токенизации на нескольких примерах:Текст преобразуется в input_ids (числа), attention_mask (1/0 для токенов), token_type_ids (0 для single sentence). Декодирование восстанавливает оригинал. Новости AG News ~40 токенов, subword разбиение работает корректно ("Linux software" → отдельные ID). Готово для sequence classification.

### 4.2. Инференс готовой pretrained модели

- Какая готовая модель использовалась: bert-base-multilingual-cased
- На каких примерах проверялся инференс: ['Red Hat upgrades security Linux software maker Red Hat on Tuesday released an update to its enterprise product with security upgrades, support for IBM Power5 servers, new driver support and bug fixes.',
 'Fans have other favorites, but Busch has other ideas As Busch celebrated his 2004 Nextel Cup championship, most of the 80,000 fans turned their backs and walked out of Homestead-Miami Speedway, leaving the 26-year-old first-time champion to himself.',
 'Solaris 10 Supports High-Volume 64-Bit Computing Santa Clara, CA; November 22, 2004 -- Sun Microsystems and AMD are celebrating the one-year anniversary of their strategic partnership.',
 'Supercomputers Aid Hurricane Forecasting Thousands of miles from the rain and wind of Hurricane Ivan, a model of the storm swirls in the memory and processors of a supercomputer that can predict its likely course and strength.',
 'Siemens awarded 2G/3G network extension in Malaysia Malaysia #39;s Maxis Communications Bhd. (Maxis) has selected Siemens as one of its 2G and 3G/UMTS infrastructure suppliers. Under the terms of the three- year framework contract, the Siemens Communications Group ']
- Насколько результаты готовой модели оказались разумными: Точность ~57%, слабая уверенность (57-62%).
- Почему готовая модель подходит или не подходит под выбранную задачу: Не подходит - предсказания близки к случайным. Требуется fine-tuning для большей accuracy.

## 5. Fine-tuning и оценка

Опишите коротко и по делу:

- Модель для fine-tuning: rubert-tiny2
- Как была организована токенизация датасета: батчевая токенизация через map() с remove_columns(["text"]) и dynamic padding через DataCollatorWithPadding
- Максимальная длина (`max_length`): 128 токенов
- Batch size: 8 (train), 16 (eval)
- Максимальное число эпох: 3
- Optimizer / learning rate: AdamW / 2e-5
- Критерий выбора лучшего варианта: максимальный F1-macro на validation
- Какие метрики считались: accuracy, F1-macro, loss (train/validation/test)
  
## 6. Результаты

Ссылки на файлы в репозитории:

- Примеры предсказаний: `./artifacts/sample_predictions.csv`
- Матрица ошибок: `./artifacts/confusion_matrix.png`

Короткая сводка (6-10 строк):

- Итоговая `test_accuracy`: 0.9292
- Итоговая `test_f1_macro`: 0.9291
- Что показал инференс готовой модели: ~55% accuracy с низкой уверенностью.
- Что изменилось после fine-tuning: accuracy и уверенность сильно выросли.
- Какие классы распознаются лучше всего: Sports
- Какие классы модель путает чаще всего: Business и Sci/Tech

## 7. Анализ

(8-15 предложений)

Нужно прокомментировать:

- Разбор токенизации показал корректную работу BERT-токенизатора: текст преобразуется в input_ids с [CLS]/[SEP] токенами, attention_mask правильно маскирует padding, декодирование восстанавливает оригинал. Новости AG News идеально укладываются в 40-50 токенов без сильного усечения.
- Инференс готовой BERT-base-uncased дал всего ~55% точности с низкой уверенностью предсказаний — модель угадывала технические новости, но путала тематики из-за отсутствия дообучения на домене новостей. Это подтвердило необходимость fine-tuning.
- После дообучения RuBERT-tiny2 точность выросла — модель научилась четко различать темы новостей, но иногда путает Business с Sci/Tech из-за общих техно-терминов.
- Наиболее показательны ошибки в распознавании World новостей — модель недостаточно понимает геополитический контекст.
- Ограничения эксперимента: малая модель (33M параметров) может уступать большим BERT'ам; короткие тексты упрощают задачу; отсутствие аугментации данных ограничивает робастность; тест на английском при RuBERT требует проверки на мультиязычности.

## 8. Итоговый вывод

(3-7 предложений)

- Для классификации новостей как AG News рекомендую RuBERT-tiny2 как базовую модель: компактная (33M параметров), быстрая на CPU, дает 93% accuracy после 3 эпох fine-tuning.
- Главное про токенизацию: трансформеры требуют subword-токенизации с [CLS]/[SEP] токенами, attention_mask для padding и фиксированной длины (64-128 токенов оптимально для коротких текстов).
- Ключевое различие инференс и fine-tuning: готовая модель дает ~55% accuracy случайных предсказаний, fine-tuning поднимает до 93% за счет адаптации под домен и классы задачи.
